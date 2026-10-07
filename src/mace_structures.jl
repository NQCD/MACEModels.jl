using NeighbourLists

"""
    mace_configuration_from_nqcd_configuration(
    atoms::Atoms,
    cell::Union{InfiniteCell, PeriodicCell},
    R::AbstractMatrix,
)

Converter into a single mace.data.utils.Configuration to make use of MACE's data loading.
"""
function mace_configuration_from_nqcd_configuration(
    atoms::Atoms,
    cell::AbstractCell,
    R::AbstractMatrix;
    dtype::Type=Float64,
    head_name::String="Default"
)
    #! Removed positions type conversion to check if it affects prediction
    if isa(cell, InfiniteCell)
        pbc = zeros(Bool, size(R, 1))
        cell_array = zeros(dtype, size(R, 1), size(R, 1))
    elseif isa(cell, PeriodicCell)
        pbc = cell.periodicity
        cell_array = permutedims(Matrix{eltype(R)}(ustrip.(auconvert.(u"Å", cell.vectors))), (2, 1))
    end

    ase_positions = permutedims(ustrip.(auconvert.(u"Å", R)), (2, 1))

    config = mace_data[].utils.Configuration(
        atomic_numbers=PyList(atoms.numbers), # needs to be a list
        positions=numpy[].array(ase_positions), # Convert from atomic units to Ångström
        properties=Dict{String,Any}(
            "energy" => Py(zero(eltype(R))), # scalar
            "forces" => numpy[].array(zeros(eltype(R), size(R'))), # N_atoms * N_dofs
            #     "stress" => pybuiltins.None,
            #     "virials" => pybuiltins.None,
            #     "dipole" => pybuiltins.None,
            #     "charges" => pybuiltins.None,
        ),
        head=Py("Default"),
        weight=Py(one(eltype(R))),
        property_weights=Dict(
        #    "energy_weight" => Py(one(eltype(R))),
        #    "forces_weight" => Py(one(eltype(R))),
        #    "stress_weight" => Py(one(eltype(R))),
        #    "virials_weight" => Py(one(eltype(R))),
        ),
        config_type=Py("Default"),
        pbc=Py(pbc),
        cell=numpy[].array(cell_array),
    )
    return Py(config)
end

# mace.data.AtomicData recreator, generic array implementation that can use GPU arrays too.
# Ensure AtomicData are created with pointers to DLPack versions of everything so the GC is done properly.

# Translate cells to correct format
function cell_to_device(cell::PeriodicCell, device::CUDADevice, data_type::Type)::Tuple{<:CuArray, <:CuArray}
    return CuArray(data_type.(ustrip.(auconvert.(u"Å", cell.vectors)))), CuArray(cell.periodicity)
end
function cell_to_device(cell::PeriodicCell, device::CPUDevice, data_type::Type)::Tuple{<:AbstractArray, <:AbstractArray}
    return data_type.(ustrip.(auconvert.(u"Å", cell.vectors))), cell.periodicity
end
function cell_to_device(cell::InfiniteCell, device::CUDADevice, data_type::Type)::Tuple{<:CuArray, <:CuArray}
    return CuArray(zeros(data_type, 3,3)), CuArray(cell.periodicity)
end
function cell_to_device(cell::InfiniteCell, device::CPUDevice, data_type::Type)::Tuple{<:AbstractArray, <:AbstractArray}
    return zeros(data_type, 3, 3), cell.periodicity
end

function make_neighbourlist(
    model::MACEModel{T,D,M},
    atoms::Atoms,
    R::AbstractMatrix,
    cell::AbstractCell,
) where {T, D, M}
    R_angstrom = Matrix{T}(ustrip.(auconvert.(u"Å", R))) # D
    cell_device, pbc_device = cell_to_device(cell, M(), T)
    positions_neighbourlistable = mtx_to_device([SVector{3}(col) for col in eachcol(R_angstrom)], M())
    # The generic NeighbourLists API is type-unstable due to how it processes the choice of lazy-loading vs. no lazy-loading.
    # Since we must fully allocate the neighbour list anyway, we might as well skip it and use the non-API functions which are type-stable.
    # This might lead to bugs in future if they change anything though. Implemented with NeighbourLists v0.6.2
    atomsbase_neighbourlist = NeighbourLists.materialize_pairlist(
        NeighbourLists.build_cell_list(
            positions_neighbourlistable,
            model.cutoff_radius,
            auconvert.(u"Å", cell.vectors) .|> ustrip,
            cell.periodicity,
            backend = NeighbourLists.get_array_backend(positions_neighbourlistable)
        );
        backend = NeighbourLists.get_array_backend(positions_neighbourlistable)
    ) # Outputs already allocated on M
    ab_i = atomsbase_neighbourlist.i # Already allocated on M
    ab_j = atomsbase_neighbourlist.j # Already allocated on M
    ab_S = mtx_to_device(reinterpret(reshape, Int32, atomsbase_neighbourlist.S), M()) # Reinterpretation needed on M
    return ab_i, ab_j, ab_S, cell_device, pbc_device
end

function to_julia_dict(
    model::MACEModel{T,D,M},
    atoms::Atoms,
    R::AbstractMatrix,
    cell::AbstractCell,
) where {T,D,M}
    R_angstrom = Matrix{T}(ustrip.(auconvert.(u"Å", R))) # D
    ab_i, ab_j, ab_S, cell_device, pbc_device = make_neighbourlist(model, atoms, R, cell)
    non_self_edge_mask = .!(ab_i .== ab_j .&& sum(ab_S) == 0)
    non_self_edge_mask = mtx_to_device(
        non_self_edge_mask,
        M(),
    )
    # Dot product with the cell to turn into distance shifts.
    # unit_shifts_cpu = Matrix{T}(reduce(hcat, Vector(atomsbase_neighbourlist.S)))
    unit_shifts_cpu = copy(ab_S) # Maybe this variant is faster?
    distance_shifts = cell_device * unit_shifts_cpu
    # Build one-hot encoding of atomic numbers for node attributes.
    onehot = Matrix{T}(vcat(
        [permutedims(atoms.numbers .== t) for t in model.atom_types |> sort]...
    ))
    # Now use the Batch dict representation for mace data, but without batch and ptr keys since they may need changing.
    # Also keep all arrays initialised in Julia for now so they can be concatenated easily.
    dict_representation = Dict{String, Any}(
        "head" => Int[0], # one per structure
        "cell" => cell_device,
        "edge_index" => hcat(# edge_index: [2,N] in Python, get rid of connectivity for self-interactions in the same cell.
            (ab_i .-1) .* non_self_edge_mask,
            (ab_j .-1) .* non_self_edge_mask,
        ),
        "energy" => zeros(T, 1),
        "forces" => zero(R),
        "node_attrs" => onehot,
        "positions" => R_angstrom, # positions: [N,3] in Angstrom
        "shifts" => distance_shifts,
        "unit_shifts" => unit_shifts_cpu,
        "weight" => ones(T, 1),
    )
    return dict_representation
end

function mace_AtomicData_from_julia(
    model::MACEModel{T,D,M},
    atoms::Atoms,
    R::AbstractMatrix,
    cell::AbstractCell
) where {T,D,M}
    ab_i, ab_j, ab_S, cell_device, pbc_device = make_neighbourlist(model, atoms, R, cell)
    # MACE reduces the neighbour list to remove self-interaction within the same cell. This mask should be applied to the edge indices to hide those interactions from MACE.
    non_self_edge_mask = .!(ab_i .== ab_j .&& sum(ab_S) == 0)
    non_self_edge_mask = mtx_to_device(
        non_self_edge_mask,
        M(),
    )
    # Dot product with the cell to turn into distance shifts.
    # unit_shifts_cpu = Matrix{T}(reduce(hcat, Vector(atomsbase_neighbourlist.S)))
    unit_shifts_cpu = copy(ab_S) # Maybe this variant is faster?
    distance_shifts = cell_device * unit_shifts_cpu
    # Build one-hot encoding of atomic numbers for node attributes.
    onehot = Matrix{T}(vcat(
        [permutedims(atoms.numbers .== t) for t in model.atom_types |> sort]...
    ))
    # AtomicData constructor, mainly stolen from mace-torch:mace/data/atomic_data.py
    atomicdata = mace_data[].AtomicData(
        edge_index=DLPack.share( # edge_index: [2,N] in Python, get rid of connectivity for self-interactions in the same cell.
            hcat(
                (ab_i .-1) .* non_self_edge_mask,
                (ab_j .-1) .* non_self_edge_mask,
            ),
            torch[].from_dlpack,
        ),
        positions=DLPack.share( # positions: [N,3] in Python
            mtx_to_device(R_angstrom, M()),
            torch[].from_dlpack,
        ),
        shifts=DLPack.share(
            mtx_to_device(distance_shifts, M()),
            torch[].from_dlpack,
        ),
        unit_shifts=DLPack.share(
            mtx_to_device(unit_shifts_cpu, M()),
            torch[].from_dlpack,
        ),
        cell=DLPack.share(
            cell_device,
            torch[].from_dlpack,
        ),
        node_attrs=DLPack.share(
            mtx_to_device(onehot, M()),
            torch[].from_dlpack,
        ),
        weight=nothing,
        head=nothing,
        energy_weight=nothing |> Py,
        forces_weight=nothing |> Py,
        stress_weight=nothing |> Py,
        virials_weight=nothing |> Py,
        dipole_weight=nothing |> Py,
        charges_weight=nothing |> Py,
        polarizability_weight=nothing |> Py,
        forces=DLPack.share(mtx_to_device(zero(R), M()), torch[].from_dlpack),
        energy=nothing |> Py,
        stress=nothing |> Py,
        virials=nothing |> Py,
        dipole=nothing |> Py,
        charges=nothing |> Py,
        elec_temp=nothing |> Py,
        total_charge=nothing |> Py,
        polarizability=nothing |> Py,
        # total_spin=nothing,
        pbc=DLPack.share(pbc_device, torch[].from_dlpack),
        # # density_coefficients=density_coefficients,
        # rcell=nothing,
        # volume=nothing,
        # fermi_level=nothing,
        # external_field=nothing,
    )
    return atomicdata
end
