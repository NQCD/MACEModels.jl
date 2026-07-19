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
function cell_to_device(cell::PeriodicCell, device::CUDADevice, data_type::Type)
    return CuArray(data_type.(ustrip.(auconvert.(u"Å", cell.vectors)))), CuArray(cell.periodicity)
end
function cell_to_device(cell::PeriodicCell, device::CPUDevice, data_type::Type)
    return data_type.(ustrip.(auconvert.(u"Å", cell.vectors))), cell.periodicity
end
function cell_to_device(cell::InfiniteCell, device::CUDADevice, data_type::Type)
    return CuArray(zeros(data_type, 3,3)), CuArray(cell.periodicity)
end
function cell_to_device(cell::InfiniteCell, device::CPUDevice, data_type::Type)
    return zeros(data_type, 3, 3), cell.periodicity
end
# Move matrices to the correct device.
mtx_to_device(mtx::AbstractMatrix, device::CUDADevice) = CuArray(mtx)
mtx_to_device(mtx::AbstractMatrix, device::CPUDevice) = mtx

function mace_AtomicData_from_julia(
    model::MACEModel,
    atoms::Atoms,
    R::AbstractMatrix,
    cell::AbstractCell
)
    R_angstrom = ustrip.(auconvert.(u"Å", R))
    ab_structure = System(
        NQCBase.Structure(
            atoms,
            Array(R), # Need to pull this to CPU for conversion into something neighbour-listable.
            cell)
    )
    cell_device, pbc_device = cell_to_device(cell, model.model_device)
    positions_neighbourlistable = mtx_to_device(position(ab_structure, :) .|> ustrip, model.model_device)
    # Use generic NeighbourLists API that automatically selects device based on inputs.
    atomsbase_neighbourlist = neighbour_list(
        positions_neighbourlistable,
        model.cutoff_radius,
        cell_device,
        pbc_device,
    )
    # MACE reduces the neighbour list to remove self-interaction within the same cell. This mask should be applied to the edge indices to hide those interactions from MACE.
    non_self_edge_mask = mtx_to_device(
        .!(atomsbase_neighbourlist.i .== atomsbase_neighbourlist.j .& sum.(atomsbase_neighbourlist.S) .== 0),
        model.model_device
    )
    # Distance shifts and unit shifts need to be computed on CPU because I couldn't find a way of doing the dot product on each StaticArray in the CuArray on GPU.
    unit_shifts_cpu = reduce(hcat, Array(atomsbase_neighbourlist.S)) .|> model.data_type
    distance_shifts = deepcopy(unit_shifts_cpu)
    for idx in axes(distance_shifts, 2)
        distance_shifts[:,idx] = atomsbase_neighbourlist.C * unit_shifts_cpu[:, idx] # neighbourlist.C seems to always be on CPU
    end
    # Build one-hot encoding of atomic numbers for node attributes.
    onehot = vcat(
        [permutedims(atoms.numbers .== t) for t in model.atom_types |> sort]...
    ) .|> model.default_dtype
    # AtomicData constructor, mainly stolen from mace-torch:mace/data/atomic_data.py
    atomicdata = mace_data.AtomicData(
        edge_index=DLPack.share( # edge_index: [2,N] in Python, get rid of connectivity for self-interactions in the same cell.
            hcat(
                (atomsbase_neighbourlist.i .-1) .* non_self_edge_mask,
                (atomsbase_neighbourlist.j .-1) .* non_self_edge_mask,
            ),
            torch.from_dlpack,
        ),
        positions=DLPack.share( # positions: [N,3] in Python
            R_angstrom,
            torch.from_dlpack,
        ),
        shifts=DLPack.share(
            mtx_to_device(distance_shifts, model.model_device),
            torch.from_dlpack,
        ),
        unit_shifts=DLPack.share(
            mtx_to_device(unit_shifts_cpu, model.model_device),
            torch.from_dlpack,
        ),
        cell=DLPack.share(
            cell_device,
            torch.from_dlpack,
        ),
        node_attrs=DLPack.share(
            mtx_to_device(onehot, model.model_device),
            torch.from_dlpack,
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
        forces=DLPack.share(mtx_to_device(zeros(R), model.model_device), torch.from_dlpack),
        energy=nothing |> Py,
        stress=nothing |> Py,
        virials=nothing |> Py,
        dipole=nothing |> Py,
        charges=nothing |> Py,
        elec_temp=nothing |> Py,
        total_charge=nothing |> Py,
        polarizability=nothing |> Py,
        # total_spin=nothing,
        pbc=DLPack.share(pbc_device, torch.from_dlpack),
        # # density_coefficients=density_coefficients,
        # rcell=nothing,
        # volume=nothing,
        # fermi_level=nothing,
        # external_field=nothing,
    )
    return atomicdata
end
