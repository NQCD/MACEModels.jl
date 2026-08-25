module MACEModels

"""
This module contains an interface to the MACE machine learning interatomic potential, borrowing heavily from eval_configs.py and the MACE ase calculator.

MIT License

Copyright (c) 2022 ACEsuit/mace

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""

using PythonCall
using DLPack
using Unitful
using UnitfulAtomic
using NQCBase
using Statistics
using NQCModels: NQCModels
using CUDA
using StaticArrays

const torch = Ref{Py}()
const mace_data = Ref{Py}()
const mace_tools = Ref{Py}()
const mace_cli = Ref{Py}()
const numpy = Ref{Py}()
const importlib_meta = Ref{Py}()
const e3nn_util = Ref{Py}()

function __init__()
    torch[] = pyimport("torch")
    mace_data[] = pyimport("mace.data")
    mace_tools[] = pyimport("mace.tools")
    mace_cli[] = pyimport("mace.cli")
    numpy[] = pyimport("numpy")
    importlib_meta[] = pyimport("importlib.metadata")
    e3nn_util[] = pyimport("e3nn.util")
    # Output a warning if MACE or torch versions are unexpected.
    expected_mace_version = v"0.3.16"
    expected_torch_version = v"2.11.0"
    determined_mace_version = VersionNumber(pyconvert(String, importlib_meta[].version("mace-torch")))
    determined_torch_version = VersionNumber(pyconvert(String, importlib_meta[].version("torch")))
    for (pkg, exp, re) in zip(["mace-torch", "torch"], [expected_mace_version, expected_torch_version], [determined_mace_version, determined_torch_version])
        if re > exp
            @warn "Newer version of $pkg detected (version $re) than expected (version $exp). If you encounter issues, try downgrading to the expected version."
        end
    end
end

"""
Cache which stores the results of the last evaluation of the MACE model.
This is used to speed up concurrent energy and force evaluations, as well as to give access to ensemble predictions.

Use e.g. `get_energy_mean(MACEModel.last_eval_cache)` to access results of the last prediction made by the model.
"""
mutable struct MACEPredictionCache{T}
    energies::AbstractVector{<:AbstractVector{T}} # Model(Structure)
    forces::AbstractArray{T,3} # nDOFs * (∑_batch natoms) * models
    ptr::AbstractVector{Int} # n_structures + 1
    input_structures::AbstractVector
end

abstract type Device end
struct CPUDevice <: Device end
struct CUDADevice <: Device end
struct MetalDevice <: Device end

##  Move matrices to the correct device.
# CPU --> CUDA
mtx_to_device(mtx::AbstractArray, device::CUDADevice)::CuArray = CuArray(mtx)
# CUDA --> CPU
mtx_to_device(mtx::AnyCuArray, device::CPUDevice)::AbstractArray = Array(mtx)
# CPU --> CPU
mtx_to_device(mtx::AbstractArray, device::CPUDevice)::AbstractArray = mtx
# CUDA --> CUDA
mtx_to_device(mtx::AnyCuArray, device::CUDADevice)::AbstractArray = mtx


"""
MACE interface with support for ensemble of models and batch size selection for potentially faster inference.

DOFs currently hardcoded at 3.

# Fields
- `model_paths::Vector{String}`: Model paths to load. If multiple paths are given, an ensemble of models is used and mean energies/forces used to propagate dynamics.
- `models::Vector`: Loaded models.
- `device::Vector{String}`: Device to use for inference. If a single device is given, all models will be loaded on that device. If multiple devices are given, each model will be loaded on the corresponding device. Default is `"cpu"`.
- `default_dtype::Type{T}`: Default data type for inference. Default is `Float32`.
- `batch_size::Union{Int,Nothing}`: Batch size for inference. Default is `1`.
- `cutoff_radius::T`: Cutoff radius for the models.
- `last_eval_cache::MACEPredictionCache`: Cache for the last evaluation.
- `z_table`: Table of atomic numbers.
- `atoms::Atoms`: Atoms object for the system.
- `cell::PeriodicCell`: Periodic cell for the system.
- `ndofs::Int`: Number of degrees of freedom in the system.
- `mobile_atoms::Vector{Int}`: Indices of mobile atoms in the system.
"""
struct MACEModel{T, D, M} <: NQCModels.ClassicalModels.ClassicalModel
    model_paths::Vector{String}
    models::Vector
    device::Vector{String}
    torch_dtype::Py
    default_dtype::T
    dynamics_device::D
    model_device::M
    batch_size::Int
    cutoff_radius::AbstractFloat
    last_eval_cache::MACEPredictionCache
    atom_types::Vector{Int}
    atoms::Atoms
    cell::AbstractCell
    ndofs::Int
    mobile_atoms::Vector{Int}
end

NQCModels.ndofs(model::MACEModel) = 3
NQCModels.dofs(model::MACEModel) = Base.OneTo(3)
NQCModels.mobileatoms(model::MACEModel, n::Int) = model.mobile_atoms # Return all mobile atoms in the system.

# ToDo: Nice constructor for MACEModel and simpler input for single model.

"""
    MACEModel(
    atoms::Atoms,
    cell::AbstractCell,
    model_paths::Vector{String};
    device::Union{String,Vector{String}}="cpu",
    default_dtype::Type=Float32,
    batch_size::Int=1,
    mobile_atoms::Vector{Int}=collect(1:length(atoms)),
)
    model_paths::Vector{String},
    device::Union{String, Vector{String}}="cpu",
    default_dtype::Type=Float64,
    batch_size::Int=1,
    atoms,
    cell,
)

Interface to MACE machine learning interatomic potentials with support for ensemble of models and batch size selection for potentially faster inference.

# Arguments
- `atoms`: Atoms object for the system. `predict!` can be called with a Vector of different Atoms objects to evaluate different structures.
- `cell`: Cell object for the system. `predict!` can be called with a Vector of different Cell objects to evaluate different structures.
- `model_paths::Vector{String}`: Model paths to load. If multiple paths are given, an ensemble of models is used and mean energies/forces used to propagate dynamics.
- `device::Union{String, Vector{String}}`: Device to use for inference. If a single device is given, all models will be loaded on that device. If multiple devices are given, each model will be loaded on the corresponding device. Default is `"cpu"`.
- `default_dtype::Type`: Default data type for PyTorch (usually Float32)
- `batch_size::Int`: Batch size for inference. Can be set to `nothing` to always adapt batch size to the number of structures provided. Ring-polymer methods benefit from this.
- `mobile_atoms::Vector{Int}`: Indices of mobile atoms in the system. Default is all atoms.
"""
function MACEModel(
    atoms::Atoms,
    cell::AbstractCell,
    model_paths::Vector{String};
    device::Union{String,Vector{String}}="cpu",
    default_dtype::Type=Float32, # for better compatibility
    batch_size::Int=1,
    mobile_atoms::Vector{Int}=collect(1:length(atoms)),
    use_CuEquivariance::Bool=false,
    use_OpenEquivariance::Bool=false,
    dynamics_on::Device = CPUDevice(),
)
    # Assign device to all models if only one device is given
    isa(device, String) ? device = [device for _ in 1:length(model_paths)] : nothing
    # Check selected device types are available and assign model_device
    model_device = nothing
    for dev in device
        if split(dev, ":")[1] == "cuda"
            if pyconvert(Bool, torch[].backends.cuda.is_built())
                @debug "CUDA device available, using GPU."
                model_device = CUDADevice()
            else
                @warn "CUDA device not available, falling back to CPU."
                dev = "cpu"
                model_device = CPUDevice()
            end
            if length(split(dev, ":")) == 2
                @warn "Running on a selected GPU is currently unsupported. Falling back to cuda:0. See https://github.com/NQCD/MACEModels.jl/issues/13 for more information"
                dev = "cuda:0"
                model_device = CUDADevice()
            end
        elseif dev == "mps"
            if pyconvert(Bool, torch[].backends.mps.is_built())
                @debug "MPS device available, using GPU."
                model_device = MetalDevice()
            else
                @warn "MPS device not available, falling back to CPU."
                dev = "cpu"
                model_device = CPUDevice()
            end
        else
            @debug "CPU selected as torch.device"
            dev = "cpu"
            model_device = CPUDevice()
        end
    end
    # Set default dtype for torch
    # Conversion from Julia dtypes to PyTorch dtypes.
    dtypes_julia_python = Dict{Type,Any}(Float32 => torch[].float32, Float64 => torch[].float64)
    torch_dtype = dtypes_julia_python[default_dtype]
    torch[].set_default_dtype(torch_dtype)

    # Load MACE models
    models = []
    for (i, file_path) in enumerate(model_paths)
        try
            model = torch[].load(f=file_path, map_location=device[i] , weights_only=false)
            model = model.to(device[i])
            model = use_CuEquivariance ? mace_cli.convert_e3nn_cueq.run(model, device=device[i]).to(device) : model
            model = use_OpenEquivariance ? mace_cli.convert_e3nn_oeq.run(model, device=device[i]).to(device) : model
            model = e3nn_util[].jit.compile(model)
            # Check if any rogue atoms contained in base structure
            any(sort(unique(atoms.numbers)) ∉ Vector(from_dlpack(model.atomic_numbers.contiguous()))) || throw(ArgumentError("Example structure contains atom types not present in MACE model $(i)."))
            # Model is safe to add to ensemble
            push!(models, model)
        catch e
            throw(e)
            @error "Error loading model from file: $file_path"
        end
    end

    # Check cutoff radii are identical
    cutoff_radii = [copy(Array(from_dlpack(model.r_max))[]) for model in models]
    if length(unique(cutoff_radii)) > 1
        @warn "Cutoff radii are not identical for all models."
    end
    cutoff_radius = convert(default_dtype, unique(cutoff_radii)[1])

    # Build z-table
    # Hardcoding this to the model so we can handle input structures using a subset of all learned atom types.
    atom_numbers = Vector(from_dlpack(models[1].atomic_numbers.contiguous()))

    # Freeze parameters
    for model in models
        for parameter in model.parameters()
            parameter.requires_grad = false
        end
    end

    # Initialise an evaluation cache
    starter_mace_cache = MACEPredictionCache(
        mtx_to_device([[convert(default_dtype, 1.0)]], dynamics_on), # Energies
        mtx_to_device([convert(default_dtype, 1.0);;;], dynamics_on), # Forces
        mtx_to_device([0,1], dynamics_on), # Pointer
        mtx_to_device([[convert(default_dtype, 1.0);;]], dynamics_on), # Structures
    )

    return MACEModel(model_paths, models, device, torch_dtype, default_dtype(1.0), dynamics_on, model_device, batch_size, cutoff_radius, starter_mace_cache, atom_numbers, atoms, cell, 3, mobile_atoms)
end

function Base.show(io::IO, model::MACEModel)
    print(
        io,
        "MACEModel{$(model.default_dtype)} with $(length(model.models)) models:\n",
        "\tDevice(s): $(model.device) \n",
        "\tBatch size: $(model.batch_size) \n"
    )
end

include("mace_structures.jl")

"""
    predict!(
    mace_interface::MACEModel,
    atoms::Union{Vector{<:Atoms}, Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell}, AbstractCell},
    )

Evaluate the MACE model on a set of structures if they haven't already been evaluated.
The results are stored in `mace_interface.last_eval_cache`.

Use `methodswith(MACEPredictionCache)` to see what data is provided from the results

# Arguments
`mace_interface`: `MACEModel` to evaluate with.

`atoms`: Atoms object for the structures passed for evaluation.
Can be either a vector of different Atoms objects or a single Atoms object.

`R`: Vector of structures to evaluate.

`cell`: Cell object for the structures passed for evaluation.
Can be either a vector of different Cell objects or a single Cell object.
"""
function predict!(
    mace_interface::MACEModel{T,D,M},
    atoms::Union{Vector{<:Atoms},Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell},AbstractCell},
) where {T,D,M}
    # Reset default dtype for torch in case using another model changed it.
    torch[].set_default_dtype(mace_interface.torch_dtype)

    if R != mace_interface.last_eval_cache.input_structures # Only predict if working on new structures
        n_batches = length(R) / mace_interface.batch_size |> ceil |> Int # Number of batches to pass through MACE. ≥ 1
        structure_slices = Iterators.partition(1:length(R), mace_interface.batch_size) |> collect

        # atoms and cell must always be vectors for each structure.
        isa(cell, AbstractCell) ? cell = [cell for _ in 1:length(R)] : nothing # Always have atoms, positions and cell for each structure
        isa(atoms, Atoms) ? atoms = [atoms for _ in 1:length(R)] : nothing

        # Allocate results arrays for prediction
        mace_interface.last_eval_cache.input_structures = deepcopy(R)
        mace_interface.last_eval_cache.energies = [mtx_to_device(zeros(T, length(R)), D()) for i in eachindex(mace_interface.models)]
        full_ptr = vcat(1, [length(at.masses) for at in atoms])|> cumsum
        mace_interface.last_eval_cache.ptr = mtx_to_device(full_ptr, D())
        mace_interface.last_eval_cache.forces = mtx_to_device(zeros(T, mace_interface.ndofs, sum([length(at) for at in atoms]), length(mace_interface.models)), D())
        force_slice_start = full_ptr[1]
        force_slice_end = full_ptr[min(mace_interface.batch_size, length(R))+1]-1

        # For each batch that needs running, build the corresponding graph(s).
        for batch_idx in 1:n_batches
            # Since all models must be on the same GPU, the dict can be reused across models.
            batch_parts = [to_julia_dict(
                mace_interface,
                atoms[structure_idx],
                R[structure_idx],
                cell[structure_idx],
            ) for structure_idx in structure_slices[batch_idx]]
            # Shorthand to route Arrays to correct device and torch representation.
            torch_tensor_device(x) = DLPack.share(mtx_to_device(x, M()), torch[].from_dlpack)
            # Combine together into a single graph in Torch representation by concatenating along the highest dimension.
            the_batch = Dict{String, Py}()
            for k in keys(batch_parts[1])
                the_batch[k] = torch_tensor_device(cat([d[k] for d in batch_parts]...; dims=length(size(batch_parts[1][k]))))
            end
            # Set batch and ptr information
            batch_idx_vector = vcat([repeat([N-1], length(at.masses)) for (N,at) in enumerate(atoms[structure_slices[batch_idx]])]...)
            the_batch["batch"] = batch_idx_vector |> torch_tensor_device # length Natoms * batch size, must be a torch.int type.
            ptr = vcat(1, [length(atoms[st_idx]) for st_idx in structure_slices[batch_idx]]) |> cumsum
            the_batch["ptr"] = ptr .-1 |> torch_tensor_device # length batch size + 1, must be a torch.int type.
            # Now the batch should be set up correctly.
            # for k in keys(the_batch)
            #     println("Key: $k, PythonType: $(the_batch[k].dtype), TorchSize: $(the_batch[k].size()), TorchDevice: $(the_batch[k].device)")
            # end
            n_batches > 1 ? println("Pointer start: $(force_slice_start), Pointer end: $(force_slice_end)") : nothing
            # Evaluate each model
            for (model_index, model) in enumerate(mace_interface.models)
                model_output = Py(model.forward(the_batch |> PyDict)) # Default kwargs specify training=false, compute_stress=false, compute_node_energy=false, these must be set to not mess things up.
                # Shared memory refs to the model outputs.
                # These need moving to the CPU for now due to scalar indexing. I need to think about how to solve this separately.
                mace_interface.last_eval_cache.energies[model_index] = mtx_to_device(from_dlpack(model_output["energy"].contiguous().detach()), D())
                mace_interface.last_eval_cache.forces[:,force_slice_start:force_slice_end,model_index] .= mtx_to_device(from_dlpack(model_output["forces"].contiguous().detach()), D())
            end
            force_slice_start = batch_idx < n_batches ? full_ptr[batch_idx*mace_interface.batch_size+1] : force_slice_start
            force_slice_end = batch_idx < n_batches ? full_ptr[min((batch_idx+1)*mace_interface.batch_size, length(R))+1]-1 : force_slice_end
        end
    end
end

"""
    predict(
    mace_interface::MACEModel,
    atoms::Union{Vector{<:Atoms}, Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell}, AbstractCell},
    )

Evaluate the MACE model on a set of structures if they haven't already been evaluated.
The results are returned as a `MACEPredictionCache`.

Use `methodswith(MACEPredictionCache)` to see what data can be provided from the results.

# Arguments
`mace_interface`: `MACEModel` to evaluate with.

`atoms`: Atoms object for the structures passed for evaluation.
Can be either a vector of different Atoms objects or a single Atoms object.

`R`: Vector of structures to evaluate.

`cell`: Cell object for the structures passed for evaluation.
Can be either a vector of different Cell objects or a single Cell object.
"""
function predict(
    mace_interface::MACEModel,
    atoms::Union{Vector{<:Atoms},Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell},AbstractCell},
)
    predict!(mace_interface, atoms, R, cell)
    return deepcopy(mace_interface.last_eval_cache)
end

# Methods using MACEPredictionCache that convert MACE outputs to NQCD's atomic unit scheme. Snip length 1 caches to the basic outputs instead of unnecessary vector wrapping.

"""
    get_energy_mean(mace_cache::MACEPredictionCache)

Returns the mean potential energy of the structures stored in the evaluation cache.

Energy is returned in units of **Hartree**.
"""
function get_energy_mean(mace_cache::MACEPredictionCache)
    mean_energies = zero(mace_cache.energies[1])
    N = length(mace_cache.energies)
    for index in 1:N
        mean_energies .+= austrip.(mace_cache.energies[index] .* u"eV") # Energy is given in eV
    end
    mean_energies ./= N
    if length(mean_energies) == 1
        return Array(mean_energies) |> first
    else
        return mean_energies # Return in Hartree
    end
end

"""
    get_energy_variance(mace_cache::MACEPredictionCache)

Returns the variance of the potential energy in **Hartree²**.
"""
function get_energy_variance(mace_cache::MACEPredictionCache)
    mean_energies = zeros(eltype(mace_cache.energies[1]), length(mace_cache.energies[1]), length(mace_cache.energies))
    for index in eachindex(mace_cache.energies)
        mean_energies[:, index] .= mace_cache.energies[index]
    end
    mean_energies .= dropdims(var(austrip.(mean_energies .* u"eV"); dims=2); dims=2) # Energy is given in eV
    if length(mean_energies) == 1
        return mean_energies[1]
    else
        return mean_energies # Return in Hartree^2
    end
end

"""
    get_energy_ensemble(mace_cache::MACEPredictionCache)

Returns the potential energy evaluated by each model in the ensemble in units of **Hartree**.
"""
function get_energy_ensemble(mace_cache::MACEPredictionCache)
    ensemble_energies = zeros(eltype(mace_cache.energies[1], length(mace_cache.energies), length(mace_cache.energies[1])))
    for index in eachindex(mace_cache.energies)
        ensemble_energies[index, :] = @. austrip(mace_cache.energies[index] * u"eV") # Energy is given in eV
    end
    if size(ensemble_energies, 2) == 1
        return ensemble_energies[:,1]
    else
        return ensemble_energies # Return in Hartree
    end
end

"""
    get_forces_mean(mace_cache::MACEPredictionCache)

Returns the mean forces of the structures stored in the evaluation cache.
Forces are returned in units of **Hartree/Bohr**.
Warning: This function does not respect mobileatoms constraints. Forces for frozen atoms must be manually set to 0
"""
function get_forces_mean(mace_cache::MACEPredictionCache)
    mean_forces = dropdims(mean(austrip.(mace_cache.forces .* u"eV/Å"); dims=3); dims=3) # Force is given in eV/Å and returns in Hartree / Bohr
    return length(mace_cache.ptr) == 2 ? mean_forces : [mean_forces[:, left:right] for (left:right) in zip(mace_cache.ptr[1:end-1], mace_cache.ptr[2:end])]
end

"""
    get_forces_variance(mace_cache::MACEPredictionCache)

Returns the force variance of the structures stored in the evaluation cache.
Forces are returned in units of **Hartree²/Bohr²**.
Warning: This function does not respect mobileatoms constraints. Forces for frozen atoms must be manually set to 0
"""
function get_forces_variance(mace_cache::MACEPredictionCache)
    var_forces = dropdims(var(austrip.(mace_cache.forces .* u"eV/Å"); dims=3); dims=3) # Force is given in eV/Å and returns in Hartree / Bohr
    return length(mace_cache.ptr) == 2 ? var_forces : [var_forces[:, left:right] for (left:right) in zip(mace_cache.ptr[1:end-1], mace_cache.ptr[2:end])]
end

"""
    get_forces_ensemble(mace_cache::MACEPredictionCache)

Returns the forces evaluated by each model in the ensemble in units of **Hartree/Bohr**.

Warning: This function does not respect mobileatoms constraints. Forces for frozen atoms must be manually set to 0
"""
function get_forces_ensemble(mace_cache::MACEPredictionCache)
    return austrip.(mace_cache.forces .* u"eV/Å")
end

# ToDo: Potential and derivative for a single structure
# Passthrough for NQCDynamics Calculators
NQCModels.potential(model::MACEModel, R::AbstractMatrix) = NQCModels.potential(model, model.atoms, R, model.cell)
NQCModels.derivative(model::MACEModel, R::AbstractMatrix) = NQCModels.derivative(model, model.atoms, R, model.cell)
NQCModels.derivative!(model::MACEModel, D::AbstractMatrix, R::AbstractMatrix) = NQCModels.derivative!(model, D, model.atoms, R, model.cell)

# Single structure evaluation with custom atoms and cell.
function NQCModels.potential(model::MACEModel, atoms::Atoms, R::AbstractMatrix, cell::AbstractCell)
    # Evaluate model
    predict!(model, atoms, [R], cell)
    # Return potential (mean is trivial here)
    return get_energy_mean(model.last_eval_cache)
end


function NQCModels.derivative(model::MACEModel, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell,PeriodicCell})
    # Evaluate model
    D = zeros(eltype(R), size(R))
    predict!(model, atoms, [R], cell)
    # Return derivative (mean is trivial)
    D[:, model.mobile_atoms] .-= @views get_forces_mean(model.last_eval_cache)[:, model.mobile_atoms]
    return D
end

function NQCModels.derivative!(model::MACEModel, D::AbstractMatrix, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell,PeriodicCell})
    # Evaluate model
    predict!(model, atoms, [R], cell)
    # Return derivative
    D[:, model.mobile_atoms] .-= @views get_forces_mean(model.last_eval_cache)[:, model.mobile_atoms]
end

# ToDo: Potential and derivative for multiple structures
"""
    NQCModels.potential(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::AbstractCell)

This variant of `NQCModels.potential` can make use of batch evaluation to speed up
inference for multiple structures.
"""
function NQCModels.potential(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::AbstractCell)
    # Evaluate model
    predict!(model, atoms, R, cell)
    # Return potential (mean is trivial here)
    return get_energy_mean(model.last_eval_cache)
end

"""
    NQCModels.derivative(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::Union{InfiniteCell, PeriodicCell})

This variant of `NQCModels.derivative` can make use of batch evaluation to speed up
inference for multiple structures.
"""
function NQCModels.derivative(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::Union{InfiniteCell,PeriodicCell})
    # Evaluate model
    D = zeros(eltype(R), size(R))
    predict!(model, atoms, R, cell)
    # Return derivative (mean is trivial)
    D_full = get_forces_mean(model.last_eval_cache)
    for i in axes(D, 3)
        D[i][:, model.mobile_atoms] .-= @views D_full[i][:, model.mobile_atoms]
    end
    return D
end

"""
    NQCModels.derivative(model::MACEModel, atoms::Atoms, R::AbstractArray{T,3}, cell::Union{InfiniteCell, PeriodicCell})

This variant of `NQCModels.derivative` can make use of batch evaluation to speed up
inference for multiple structures.
"""
function NQCModels.derivative!(model::MACEModel, atoms::Atoms, D::AbstractArray{T,3}, R::Vector{<:AbstractMatrix}, cell::Union{InfiniteCell,PeriodicCell}) where {T}
    # Evaluate model
    predict!(model, atoms, R, cell)
    # Return derivative (mean is trivial)
    for i in axes(D, 3)
        D[:, model.mobile_atoms, i] .-= @views get_forces_mean(model.last_eval_cache)[i][:, model.mobile_atoms]
    end
end

include("MACEModelsMPI.jl")

export MACEModel, MACEPredictionCache, Ensemble

end
