module Ensemble

using MACEModels
using Distributed
using NQCModels
import SciMLBase

struct MultiProcessConfig
    runners::Vector{Int} # NQCDynamics workers
    evaluators::Vector{Int} # MACE inference worker
    remote_model::NQCMOdels.Model # Model to be evaluated
    model_listener::Function
    input_channels::Vector{RemoteChannel}
    output_channels::Vector{RemoteChannel}
end

"""
Ensemble algorithm which enforces the split between dynamics and inference workers provided. 

**Even when specifying this EnsembleAlgorithm, you still need to manually dispatch**

"""
struct CustomSplitDistributed<:SciMLBase.BasicEnsembleAlgorithm
    config::MultiProcessConfig
    pmap_args::Dict{Symbol, Any}
end

"""
    MultiProcessConfig(runners, evaluators, model_load_function::Function, positions_prototype; model_listener::Function=batch_evaluation_loop)

Configuration for distributed execution dynamics propagation and model evaluation. 

# Arguments
- `runners::Vector{Int}`: Process IDs of the NQCDynamics workers.
- `evaluators::Vector{Int}`: Process IDs of the model evaluators.
- `model_load_function::Function`: Function `()->x<:Model` that can load the model to be evaluated on any process. (i.e. not specific to a process)
- `model_listener::Function`: Function that listens for structures on the input channels and outputs the results on the output channels. Default is `batch_evaluation_loop`. `(model, input_channels, output_channels)->()`
- `positions_prototype`: Sample atomic positions to determine system size. 
"""
function MultiProcessConfig(runners, evaluators, model_load_function::Function, positions_prototype; model_listener::Function=batch_evaluation_loop)
    input_channels = [RemoteChannel(()->Channel{typeof(positions_prototype)}(1)) for _ in runners]
    output_channels = [RemoteChannel(()->Channel{EnergyForcesCache{eltype(positions_prototype), typeof(positions_prototype)}}(1)) for _ in runners]
    model = remotecall_fetch(model_load_function, first(evaluators))
    return MultiProcessConfig(runners, evaluators, model, model_listener, input_channels, output_channels)
end

"""
Containers for energy and forces, to be used in the RemoteModel.
"""
struct EnergyForcesCache{T, M}
    energy::T
    forces::M
end  


mutable struct RemoteModel <: NQCModels.AdiabaticModels.AdiabaticModel
    config::MultiProcessConfig # static
    mace_cache::EnergyForcesCache # mutable
    dofs::Int # static
    mobile_atoms # static
end


function mace_batch_predict(model::MACEModels.MACEModel, structures)
    @debug "Predicting batch of $(length(structures)) structures on process $(myid())"
    MACEModels.predict!(model, model.atoms, structures, model.cell)
    energies = MACEModels.get_energy_mean(model.last_eval_cache)
    forces = MACEModels.get_forces_mean(model.last_eval_cache)
    return (energies, forces)
end

"""
    batch_evaluation_loop(model<:Model, input_channels::Vector{RemoteChannel}, output_channels::Vector{RemoteChannel}; user_model_function::Function=mace_batch_predict, max_delay = 1000)

**This doesn't know how to use your model most efficiently, it only handles I/O between processes!**

To use this communication wrapper, you need to define a function that evaluates your model and returns a tuple of energies and forces, as shown in the example:

```
function user_model_function(model<:Model, structures::AbstractVector{AbstractMatrix{Number}})
    MACEModels.predict!(model, model.atoms, structures, model.cell)
    energies = MACEModels.get_energy_mean(model.last_eval_cache) # This is a Vector{Number} type
    forces = MACEModels.get_forces_mean(model.last_eval_cache) # This is a Vector{Matrix{Number}} type
    return (energies, forces)
end
```

It polls the input channels for new structures, and when it finds them, it evaluates them using the user-defined model function. 
The results are then put back into the correct output channels.

# Arguments
- `model::Model`: The model that will be used for evaluation.
- `input_channels::Vector{RemoteChannel}`: The input channels that will be used to receive structures.
- `output_channels::Vector{RemoteChannel}`: The output channels that will be used to send the results.
- `user_model_function::Function`: The user-defined function that will be used to evaluate the model.

"""
function batch_evaluation_loop(model::NQCModels.Model, input_channels::Vector{RemoteChannel}, output_channels::Vector{RemoteChannel}; user_model_function::Function=mace_batch_predict, max_delay = 1000)
    i = 1 # Start polling for input every 1ms, then increase delay for every unsuccessfull poll
    while true
        sleep(1e-3 * i)
        # Check for ready input channels, add structures to array
        to_process = findall(isready, input_channels)
        if isempty(to_process) # Nothing to do, increase delay before asking again. 
            i ≤ max_delay ? i += 1 : nothing
        else # Predict for all ready channels
            structures = [take!(channel) for channel in input_channels[to_process]]
            energies, forces = user_model_function(model, structures)
            number_type = eltype(eltype(input_channels[1])) # Ensure that the output type is the same as the input type. 
            !isa(energies, Vector) ? energies = [energies] : nothing
            energies = convert.(number_type, energies)
            !isa(forces, Vector) ? forces = [forces] : nothing
            forces = convert.(Matrix{number_type}, forces)
            # Put new outputs into output channels
            for (predictor_idx, channel_idx) in enumerate(to_process)
                put!(
                    output_channels[channel_idx], # Assign back to the channel that requested the prediction
                    EnergyForcesCache(energies[predictor_idx], 
                    forces[predictor_idx]
                    )
                )
                i = 1 # Reset delay
            end
        end
    end
end


"""
    RemoteModel(config::MultiProcessConfig, structure_prototype)

Creates a RemoteModel that can be used to evaluate structures on a remote process.

**The remote model must already be loaded, since model DoFs and mobile atoms are cached process-locally.**
"""
function RemoteModel(config::MultiProcessConfig, structure_prototype)
    RemoteModel(
        config,
        EnergyForcesCache(zero(eltype(structure_prototype)), zeros(eltype(structure_prototype), size(structure_prototype))),
        remotecall_fetch(() -> NQCModels.ndofs(config.remote_model), first(config.evaluators)),
        remotecall_fetch(() -> NQCModels.mobileatoms(config.remote_model, size(structure_prototype, 2)), first(config.evaluators))
    )
end

NQCModels.dofs(model::RemoteModel) = 1:model.dofs
NQCModels.ndofs(model::RemoteModel) = model.dofs

function NQCModels.potential(model::RemoteModel, R::Matrix{Float64})
    @debug "Evaluating potential on process $(myid())"
    put!(model.config.input_channels[findfirst(model.config.runners .== myid())], R)
    wait(model.config.output_channels[findfirst(model.config.runners .== myid())])
    model.mace_cache = take!(model.config.output_channels[findfirst(model.config.runners .== myid())])
    return model.mace_cache.energy
end

function NQCModels.derivative!(model::RemoteModel, D::Matrix{Float64}, R::Matrix{Float64})
    @debug "Evaluating derivative on process $(myid())"
    put!(model.config.input_channels[findfirst(model.config.runners .== myid())], R)
    wait(model.config.output_channels[findfirst(model.config.runners .== myid())])
    model.mace_cache = take!(model.config.output_channels[findfirst(model.config.runners .== myid())])
    D .-= model.mace_cache.forces
end

function SciMLBase.solve_batch(prob, alg, ensemblealg::CustomSplitDistributed, II, pmap_batch_size;
    kwargs...)
    runner_pool = CachingPool(ensemblealg.config.runners)

    batch_data = pmap(runner_pool, II, batch_size = pmap_batch_size, retry_delays=[1,1]) do i
        SciMLBase.batch_func(i, prob, alg; kwargs...)
    end

    SciMLBase.tighten_container_eltype(batch_data)
end

function start(config::MultiProcessConfig)
    for pid in config.evaluators
        remote_do(config.model_listener, pid, config.remote_model, config.input_channels, config.output_channels)
        @debug "Evaluator dispatched on process $pid"
    end
end

export RemoteModel, MultiProcessConfig, CustomSplitDistributed, start

end