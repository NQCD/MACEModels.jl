### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 74318bde-836a-11f1-884f-2f1ebdb1b6fd
import Pkg

# ╔═╡ 3bddf6ec-a18f-4497-b62f-a66c44bac1b9
Pkg.activate(".")

# ╔═╡ 4183f222-0ae5-4d8a-8513-a2364ca9ab72
using MACEModels, CairoMakie, NQCDynamics, NQCDInterfASE

# ╔═╡ 57b17f64-0fd9-4f80-8786-964356b67986
using CUDA

# ╔═╡ d879518d-3eec-41bc-bf2b-2cd7a736fce4
using NQCModels

# ╔═╡ 9ab36373-e3c3-4e2f-aeba-440e3506d3e7
begin # Loading the ASE calculator
    ENV["JULIA_PYTHONCALL_EXE"]="/fs/home/alex/programs/pyenv/versions/mace_release/bin/python"
    using PythonCall
    mace_calcs = pyimport("mace.calculators")
    ase_io = pyimport("ase.io")
    function make_mace_model(device::String)
        mace_calc = mace_calcs.MACECalculator(
            model_paths=["../test/test_model/MACE_model_swa.model"],
            device=device,
            default_dtype="float32" # Hardcoded for this particular model.
        )
        ase_structure = ase_io.read("../test/test_model/h2cu_diffusion_desorption_validation.xyz", index=0)
        ase_structure.calc = mace_calc
        mace_model_ase = ClassicalASEModel(ase_structure)
        return mace_model_ase
    end
    mace_torch_cpu = make_mace_model("cpu")
    mace_torch_cuda = make_mace_model("cuda")
end

# ╔═╡ 16a1c475-985b-4ae3-9b6a-ce5d665aeffd
using ProgressLogging

# ╔═╡ 5dd10225-6b2a-4365-9f6e-fb1075b9d5bb
using PProf

# ╔═╡ ca53c48d-23f6-41be-863b-e51d4032c27d
using Profile

# ╔═╡ 03c97cd3-8299-4a52-8718-c29df30a2418
using Unitful, UnitfulAtomic, Statistics

# ╔═╡ a29de054-7fc3-42e7-9838-662bd06347a5
# ╠═╡ disabled = true
#=╠═╡
Pkg.add("NQCModels")
  ╠═╡ =#

# ╔═╡ c9fd0e07-0cd2-44ed-bcab-2ae9d4a04df3
md"# Benchmarking `mace-torch` via PythonCall vs. MACEModels.jl"

# ╔═╡ 5a18508e-1763-4b47-9947-bb84e32f393e
md"""
## MD comparison

This round is benchmarked by evaluating the same set of 3000 structures with a batch size of 1 to emulate no benefit from batching calculations. 

Due to the Julia-based neighbour list generation with `NeighbourLists.jl` in MACEModels, it should hopefully be a bit faster. 
"""

# ╔═╡ e9c2865e-f53a-4971-9c01-e39923757ecb
structures = first(read_extxyz("../test/test_model/h2cu_diffusion_desorption_validation.xyz"), 3000)

# ╔═╡ 77ef40f1-383a-414d-9a92-42bdc2423350
function make_macemodels_model(device::String, batch_size::Int)
    mace_model = MACEModel(
        structures[1],
        ["../test/test_model/MACE_model_swa.model"],
        device = device,
        default_dtype=Float32,
        batch_size = batch_size,
    )
    # Force the JIT compilation to have definitely run before evaluating. 
    MACEModels.predict(mace_model, [structures[1]])
    return mace_model
end

# ╔═╡ 45ec9db6-2208-4ca5-95a4-a0b55a6e26f7
begin
    macemodels_b1_cpu = make_macemodels_model("cpu", 1)
    macemodels_b1_cuda = make_macemodels_model("cuda", 1)
end

# ╔═╡ 9f627a5e-a0d8-4063-b0f0-34ae735130e2
Float32(1.0) |> typeof

# ╔═╡ 59d8c1db-cf28-42fa-987f-e02556a9b2d1
# ╠═╡ disabled = true
#=╠═╡
@code_warntype MACEModels.mace_AtomicData_from_julia(
	macemodels_b1_cpu,
	structures[1].atoms,
	structures[1].positions |> cu,
	structures[1].cell,
)
  ╠═╡ =#

# ╔═╡ 762d7a03-cde0-4db8-9100-b5c904ba3e8b
# ╠═╡ disabled = true
#=╠═╡
@progress out = [NQCModels.derivative(mace_torch_cpu, s.positions) for s in structures]
  ╠═╡ =#

# ╔═╡ ec10dbb6-ac35-4132-9b0b-eb572c0cbca3
t1=time_ns()

# ╔═╡ e125b99b-94c7-4be4-b8ef-e5010d78907e
t2 = time_ns()

# ╔═╡ 57832997-bc68-4264-bb6c-811a82aaf14a
function evaluate_model(model, st, label = "Model")
    results = [zero(s.positions) for s in st]
    start_time = time_ns()
    @progress name=label for idx in eachindex(st)
        NQCModels.derivative!(model, results[idx], st[idx].positions)
    end
    end_time = time_ns()
    time_diff = Float64(end_time-start_time) / 1e6 # time in ms
    return time_diff, results
end

# ╔═╡ 2378c983-075b-418e-aa8c-dbe637057993
md_benchmark = Dict(
    Symbol("MACEModels CPU") => evaluate_model(macemodels_b1_cpu, first(structures, 3000), "MACEModels CPU"),
    Symbol("MACEModels GPU") => evaluate_model(macemodels_b1_cuda, first(structures, 3000), "MACEModels GPU"),
    Symbol("mace-torch CPU") => evaluate_model(mace_torch_cpu, first(structures, 3000), "mace-torch CPU"),
    Symbol("mace-torch GPU") => evaluate_model(mace_torch_cuda, first(structures, 3000), "mace-torch GPU"),
)

# ╔═╡ 33b8e36e-9870-4fc1-a795-b74b042aa8fc
for k in keys(md_benchmark)
    println(k)
    println(md_benchmark[k][1])
end

# ╔═╡ dd6d65ce-1b29-431a-affc-c6c2977b32b0
with_theme() do
    benchmark1_figure = Figure()
    b1_ax = Axis(
        benchmark1_figure[1,1],
        xlabel = "Time per structure / ms",
        ygridvisible = false,
        xgridvisible = false,
        # ylabel = "Model",
        xautolimitmargin = (0.0,0.05),
        yticks = (1:4, [string(k) for k in keys(md_benchmark)]),
        title = "CPU: $(Sys.cpu_info()[1].model)\nGPU: $(CUDA.name(CUDA.device()))"
    )
    barplot!(
        b1_ax,
        1:4,
        [md_benchmark[k][1] / length(md_benchmark[k][2]) for k in keys(md_benchmark)],
        strokewidth = 1.0,
        color = [
            colorant"#1b998b",
            colorant"#700548",
            colorant"#fa9f42",
            colorant"#003459",
            
        ],
        strokecolor = colorant"black",
        bar_labels = :y,
        flip_labels_at = 0.1,
        label_color = colorant"white",
        direction = :x,
    )
    save("benchmark1-md.png", benchmark1_figure)
    benchmark1_figure
end

# ╔═╡ 8c38be3b-91b3-40d4-bbd4-61f2ebba03ec
md"""
# Batching comparison

MACEModels.jl supports batch operations, which should make copying to/from GPUs faster. 
"""

# ╔═╡ d12f000b-3323-489c-89c8-e5934321f495
function evaluate_batch_mace_model(model, st, label = "Model")
    results = [zero(s.positions) for s in st]
    start_time = time_ns()
    p = MACEModels.predict(
        model,
        st[1].atoms,
        [s.positions for s in st], 
        st[1].cell,
    )
    results = MACEModels.get_forces_mean(p)
    end_time = time_ns()
    time_diff = Float64(end_time-start_time) / 1e6 # time in ms
    return time_diff, results
end

# ╔═╡ 4395b1da-120b-43f7-a959-a1fae58caf24
begin
    macemodels_b20_cpu = make_macemodels_model("cpu", 20)
    macemodels_b20_cuda = make_macemodels_model("cuda", 20)
end

# ╔═╡ 3b58a5f1-8060-4ded-96f6-2d5e2448f2ca
cat([rand(10), rand(11)]..., dims=1)

# ╔═╡ f9d6c89e-6d5a-46c6-a3e9-6651cce7ad01
batch_benchmark = Dict(
    Symbol("mace-torch CPU") => md_benchmark[Symbol("mace-torch CPU")],
    Symbol("mace-torch GPU") => md_benchmark[Symbol("mace-torch GPU")],
    Symbol("MACEModels CPU B20") => evaluate_batch_mace_model(macemodels_b20_cpu, first(structures, 3000), "MACEModels CPU B20"),
    Symbol("MACEModels GPU B20") => evaluate_batch_mace_model(macemodels_b20_cuda, first(structures, 3000), "MACEModels GPU B20"),
)

# ╔═╡ e092233b-7708-46d5-bfa8-e6d931c88fa2
bt = batch_benchmark[Symbol("MACEModels CPU B20")]

# ╔═╡ d23f71ba-dab3-4182-b7e5-a5c3ccc9971c
bt[2][1]

# ╔═╡ ba8e5345-de3d-43b4-920b-2e9189ed6b94
batch_predict = MACEModels.predict(
        macemodels_b20_cpu,
        structures[1].atoms,
        [s.positions for s in first(structures, 30)], 
        structures[1].cell,
    )

# ╔═╡ 2b7ac4fc-23d9-4024-895c-0558672f4004


# ╔═╡ 020d6efe-2bd2-4acc-b646-ac05bb97d8c5
batch_predict.energies

# ╔═╡ 3546a254-5815-43e5-a022-fe549f2ea339
with_theme() do
    benchmark2_figure = Figure()
    b2_ax = Axis(
        benchmark2_figure[1,1],
        xlabel = "Time per structure / ms",
        ygridvisible = false,
        xgridvisible = false,
        # ylabel = "Model",
        xautolimitmargin = (0.0,0.05),
        yticks = (1:4, [string(k) for k in keys(batch_benchmark)]),
        title = "CPU: $(Sys.cpu_info()[1].model)\nGPU: $(CUDA.name(CUDA.device()))"
    )
    barplot!(
        b2_ax,
        1:4,
        [batch_benchmark[k][1] / length(batch_benchmark[k][2]) for k in keys(batch_benchmark)],
        strokewidth = 1.0,
        color = [
            colorant"#1b998b",
            colorant"#700548",
            colorant"#fa9f42",
            colorant"#003459",
            
        ],
        strokecolor = colorant"black",
        bar_labels = :y,
        flip_labels_at = 0.1,
        label_color = colorant"white",
        direction = :x,
    )
    save("benchmark2-batch.png", benchmark2_figure)
    benchmark2_figure
end

# ╔═╡ f613b14a-0791-40f1-94d5-471ab23e74fb
md"# Fixing batch dimensionality issues"

# ╔═╡ 99b35f02-e086-42ef-abd1-e1b22fbe3b40
torch_tensor_device(x) = MACEModels.DLPack.share(MACEModels.mtx_to_device(x, MACEModels.CPUDevice()), MACEModels.torch[].from_dlpack)

# ╔═╡ 29b54484-6245-40b3-9d7e-168f9c4290f3
the_batch = Dict{String, Py}()

# ╔═╡ 02cf01b1-de34-4452-91c3-daf5ef4aec36
# Convert to Julia version of the PyTorch dict
batch_parts = [MACEModels.to_julia_dict(
                macemodels_b20_cpu,
                structures[structure_idx].atoms,
                structures[structure_idx].positions,
                structures[structure_idx].cell,
            ) for structure_idx in 1:20]

# ╔═╡ 1b976885-120a-4423-bd9f-ee747d910799
batch_parts[1]

# ╔═╡ ab26fbe4-86f8-49b1-b1ab-41a352804bd2
the_batch["head"] = cat([d["head"] for d in batch_parts]...;dims = 1) |> torch_tensor_device

# ╔═╡ a26a0506-d2b6-4ae3-a7b5-db94b5517ad9
the_batch["cell"] = cat([d["cell"] for d in batch_parts]...;dims = 2) |> torch_tensor_device

# ╔═╡ 4f92ac9d-b576-404f-98b4-7d8ed941b837
begin
	the_batch["energy"] = cat([d["energy"] for d in batch_parts]...;dims = 1) |> torch_tensor_device
    the_batch["forces"] = cat([d["forces"] for d in batch_parts]...;dims = 2) |> torch_tensor_device
    the_batch["node_attrs"] = cat([d["node_attrs"] for d in batch_parts]...;dims = 2) |> torch_tensor_device
    the_batch["positions"] = cat([d["positions"] for d in batch_parts]...;dims = 2) |> torch_tensor_device
    the_batch["shifts"] = cat([d["shifts"] for d in batch_parts]...;dims = 2) |> torch_tensor_device
    the_batch["unit_shifts"] = cat([d["unit_shifts"] for d in batch_parts]...;dims = 2) |> torch_tensor_device
    the_batch["weight"] = cat([d["weight"] for d in batch_parts]...;dims = 1) |> torch_tensor_device
end

# ╔═╡ 89651b98-f451-43c5-b5db-17c48c38fa6d
batch_idx_vector = vcat([repeat([N-1], length(at.masses)) for (N,at) in enumerate([st.atoms for st in structures[1:20]])]...)

# ╔═╡ 73c960df-4b13-4ac6-9891-02a79e4142a0
the_batch["batch"] = batch_idx_vector |> torch_tensor_device # length Natoms * batch size, must be a torch.int type.

# ╔═╡ c6dd8b36-d780-4022-833a-d30be737296f
ptr = vcat(1, [length(structures[st_idx].atoms) for st_idx in 1:20]) |> cumsum

# ╔═╡ c4989dcb-709e-4056-92e0-db30a2ad2a67
the_batch["edge_index"] = cat([batch_parts[d]["edge_index"] .+ (ptr[d] - 1) for d in eachindex(batch_parts)]...;dims = 1) |> torch_tensor_device

# ╔═╡ 490bc8cb-f28e-4205-8de1-8f3afb4d6bb7
 the_batch["ptr"] = ptr .-1 |> torch_tensor_device # length batch size + 1, must be a torch.int type.

# ╔═╡ f0baf97a-9c66-4df7-917f-11aa50145cb8
test_atomicdata = [MACEModels.mace_data[].AtomicData.from_config(
	MACEModels.mace_configuration_from_nqcd_configuration(
		a.atoms,
		a.cell,
		a.positions,
		dtype = Float32,
	),
	mace_torch_cpu.atoms.calc.z_table,
	mace_torch_cpu.atoms.calc.r_max,
	mace_torch_cpu.atoms.calc.available_heads,
) for a in structures[1:20]]

# ╔═╡ 2bf200b5-5640-4331-a914-3b04afbeb73b
pybatch = MACEModels.mace_tools[].torch_geometric.dataloader.DataLoader(
	dataset = test_atomicdata,
	batch_size = 20,
	shuffle = false,
	drop_last = false,
)

# ╔═╡ 283d792f-43cb-4ac6-80dd-ec022117271c
pybatches = [bt for bt in pybatch]

# ╔═╡ 72eb9394-6ec1-4143-87f1-043d484d3a71
sample_pybatch = pybatches[1].to_dict()

# ╔═╡ 01fcd9d9-29c5-4f29-9409-6d2f93ba0e45


# ╔═╡ ea9c0ef8-ff43-4797-959a-8a1dfeb4774f
sample_pybatch["cell"].shape

# ╔═╡ 94cc47b5-ceae-4827-94db-f249e608836f
the_batch["cell"].shape

# ╔═╡ 2254eb58-0edd-405d-8eca-06dcab9afe02
begin
	printstyled("Batch format comparison:\n ", color=:light_yellow, bold=true)
	for k in keys(pyconvert(Dict, sample_pybatch))
		printstyled(k, ":\n ", color = :light_magenta)
		println(
			"Python: ", 
			sample_pybatch[k].shape, 
			" Julia: ", 
			k ∈ keys(pyconvert(Dict, the_batch)) |> collect ? the_batch[k].shape : "not defined",
		)
		if k ∈ keys(pyconvert(Dict, the_batch)) 
			# Check values identical
			python_version = MACEModels.DLPack.from_dlpack(sample_pybatch[k].detach())
			julia_version = MACEModels.DLPack.from_dlpack(the_batch[k].detach())
			println("Data identical to within 1%: ", all(isapprox.(python_version, julia_version, rtol = 0.01)))
		end
	end
end

# ╔═╡ 94e9a26f-e451-497a-a4ae-003ff6f0aa0b
sample_pybatch["ptr"] == the_batch["ptr"]

# ╔═╡ 3ea7942b-d957-4e25-b474-03e2b6a8544a
sample_pybatch["ptr"] == the_batch["ptr"]

# ╔═╡ 0ec91279-52e2-4dc5-88e1-e67745649031
sample_pybatch["edge_index"]

# ╔═╡ 9e0d6c39-5247-4159-bacf-bc8e77feb314


# ╔═╡ 19f6d43b-26c9-42dc-845c-f23d8515c921
the_batch["edge_index"]

# ╔═╡ aa9c039b-0303-43ab-a3f4-5ef3964bcb10
evald_the_batch = macemodels_b20_cpu.models[1].forward(the_batch)

# ╔═╡ 5639654f-e00d-4626-853b-8160007c49b1
evald_sample_pybatch = macemodels_b20_cpu.models[1].forward(sample_pybatch)

# ╔═╡ 8dffc456-9e53-46c0-b9c3-8597afd4562f
evald_the_batch.keys()

# ╔═╡ f7dc29e0-78d4-4eba-9bd6-199f6880d7e6


# ╔═╡ 3c5e99d4-cc26-4974-ad2b-dabd7a0dc8c5
begin
	interesting_outputs = ["energy", "forces", "node_feats", "node_energy", "interaction_energy"]
	printstyled("Output batch format comparison:\n ", color=:light_yellow, bold=true)
	for k in interesting_outputs
		printstyled(k, ":\n ", color = :light_magenta)
		println(
			"Python: ", 
			evald_sample_pybatch[k].shape, 
			" Julia: ", 
			k ∈ interesting_outputs |> collect ? evald_the_batch[k].shape : "not defined",
		)
		if k ∈ interesting_outputs 
			# Check values identical
			python_version = MACEModels.DLPack.from_dlpack(evald_sample_pybatch[k].detach())
			julia_version = MACEModels.DLPack.from_dlpack(evald_the_batch[k].detach())
			println("Data identical to within 1%: ", all(isapprox.(python_version, julia_version, rtol = 0.01)))
		end
	end
end

# ╔═╡ 8699f099-4260-462e-a151-23e02f5e4f34
macemodels_b20_cpu.last_eval_cache

# ╔═╡ 2bd7b6af-2faf-4275-b51a-1f31a779b97a
MACEModels.get_forces_mean(macemodels_b20_cpu.last_eval_cache)

# ╔═╡ a6581a8f-8c85-4e9d-a1d1-9c2021cacd93
mean_forces = macemodels_b20_cpu.last_eval_cache.energies

# ╔═╡ 5482af46-d866-4deb-9991-1c79161ce6bd
macemodels_b20_cpu.last_eval_cache.ptr

# ╔═╡ d1f2ebd1-7f8d-4a39-98d7-5acb485cff33
[mean_forces[:, left:(right-1)] for (left, right) in zip(macemodels_b20_cpu.last_eval_cache.ptr[1:end-1], macemodels_b20_cpu.last_eval_cache.ptr[2:end])]

# ╔═╡ Cell order:
# ╠═74318bde-836a-11f1-884f-2f1ebdb1b6fd
# ╠═3bddf6ec-a18f-4497-b62f-a66c44bac1b9
# ╠═4183f222-0ae5-4d8a-8513-a2364ca9ab72
# ╠═57b17f64-0fd9-4f80-8786-964356b67986
# ╠═a29de054-7fc3-42e7-9838-662bd06347a5
# ╠═d879518d-3eec-41bc-bf2b-2cd7a736fce4
# ╠═9ab36373-e3c3-4e2f-aeba-440e3506d3e7
# ╠═c9fd0e07-0cd2-44ed-bcab-2ae9d4a04df3
# ╠═5a18508e-1763-4b47-9947-bb84e32f393e
# ╠═e9c2865e-f53a-4971-9c01-e39923757ecb
# ╠═77ef40f1-383a-414d-9a92-42bdc2423350
# ╠═45ec9db6-2208-4ca5-95a4-a0b55a6e26f7
# ╠═9f627a5e-a0d8-4063-b0f0-34ae735130e2
# ╠═59d8c1db-cf28-42fa-987f-e02556a9b2d1
# ╠═16a1c475-985b-4ae3-9b6a-ce5d665aeffd
# ╠═762d7a03-cde0-4db8-9100-b5c904ba3e8b
# ╠═ec10dbb6-ac35-4132-9b0b-eb572c0cbca3
# ╠═e125b99b-94c7-4be4-b8ef-e5010d78907e
# ╠═2378c983-075b-418e-aa8c-dbe637057993
# ╠═57832997-bc68-4264-bb6c-811a82aaf14a
# ╠═33b8e36e-9870-4fc1-a795-b74b042aa8fc
# ╠═dd6d65ce-1b29-431a-affc-c6c2977b32b0
# ╠═8c38be3b-91b3-40d4-bbd4-61f2ebba03ec
# ╠═d12f000b-3323-489c-89c8-e5934321f495
# ╠═4395b1da-120b-43f7-a959-a1fae58caf24
# ╠═3b58a5f1-8060-4ded-96f6-2d5e2448f2ca
# ╠═f9d6c89e-6d5a-46c6-a3e9-6651cce7ad01
# ╠═e092233b-7708-46d5-bfa8-e6d931c88fa2
# ╠═d23f71ba-dab3-4182-b7e5-a5c3ccc9971c
# ╠═ba8e5345-de3d-43b4-920b-2e9189ed6b94
# ╠═2b7ac4fc-23d9-4024-895c-0558672f4004
# ╠═020d6efe-2bd2-4acc-b646-ac05bb97d8c5
# ╠═3546a254-5815-43e5-a022-fe549f2ea339
# ╠═5dd10225-6b2a-4365-9f6e-fb1075b9d5bb
# ╠═ca53c48d-23f6-41be-863b-e51d4032c27d
# ╠═f613b14a-0791-40f1-94d5-471ab23e74fb
# ╠═99b35f02-e086-42ef-abd1-e1b22fbe3b40
# ╠═29b54484-6245-40b3-9d7e-168f9c4290f3
# ╠═02cf01b1-de34-4452-91c3-daf5ef4aec36
# ╠═1b976885-120a-4423-bd9f-ee747d910799
# ╠═ab26fbe4-86f8-49b1-b1ab-41a352804bd2
# ╠═a26a0506-d2b6-4ae3-a7b5-db94b5517ad9
# ╠═c4989dcb-709e-4056-92e0-db30a2ad2a67
# ╠═4f92ac9d-b576-404f-98b4-7d8ed941b837
# ╠═89651b98-f451-43c5-b5db-17c48c38fa6d
# ╠═73c960df-4b13-4ac6-9891-02a79e4142a0
# ╠═c6dd8b36-d780-4022-833a-d30be737296f
# ╠═490bc8cb-f28e-4205-8de1-8f3afb4d6bb7
# ╠═f0baf97a-9c66-4df7-917f-11aa50145cb8
# ╠═2bf200b5-5640-4331-a914-3b04afbeb73b
# ╠═283d792f-43cb-4ac6-80dd-ec022117271c
# ╠═72eb9394-6ec1-4143-87f1-043d484d3a71
# ╠═01fcd9d9-29c5-4f29-9409-6d2f93ba0e45
# ╠═ea9c0ef8-ff43-4797-959a-8a1dfeb4774f
# ╠═94cc47b5-ceae-4827-94db-f249e608836f
# ╠═2254eb58-0edd-405d-8eca-06dcab9afe02
# ╠═94e9a26f-e451-497a-a4ae-003ff6f0aa0b
# ╠═3ea7942b-d957-4e25-b474-03e2b6a8544a
# ╠═0ec91279-52e2-4dc5-88e1-e67745649031
# ╠═9e0d6c39-5247-4159-bacf-bc8e77feb314
# ╠═19f6d43b-26c9-42dc-845c-f23d8515c921
# ╠═aa9c039b-0303-43ab-a3f4-5ef3964bcb10
# ╠═5639654f-e00d-4626-853b-8160007c49b1
# ╠═8dffc456-9e53-46c0-b9c3-8597afd4562f
# ╠═f7dc29e0-78d4-4eba-9bd6-199f6880d7e6
# ╠═3c5e99d4-cc26-4974-ad2b-dabd7a0dc8c5
# ╠═8699f099-4260-462e-a151-23e02f5e4f34
# ╠═2bd7b6af-2faf-4275-b51a-1f31a779b97a
# ╠═03c97cd3-8299-4a52-8718-c29df30a2418
# ╠═a6581a8f-8c85-4e9d-a1d1-9c2021cacd93
# ╠═5482af46-d866-4deb-9991-1c79161ce6bd
# ╠═d1f2ebd1-7f8d-4a39-98d7-5acb485cff33
