### A Pluto.jl notebook ###
# v1.0.3

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
        structures[1].atoms,
        structures[1].cell,
        ["../test/test_model/MACE_model_swa.model"],
        device = device,
        default_dtype=Float32,
        batch_size = batch_size,
    )
    return mace_model
end

# ╔═╡ 45ec9db6-2208-4ca5-95a4-a0b55a6e26f7
begin
    macemodels_b1_cpu = make_macemodels_model("cpu", 1)
    macemodels_b1_cuda = make_macemodels_model("cuda", 1)
end

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
    Symbol("mace-torch CPU") => evaluate_model(mace_torch_cpu, first(structures, 3000), "mace-torch CPU"),
    Symbol("mace-torch GPU") => evaluate_model(mace_torch_cuda, first(structures, 3000), "mace-torch GPU"),
    Symbol("MACEModels CPU") => evaluate_model(macemodels_b1_cpu, first(structures, 3000), "MACEModels CPU"),
    Symbol("MACEModels GPU") => evaluate_model(macemodels_b1_cuda, first(structures, 3000), "MACEModels GPU"),
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

# ╔═╡ f9d6c89e-6d5a-46c6-a3e9-6651cce7ad01
batch_benchmark = Dict(
    Symbol("mace-torch CPU") => md_benchmark[Symbol("mace-torch CPU")],
    Symbol("mace-torch GPU") => md_benchmark[Symbol("mace-torch GPU")],
    Symbol("MACEModels CPU B20") => evaluate_batch_mace_model(macemodels_b20_cpu, first(structures, 3000), "MACEModels CPU B20"),
    Symbol("MACEModels GPU B20") => evaluate_batch_mace_model(macemodels_b20_cuda, first(structures, 3000), "MACEModels GPU B20"),
)

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
# ╠═f9d6c89e-6d5a-46c6-a3e9-6651cce7ad01
# ╠═3546a254-5815-43e5-a022-fe549f2ea339
