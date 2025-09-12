using MACEModels
using PythonCall
using ProgressMeter
using NQCDynamics, NQCModels, NQCBase, NQCDInterfASE
using Test

@info "Loading python modules - if things fail here, check CondaPkg is working, and whether MACE has had any changes."
ase_io = pyimport("ase.io")
mc = pyimport("mace.calculators")
model_path = "$(@__DIR__)/test_model/MACE_model_swa.model"
mace_calc_small = mc.MACECalculator(
	model_paths=[model_path], 
	device="cpu", 
	default_dtype="float32"
)
ase_structure = ase_io.read("$(@__DIR__)/test_model/h2cu_diffusion_desorption_validation.xyz", index=0)
ase_structure.calc = mace_calc_small
mace_model_ase = ClassicalASEModel(ase_structure)

@info "Checking PyTorch backends."
torch = pyimport("torch")

cuda_avail = pyconvert(Bool, torch.backends.cuda.is_built())
#=
if cuda_avail
	using CUDA
end
=#
mps_avail = pyconvert(Bool, torch.backends.mps.is_built())

structures = NQCBase.read_extxyz("$(@__DIR__)/test_model/h2cu_diffusion_desorption_validation.xyz")
structures_to_test = first(structures[2], 3000)

@info "Available PyTorch backends:" CUDA=cuda_avail MPS=mps_avail

@info "Now entering package tests."
@testset "Model loading" begin
    # Write your tests here.
	for backend in ["cpu"]
		MACEModel(
			structures[1],
			structures[3],
			[model_path];
			default_dtype = Float32,
			device = backend
		)
	end
end

@testset "Model inference" begin
	model = MACEModel(
		structures[1],
		structures[3],
		[model_path];
		default_dtype = Float32,
		device = "cpu",
		batch_size = 10, # This ensures batching and un-batching operations work as well. 
	)
	@info "Evaluating structures using MACEModels.predict!()"
	mace_prediction = MACEModels.predict(
		model, 
		structures[1],
		structures_to_test,
		structures[3]
	)
	energies_macemodels = MACEModels.get_energy_mean(mace_prediction)
	forces_macemodels = MACEModels.get_forces_mean(mace_prediction)
	@info "Evaluating ase calculator using NQCModels - Check NQCDInterfASE.jl if something breaks here. "
	energies_mace = @showprogress [NQCModels.potential(mace_model_ase, pos) for pos in structures_to_test] # energies in a.u. from ASE calculator. 
	forces_mace = @showprogress [-NQCModels.derivative(mace_model_ase, pos) for pos in structures_to_test] # forces in a.u. from ASE calculator. 
	@info "Checking equality to within 1e-5"
	compare_energies = isapprox.(energies_mace, energies_macemodels; atol = 1e-5)
	for energy in compare_energies
		@test energy
	end
	for forces in zip(forces_mace, forces_macemodels)
		compare = isapprox.(forces...; atol=1e-5)
		for i in eachindex(compare)
			@test compare[i]
		end
	end
end

#= Remove temporarily for now
if cuda_avail
	mace_calc_small = mc.MACECalculator(
		model_paths=["$(@__DIR__)/test_model/MACE_model_swa.model"], 
		device="cuda", 
		default_dtype="float32"
	)
	ase_structure = ase_io.read("$(@__DIR__)/test_model/h2cu_diffusion_desorption_validation.xyz", index=0)
	ase_structure.calc = mace_calc_small
	mace_model_ase = ClassicalASEModel(ase_structure)
	model = MACEModel(
		structures[1],
		structures[3],
		[model_path];
		default_dtype = Float32,
		device = "cuda",
		batch_size = 25, # This ensures batching and un-batching operations work as well. 
	)
	@info "Evaluating structures using MACEModels.predict!()"
	mace_prediction = MACEModels.predict(
		model, 
		structures[1],
		structures_to_test,
		structures[3]
	)
	energies_macemodels = MACEModels.get_energy_mean(mace_prediction)
	forces_macemodels = MACEModels.get_forces_mean(mace_prediction)
	@info "Evaluating ase calculator using NQCModels - Check NQCDInterfASE.jl if something breaks here. "
	energies_mace = @showprogress [NQCModels.potential(mace_model_ase, pos) for pos in structures_to_test] # energies in a.u. from ASE calculator. 
	forces_mace = @showprogress [NQCModels.derivative(mace_model_ase, pos) for pos in structures_to_test] # forces in a.u. from ASE calculator. 
	@info "Checking equality to within 1e-5"
	@. @test "Energies" isapprox(energies_mace, energies_macemodels; atol = 1e-5)
	for forces in zip(forces_mace, forces_macemodels)
		@. @test "Forces" isapprox(forces...; atol=1e-5)
	end
end
=#
