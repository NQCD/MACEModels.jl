using MACEModels
using PythonCall
using ProgressMeter
using NQCDynamics, NQCModels, NQCBase, NQCDInterfASE
using Test

@info "Loading python modules - if things fail here, check CondaPkg is working, and whether MACE has had any changes."
ase_io = pyimport("ase.io")
mc = pyimport("mace.calculators")
model_path = "$(@__DIR__)/test_model/MACE_model_swa.model"

@info "Checking PyTorch backends."
torch = pyimport("torch")
cuda_avail = get(ENV, "JULIA_MACEMODELS_TEST_CUDA", pyconvert(Bool, torch.backends.cuda.is_built()))
#=
if cuda_avail
	using CUDA
end
=#
mps_avail = get(ENV, "JULIA_MACEMODELS_TEST_MPS", pyconvert(Bool, torch.backends.mps.is_built()))

backends = ["cpu", "cuda", "mps"] # ["cpu", "cuda", "mps"]
backends_avail = [cuda_avail, mps_avail] # [true, cuda_avail, mps_avail]
if cuda_avail
    using CUDA
end
if mps_avail
    using Metal
end

@info "Available PyTorch backends:" CUDA = cuda_avail MPS = mps_avail

mace_calc_small = mc.MACECalculator(
    model_paths=[model_path],
    device=backends[findlast(backends_avail)],
    default_dtype="float32" # Hardcoded for this particular model.
)
ase_structure = ase_io.read("$(@__DIR__)/test_model/h2cu_diffusion_desorption_validation.xyz", index=0)
ase_structure.calc = mace_calc_small
mace_model_ase = ClassicalASEModel(ase_structure)





structures = read_extxyz("$(@__DIR__)/test_model/h2cu_diffusion_desorption_validation.xyz", :)
structures_to_test = first(structures, 3000)


@info "Now entering package tests."
for device_string in backends[backends_avail] # Select backends to test based on what the machine we're testing on can do.
    @testset "Model loading ($(device_string))" begin
        # Write your tests here.
        model = MACEModel(
                structures[1].atoms,
                structures[1].cell,
                [model_path];
                default_dtype=Float32,
                device=device_string
            )
        # Check model has a device field corresponding to where it should be loaded.
        @test model.device == [device_string]
    end

    @testset "Model inference ($(device_string))" begin
        model = MACEModel(
            structures[1].atoms,
            structures[1].cell,
            [model_path];
            default_dtype=Float32,
            device=device_string,
            batch_size=10, # This ensures batching and un-batching operations work as well.
        )
        @info "Evaluating structures using MACEModels.predict!()"
        mace_prediction = @time MACEModels.predict(
            model,
            [s.atoms for s in structures_to_test],
            [s.positions for s in structures_to_test],
            [s.cell for s in structures_to_test],
        )
        energies_macemodels = MACEModels.get_energy_mean(mace_prediction)
        forces_macemodels = MACEModels.get_forces_mean(mace_prediction)
        @info "Evaluating ase calculator using NQCModels - Check NQCDInterfASE.jl if something breaks here. "
        energies_mace = @showprogress [NQCModels.potential(mace_model_ase, st.positions) for st in structures_to_test] # energies in a.u. from ASE calculator.
        forces_mace = @showprogress [-NQCModels.derivative(mace_model_ase, st.positions) for st in structures_to_test] # forces in a.u. from ASE calculator.
        @info "Checking equality to within 1e-5 Hartree / ~25 meV"
        compare_energies = isapprox.(energies_mace, energies_macemodels; atol=1e-5)
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
end
