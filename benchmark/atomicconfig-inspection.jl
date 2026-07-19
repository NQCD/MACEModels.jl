### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 583d44ce-828c-11f1-8cd6-0d8d5c8c0dbe
import Pkg

# ╔═╡ 94655f2b-bfce-4d6e-b9ca-c7edc9f1e83b
Pkg.activate(".")

# ╔═╡ 10a8c530-e531-43b3-af09-e9bfcad41c61
# ╠═╡ disabled = true
#=╠═╡
Pkg.add("DLPack")
  ╠═╡ =#

# ╔═╡ 529161ff-6094-45ae-a918-f875c455353b
using PythonCall

# ╔═╡ acdea4c1-e492-48c0-ba61-507c3ab2410d
using DLPack

# ╔═╡ 2f93f960-1d5d-4747-a7e0-0a2c15ffffe9
using NeighbourLists

# ╔═╡ 3e7fbf33-49b9-400d-b0d5-c008803efcf7
using NQCBase, CUDA

# ╔═╡ 5d145076-10ba-4cbf-b00d-e48049756fe0
using Unitful, UnitfulAtomic

# ╔═╡ 6372a38a-a078-4d59-9b74-57d9586772c9
using LinearAlgebra

# ╔═╡ 2605e644-9e81-4de0-a357-e90419fa0340
using KernelAbstractions

# ╔═╡ 27f13aa4-b824-40b9-9038-6133fae12df9
begin
    torch = pyimport("torch")
    torch.set_default_dtype(torch.float32)
    mace_data = pyimport("mace.data")
    ase_io = pyimport("ase.io")
    mace_utils = pyimport("mace.tools.utils")
end

# ╔═╡ 9918c7a0-971a-4261-a518-6ed4616a8d6e
md"""
# `mace-torch` AtomicData reference
"""

# ╔═╡ 74fdcf52-a5b4-444e-a530-8b761ca4d8d9
mace_atomicconfig = mace_data.AtomicData.from_config(
    mace_data.config_from_atoms(
        ase_io.read("../test/test_model/h2cu_diffusion_desorption_validation.xyz", index="0"),
        head_name = "Default",
    ), # MACE config
    z_table = mace_utils.AtomicNumberTable(zs = PyList([1,29])), #
    cutoff = 5.0, # Cutoff in Angstrom
)

# ╔═╡ 067f6243-24a3-44c7-aa21-ddbd1f86e2a1
from_dlpack(mace_atomicconfig.node_attrs)

# ╔═╡ 52522e64-f3cb-4bf4-b67c-0de4d34c25d1
mace_atomicconfig.edge_index |> from_dlpack

# ╔═╡ 16ef6723-67dc-4583-9af7-f4f56fe6209e
mace_atomicconfig.shifts |> from_dlpack

# ╔═╡ c2d51bd6-0acb-463e-8824-20d8dcff41a6
mace_atomicconfig.unit_shifts |> from_dlpack

# ╔═╡ 76c14080-2dc7-431a-b088-6a3f663bac64
md"""
# Julia recreation
"""

# ╔═╡ bf10e2d4-2545-4839-829c-3c2138adbe1f
# ╠═╡ disabled = true
#=╠═╡
Pkg.add("NeighbourLists")
  ╠═╡ =#

# ╔═╡ 8be70a9d-390f-4309-bdb9-fdb309d33f56
# ╠═╡ disabled = true
#=╠═╡
Pkg.add("NQCBase")
  ╠═╡ =#

# ╔═╡ e21a7e34-107f-4450-b3dc-6785c1978e95
# ╠═╡ disabled = true
#=╠═╡
Pkg.add("Unitful")
  ╠═╡ =#

# ╔═╡ 94ae92b8-2544-41cc-b08f-f695c2473306
# ╠═╡ disabled = true
#=╠═╡
Pkg.add("UnitfulAtomic")
  ╠═╡ =#

# ╔═╡ b2267c48-e6c8-427f-aa4c-70c094a50804
ab_structure = System(first(NQCBase.read_extxyz("../test/test_model/h2cu_diffusion_desorption_validation.xyz", 1)))

# ╔═╡ fe66bc47-0ee3-4880-ba89-ad1aa591d0e2
pos_cu = position(ab_structure, :) .|> ustrip |> cu

# ╔═╡ 02bca0bd-d37e-41e5-8b06-6a654961dc37
ta = CUDA.zeros(Float32, 3,56)

# ╔═╡ 6e63bfaa-e12c-4308-ba36-9690cca1354a
ta

# ╔═╡ 6e7f14ce-08c6-4ee1-b8d1-e8096fc28172
# ab_nl = neighbour_list(ab_structure, 5.0u"Å")

# ╔═╡ 5f1af5ac-45a4-48ee-861c-577b4487c151
ab_structure.cell.periodicity

# ╔═╡ 7f6e31fe-a24d-4e78-96dc-0b2c8aecb4b4
cell_cu = reduce(hcat, ab_structure.cell.cell_vectors.|> ustrip) |> CuArray

# ╔═╡ 61c839df-195f-4412-b438-5f476ffb2ba9
ab_nl = neighbour_list(pos_cu, 5.0, hcat(ab_structure.cell.cell_vectors...) |> Matrix .|> ustrip, ab_structure.cell.periodicity; backend = CUDABackend())

# ╔═╡ 9f5460ac-8a23-4711-ad98-8a2922194707
sr = vcat((ab_nl.i .-1)', (ab_nl.j .-1)')

# ╔═╡ f0065583-9cd5-4d41-a20c-8de0f355dff9
# Need to remove self-interactions within the original unit cell from the graph. So mask with symdiff(eachindex(i), i.==j .& sum.(S) .== 0)

# ╔═╡ 742089b7-07b6-4904-88e9-89bd1b1af0d6
mask = .!(ab_nl.i .== ab_nl.j .& sum.(ab_nl.S) .== 0)

# ╔═╡ 95fb4cf4-54fa-493a-b12a-67c9d8a0c486
vcat((ab_nl.i .-1)', (ab_nl.j .-1)') |> permutedims == mace_atomicconfig.edge_index |> from_dlpack

# ╔═╡ 0c5393f5-5799-4232-bf59-bfdafdb7bf18
findall(ab_nl.i .== ab_nl.j)

# ╔═╡ d8f9ed81-dd7f-468a-88bb-2368ea1af615
# ╠═╡ disabled = true
#=╠═╡
Pkg.add("KernelAbstractions")
  ╠═╡ =#

# ╔═╡ e7c0df73-dc5d-4b1b-a8ea-91be91d1a572
@kernel function shift_dot!(distance_shifts, unit_shifts, cell)
    i,j = @index(Global, NTuple)
    @inbounds distance_shifts[i,j] += cell[i,k] * unit_shifts[k,j]
end

# ╔═╡ 57fb93b5-b454-478c-8cb8-87c437fadcf6
Smat = reinterpret(reshape, Int32, ab_nl.S)

# ╔═╡ 93135cde-c8f5-45f1-b06b-1a56d020b05a
shift_dot!(zero(Smat), Smat, cell_cu)

# ╔═╡ 775f9a74-c64b-4ecf-8e2c-05d99ed5fc39
begin
    shifts2 = reduce(hcat, Array(ab_nl.S)) .|> Float32
    shifts3 = deepcopy(shifts2)
    for idx in axes(shifts3, 2)
        shifts3[:,idx] = ab_nl.C * shifts2[:, idx]
    end
    shifts3 = cu(shifts3)
end

# ╔═╡ 573f6781-5002-4c77-96c4-e9a6d7bff6e8
begin
    shifts = reinterpret(reshape, Int32, ab_nl.S)
    # for idx in eachindex(ab_nl.S)
    #     shifts[:,idx] .= ab_nl.C * ab_nl.S[idx]
    # end
    
end

# ╔═╡ 526969ee-e5fb-41bb-9d92-c8746e5d8beb
@code_lowered stack(x -> ab_nl.C * x, ab_nl.S)

# ╔═╡ a3cbad78-7464-4445-b02d-eaf5647c34cb
isapprox.(eachcol(shifts3), eachcol(mace_atomicconfig.shifts |> from_dlpack), rtol = 1e-6) |> all # Deviates due to Float32 vs. Float64, but not significantly. 

# ╔═╡ fd86fcc3-842e-413a-aae7-4179bcdc0b8b
nqcd_structure = first(NQCBase.read_extxyz("../test/test_model/h2cu_diffusion_desorption_validation.xyz", 1))

# ╔═╡ a7fcd2e3-445b-4ef9-9cfc-37dfb6305c24
nqcd_structure.cell.vectors |> CuArray |> PeriodicCell

# ╔═╡ 33d71769-e137-4c1a-8ec7-5651656362c0
nqcd_structure.positions |> CuArray

# ╔═╡ c54292bf-cb54-4cbe-972f-cebc1822ff67
ta .= nqcd_structure.positions |> CuArray .|> Float32

# ╔═╡ 3b35fe3c-17af-45e3-b262-adc291352543
onehot = vcat(
    [permutedims(nqcd_structure.atoms.numbers .== t) for t in unique(nqcd_structure.atoms.numbers) |> sort]...
)

# ╔═╡ 358e737d-0a1d-4b75-b796-e40e356955d2
ab_nl.i .* .!mask

# ╔═╡ e293a428-e577-4048-98a8-8809dd439051
onehot == from_dlpack(mace_atomicconfig.node_attrs)

# ╔═╡ 0d276d54-6470-453d-ada6-cd5ec0a8ed5e
nqcd_structure.atoms.numbers

# ╔═╡ 6cd29d06-52ab-4ea6-a918-e6de1331ef86
DLPack.share(
            vcat((ab_nl.i .-1)', (ab_nl.j .-1)') |> permutedims,
            torch.from_dlpack,
        )

# ╔═╡ ea7db04c-1430-4531-b4d5-b6a479c4cadf
DLPack.share(
            Float64.(onehot),
            torch.from_dlpack,
        )

# ╔═╡ 17d4c100-f314-410f-acc9-4f9e80ee80e1
DLPack.share(Float64[], torch.from_dlpack).shape |> length

# ╔═╡ c6ffaebe-45c7-4ac7-9539-db02c206f0b5
DLPack.share(zero(nqcd_structure.positions), torch.from_dlpack).shape

# ╔═╡ 6f837599-7590-4520-ae28-0f42114b3c0a
DLPack.share(nqcd_structure.cell.periodicity, torch.from_dlpack).dtype

# ╔═╡ fa775138-068c-4b87-968d-7209e1bf0889
@pyeval (x=nothing) => `x is None` => Bool

# ╔═╡ e788063a-53cb-4bcb-861b-670f3fabae5d
Py(nothing)

# ╔═╡ 11486b89-b7af-4ee1-8d81-7aae8534b202
edge_index=DLPack.share(
            hcat(
                (ab_nl.i .-1) .* mask, 
                (ab_nl.j .-1) .* mask,
            ),
            torch.from_dlpack,
        )

# ╔═╡ 66f746c9-664e-4d68-bfdd-c22aafaeb840
DLPack.share(
            pos_cu,
            torch.from_dlpack,
        )

# ╔═╡ 2b51ec5c-9028-4967-91ef-1d0ec7d5b5c6
begin # Trying to reassemble an AtomicData class
    atd_recreate = mace_data.AtomicData(
        edge_index,
        positions=DLPack.share(
            auconvert.(u"Å", nqcd_structure.positions) .|> ustrip |> cu,
            torch.from_dlpack,
        ),
        shifts=DLPack.share(
            shifts3,
            torch.from_dlpack,
        ),
        unit_shifts=DLPack.share(
            cu(shifts2),
            torch.from_dlpack,
        ),
        cell=DLPack.share(
            cell_cu,
            torch.from_dlpack,
        ),
        node_attrs=DLPack.share(
            Float32.(onehot) |> cu,
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
        forces=DLPack.share(zero(nqcd_structure.positions), torch.from_dlpack),
        energy=nothing |> Py,
        stress=nothing |> Py,
        virials=nothing |> Py,
        dipole=nothing |> Py,
        charges=nothing |> Py,
        elec_temp=nothing |> Py,
        total_charge=nothing |> Py,
        polarizability=nothing |> Py,
        # total_spin=nothing,
        pbc=DLPack.share(nqcd_structure.cell.periodicity, torch.from_dlpack),
        # # density_coefficients=density_coefficients,
        # rcell=nothing,
        # volume=nothing,
        # fermi_level=nothing,
        # external_field=nothing,
    )
end

# ╔═╡ 800a4c3f-d4b0-431a-b098-1713f624c606
atd_recreate == mace_atomicconfig

# ╔═╡ 2ee0201c-9824-4139-9681-031f368cf19f
md"# Model evaluation test"

# ╔═╡ c7eadd90-19ad-41b5-81ad-2f0646c92724
mace_torch_geometric = pyimport("mace.tools.torch_geometric")

# ╔═╡ bc88f1c7-dd39-4f09-83ec-6bd8151c3699
dataloader = mace_torch_geometric.DataLoader(
    PyList(repeat([atd_recreate ], 20)),
    batch_size = 2,
    shuffle = false,
    drop_last = false,
)

# ╔═╡ 3d852595-33af-42af-ac5b-abc6b66d0a1d
model = torch.load("../test/test_model/MACE_model_swa.model").to("cuda")

# ╔═╡ 53428dee-8cb3-43ae-9124-a2645a5ea5cd
for p in model.parameters()
    
end

# ╔═╡ be1aebc1-fa59-45a4-a971-4d5e64ab4dfc
out = Py[]

# ╔═╡ d9152912-7e1b-4d25-b67e-46d7155a9539
for (i,batch) in enumerate(dataloader)
    println(i)
    push!(out, model(batch.to("cuda")))
    println(out[end]["forces"] |> from_dlpack)
end

# ╔═╡ 01ed7e3d-3a1b-4545-a4f0-cfec0459ed69
out_cuda = out[1]["forces"] |> from_dlpack

# ╔═╡ 8c204976-fa56-4605-af30-68f7d21bba2d
@views st1 = out_cuda[:,1:56]

# ╔═╡ 7f6f65c2-9e75-4faf-b5c1-c40dbfb55a7c
st1_cpu = zeros(3,56)

# ╔═╡ 3192dd10-65bf-41e4-b32d-cc9bb50bcc76
st1_cpu .= st1

# ╔═╡ Cell order:
# ╠═583d44ce-828c-11f1-8cd6-0d8d5c8c0dbe
# ╠═94655f2b-bfce-4d6e-b9ca-c7edc9f1e83b
# ╠═529161ff-6094-45ae-a918-f875c455353b
# ╠═10a8c530-e531-43b3-af09-e9bfcad41c61
# ╠═acdea4c1-e492-48c0-ba61-507c3ab2410d
# ╠═27f13aa4-b824-40b9-9038-6133fae12df9
# ╟─9918c7a0-971a-4261-a518-6ed4616a8d6e
# ╠═74fdcf52-a5b4-444e-a530-8b761ca4d8d9
# ╠═067f6243-24a3-44c7-aa21-ddbd1f86e2a1
# ╠═52522e64-f3cb-4bf4-b67c-0de4d34c25d1
# ╠═16ef6723-67dc-4583-9af7-f4f56fe6209e
# ╠═c2d51bd6-0acb-463e-8824-20d8dcff41a6
# ╟─76c14080-2dc7-431a-b088-6a3f663bac64
# ╠═bf10e2d4-2545-4839-829c-3c2138adbe1f
# ╠═2f93f960-1d5d-4747-a7e0-0a2c15ffffe9
# ╠═8be70a9d-390f-4309-bdb9-fdb309d33f56
# ╠═3e7fbf33-49b9-400d-b0d5-c008803efcf7
# ╠═e21a7e34-107f-4450-b3dc-6785c1978e95
# ╠═94ae92b8-2544-41cc-b08f-f695c2473306
# ╠═5d145076-10ba-4cbf-b00d-e48049756fe0
# ╠═b2267c48-e6c8-427f-aa4c-70c094a50804
# ╠═a7fcd2e3-445b-4ef9-9cfc-37dfb6305c24
# ╠═fe66bc47-0ee3-4880-ba89-ad1aa591d0e2
# ╠═33d71769-e137-4c1a-8ec7-5651656362c0
# ╠═02bca0bd-d37e-41e5-8b06-6a654961dc37
# ╠═c54292bf-cb54-4cbe-972f-cebc1822ff67
# ╠═6e63bfaa-e12c-4308-ba36-9690cca1354a
# ╠═6e7f14ce-08c6-4ee1-b8d1-e8096fc28172
# ╠═5f1af5ac-45a4-48ee-861c-577b4487c151
# ╠═7f6e31fe-a24d-4e78-96dc-0b2c8aecb4b4
# ╠═61c839df-195f-4412-b438-5f476ffb2ba9
# ╠═9f5460ac-8a23-4711-ad98-8a2922194707
# ╠═f0065583-9cd5-4d41-a20c-8de0f355dff9
# ╠═742089b7-07b6-4904-88e9-89bd1b1af0d6
# ╠═95fb4cf4-54fa-493a-b12a-67c9d8a0c486
# ╠═0c5393f5-5799-4232-bf59-bfdafdb7bf18
# ╠═6372a38a-a078-4d59-9b74-57d9586772c9
# ╠═d8f9ed81-dd7f-468a-88bb-2368ea1af615
# ╠═2605e644-9e81-4de0-a357-e90419fa0340
# ╠═e7c0df73-dc5d-4b1b-a8ea-91be91d1a572
# ╠═57fb93b5-b454-478c-8cb8-87c437fadcf6
# ╠═93135cde-c8f5-45f1-b06b-1a56d020b05a
# ╠═775f9a74-c64b-4ecf-8e2c-05d99ed5fc39
# ╠═573f6781-5002-4c77-96c4-e9a6d7bff6e8
# ╠═526969ee-e5fb-41bb-9d92-c8746e5d8beb
# ╠═a3cbad78-7464-4445-b02d-eaf5647c34cb
# ╠═fd86fcc3-842e-413a-aae7-4179bcdc0b8b
# ╠═3b35fe3c-17af-45e3-b262-adc291352543
# ╠═358e737d-0a1d-4b75-b796-e40e356955d2
# ╠═e293a428-e577-4048-98a8-8809dd439051
# ╠═0d276d54-6470-453d-ada6-cd5ec0a8ed5e
# ╠═6cd29d06-52ab-4ea6-a918-e6de1331ef86
# ╠═ea7db04c-1430-4531-b4d5-b6a479c4cadf
# ╠═17d4c100-f314-410f-acc9-4f9e80ee80e1
# ╠═c6ffaebe-45c7-4ac7-9539-db02c206f0b5
# ╠═6f837599-7590-4520-ae28-0f42114b3c0a
# ╠═fa775138-068c-4b87-968d-7209e1bf0889
# ╠═e788063a-53cb-4bcb-861b-670f3fabae5d
# ╠═11486b89-b7af-4ee1-8d81-7aae8534b202
# ╠═66f746c9-664e-4d68-bfdd-c22aafaeb840
# ╠═2b51ec5c-9028-4967-91ef-1d0ec7d5b5c6
# ╠═800a4c3f-d4b0-431a-b098-1713f624c606
# ╟─2ee0201c-9824-4139-9681-031f368cf19f
# ╠═c7eadd90-19ad-41b5-81ad-2f0646c92724
# ╠═bc88f1c7-dd39-4f09-83ec-6bd8151c3699
# ╠═3d852595-33af-42af-ac5b-abc6b66d0a1d
# ╠═53428dee-8cb3-43ae-9124-a2645a5ea5cd
# ╠═be1aebc1-fa59-45a4-a971-4d5e64ab4dfc
# ╠═d9152912-7e1b-4d25-b67e-46d7155a9539
# ╠═01ed7e3d-3a1b-4545-a4f0-cfec0459ed69
# ╠═8c204976-fa56-4605-af30-68f7d21bba2d
# ╠═7f6f65c2-9e75-4faf-b5c1-c40dbfb55a7c
# ╠═3192dd10-65bf-41e4-b32d-cc9bb50bcc76
