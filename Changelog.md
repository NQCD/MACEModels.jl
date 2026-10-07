# Changelog 

## v1.6.0
This update makes a few breaking changes across the board. Make sure to read through the rest of this changelog and check the updated API docs [here](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCModels/mace/)

### Breaking changes
- `MACEModel` now requires a `NQCBase.Structure` for construction instead of just atoms and cell. This was done since warming up the compiled versions of the model requires running all the relevant code at least once. 

### Efficiency improvements
#### Model inference speed-up
After investigating how MACE inference is sped up in LAMMPS, I've started trying to recreate something similar. LAMMPS makes use of the Torch C++ libraries to perform inference on structures in a particular dictionary format. 
This dictionary format is similar to how `mace-torch` inference works when using e.g. the ase calculator or `mace_eval_configs`. 
With this update, I've recreated the machinery needed to build this dictionary structure in Julia and translate the data across to Python. 
As a result, this bypasses all of the `torch.geometric.DataLoader` construction. 

While this means I'm still hooking into `mace-torch` with `PythonCall` now, I could move to directly calling the C++ library for Torch in future to eliminate the need for Python dependencies entirely. 
As long as Python is here, I can more easily find common model loading errors and warn about them, which would be slightly more annoying to do with C++. 
Also, I don't believe anyone has built a BinaryBuilder package to distribute libtorch yet, which I also don't want to get into at the moment. 

#### Support for JIT-compiled models
`mace-torch` supports e3nn's compile mode which is the precursor to running model inference with mostly C code. 
When creating a `MACEModel`, set `compile=true` to use this feature, which should give a slight efficiency boost even though the model is still being called through a Julia --> Python --> PyTorch chain. 

#### Device-specific neighbour lists
`NeighbourLists.jl` v0.6 has added a device-agnostic neighbour list implementation which yields the same results as the matscipy neighbour list used in `mace-torch`. 
Benchmarks on my laptop indicate that structures around 2000 atoms begin to have a speed advantage for GPU-based neighbour list construction, and CPU-based neighbour lists are at least as fast as those in `mace-torch`. 

### Further changes
- Added `MACEModels.predict!` and variants using `NQCBase.Structure` inputs. This allows for batch evaluating structures similar to `mace_eval_configs`.
- Added support for converting model inputs and outputs between different device backends. This could be useful e.g. for GPU-based dynamics + GPU-based inference. Output functions for useful quantities should hopefully be device-agnostic too. 
- Added benchmarking / testing notebooks to a subfolder. These can be used to showcase different aspects of the package.
