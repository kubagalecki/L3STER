# Examples

This directory houses self-contained examples of L3STER usage.
You may build and run them directly from the directories in which they reside.

The examples consume L3STER as a CMake project directory.
If you copy a directory containing an example to a different location, be sure to update the path referenced by `add_subdirectory` in the example's `CMakeLists.txt`

### Usage
Assuming all L3STER dependencies are visible to CMake (e.g. when a spack environment containing them is active), you can build and run the examples as follows:

```bash
# Starting in example directory
pwd # [...]/L3STER/examples/example-name

# Subdirectory for the build
mkdir build
cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpic++ ..

# Build
cmake --build .

# Set OpenMP thread-to-core binding via environment variables
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Run as MPI executable, use srun if running on a cluster managed by slurm
mpirun -n X example-name
```