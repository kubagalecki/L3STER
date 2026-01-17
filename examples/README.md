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
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_CXX_FLAGS="-march=native -mtune=native" ..

# Build
cmake --build .

# Due to mixing 2 different parallel runtimes (OpenMP & TBB), setting these environment variables is crucial for good performance
export OMP_PROC_BIND=false     # Disable binding - let the system scheduler take care of the threads (also silences Kokkos warning)
export OMP_WAIT_POLICY=PASSIVE # OpenMP threads should go to sleep immediately after parallel work completes. Important!
export OMP_NUM_THREADS=[NHWT]  # NHWT=number of hardware threads (this is likely the default behavior, but it's best to make sure)

# If you're just running the example on your local machine (which is entirely sufficient), simply run the produced executable
./example-name
```

### Index

| \# | Name                | Description                                                | New topics covered                                                                                       |
|:---|:--------------------|:-----------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|
| 1  | hello-world         | Library initialization                                     | scope guard, MPI communicator                                                                            |
| 2  | diffusion-2D        | Solve 2D diffusion problem                                 | simple mesh generation, algebraic system, solution manager, equation kernel, direct solver, VTK export   |
| 3  | advection-2D        | Solve 2D advection problem on prescribed velocity field    | space-dependent kernels, time stepping, accessing field values in kernels, Dirichlet BCs                 |
| 4  | periodic-bc         | Solve 2D advection problem in periodic domain              | periodic BCs, setting the initial solution, higher order time-stepping, computing the solution error     |
| 5  | static-condensation | Case from example \#4 with static condensation             | static condensation                                                                                      |
| 6  | matrix-free         | Case from example \#4 with matrix-free operator evaluation | matrix-free operator evaluation, iterative solvers, preconditioners                                      |
| 7  | karman-2D           | Solve 2D incompressible flow problem                       | reading the mesh from a gmsh file, exporting vector fields to VTK, simple post-processing                |
| 8  | native-io           | Save/load mesh and results in native L3STER format         | native I/O                                                                                               |