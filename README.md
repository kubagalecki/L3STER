[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Tests](https://github.com/kubagalecki/L3STER/workflows/tests/badge.svg)](https://github.com/kubagalecki/L3STER/actions)
[![codecov](https://codecov.io/gh/kubagalecki/L3STER/branch/main/graph/badge.svg?token=6VT1TVS7FG)](https://codecov.io/gh/kubagalecki/L3STER)

# L3STER :cheese:

L3STER (pronounced like the delicious English cheese) stands for "**L**east-**S**quares **S**calable **S**pec**T**ral/*hp* **E**lement f**R**amework".
L3STER provides a scalable, flexible framework for the solution of systems of partial differential equations.
Thanks to the employment of the least-squares finite element method, no weak formulation is needed - you can directly implement any first-order PDE:

$$ \mathcal{A}(u) = f \hspace{.1cm} \mathrm{in} \hspace{.1cm} \Omega $$

$$ \mathcal{B}(u) = g \hspace{.1cm} \mathrm{on} \hspace{.1cm} \partial\Omega $$

The guiding philosophy of the project is: *"From a set of PDEs and a mesh to a working simulation within an afternoon!"*

Features of the library include:
- A modern implementation leveraging C++23
- Scalability using hybrid parallelism (MPI + multithreading)
- Computational efficiency thanks to a high-order discretization
- Mesh import from Gmsh
- Results export to VTK, simple postprocessing (flux integrals etc.) available natively
- Easy setup (all dependencies available in Spack)

If you'd like to use L3STER, but you need a different I/O format, please drop us an issue!

## Formulating the system of PDEs

If your equation is of a higher order, you'll first need to recast it by introducing auxiliary unknowns (e.g. gradients).
At the end of the day, each equation takes the form of:

$$ A_0 u + \left( \sum_{i=1}^{D} A_i \frac{\partial}{\partial x_i} \right) u = f $$

where $D \in \{ 2,3 \}$ is the spatial dimension of the problem,
$u : \Omega \rightarrow \mathbb{R}^U$ is the unknown vector field,
$A_i : \Omega \rightarrow \mathbb{R}^{E \times U}$ describe the first-order differential operator,
$f : \Omega \rightarrow \mathbb{R}^E$ is the source term,
$E$ is the number of equations, and $U$ the number of unknowns ($E$ and $U$ may not be equal).

### Boundary conditions

You can also use the approach outlined above to describe an arbitrary boundary condition.
In L3STER, the only difference between domain equations and boundary conditions is that when defining BCs, you gain access to the boundary normal vector.

Only Dirichlet BCs are treated in a special fashion.
This is because they are strongly imposed on the resulting algebraic system.
It is possible to define them in the equation sense, but this is not recommended.

### Time-dependent problems

Note that this formulation does not contain a time derivative.
If you are solving an unsteady problem, you'll first need to discretize your problem in time.
For example, you can use the backward Euler scheme:

$$ \frac{\partial u}{\partial t} \approx \frac{u_{n+1} - u_n}{\Delta t} $$

You can then add $I \Delta t$ to $A_0$ and add $u_n / \Delta t$ to the source term to obtain a PDE for $u$ at the next time step.

### Non-linear problems

If your problem is non-linear, you'll first need to linearize it, e.g., using Newton's method.
You can then iterate to obtain your solution.
L3STER provides a convenient way of accessing previously computed fields (and their derivatives) when defining your equations.
This mechanism can also be used for time-stepping (where the previous value(s) of $u$ are needed) or dependencies between different systems, e.g., solving an advection-diffusion equation on a previously computed flow field.

## Installation

L3STER is a header-only library, which means you don't need to install it.
Simply point your CMake project at the directory where L3STER resides and use the provided target:

```cmake
# In your project's CMakeLists.txt
add_subdirectory( path/to/L3STER L3STER-bin )
target_link_libraries( my-executable-target L3STER )
```

That being said, L3STER has several dependencies, which will need to be installed first:

- CMake 3.24 or newer
- A C++ 23 compliant compiler, gcc 14 or newer will work
- MPI
- Hwloc
- Metis
- Trilinos 14.0 or newer. The following packages are currently used:
  - Kokkos (which can be built separately from Trilinos)
  - Tpetra
  - Belos - optional, needed for iterative solvers
  - Amesos2 - optional, needed for direct solvers
  - Ifpack2 - optional, used for preconditioners (L3STER provides a few simple ones natively)
- Intel OneTBB
- Eigen version 3.4

All of these dependencies are available via [Spack](https://spack.readthedocs.io/en/latest/index.html).
You can easily install them as follows:

```bash
# Get spack and set up the shell
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
cd spack
git checkout tags/releases/latest
. share/spack/setup-env.sh # consider adding this to your .bashrc

# Find some common packages so that spack doesn't have to build them from scratch (saves time)
spack external find binutils cmake coreutils curl diffutils findutils git gmake openssh perl python sed tar

# If you have a sufficiently recent compiler, skip this section
spack install gcc
spack load gcc
spack compiler find

# Create a spack environment for L3STER dependencies and install them
# Some of the libraries listed above are not mentioned explicitly, they will be built as dependencies of other packages
spack env create l3ster
spacktivate l3ster
spack add eigen intel-oneapi-tbb parmetis kokkos+openmp trilinos cxxstd=17 +openmp +amesos2 +belos +tpetra +ifpack2
spack concretize
spack install

# Cleanup to save disc space
spack gc -y
spack clean -a
```

When using L3STER, all you need to do is call `spacktivate l3ster` before invoking CMake.

> Your cluster administrators may provide a global spack instance.
> You can take advantage of it using spack chaining.
> If not, you should use the MPI installation provided for you by the admins, not build your own.
> Please consult the spack documentation on how to use external packages.

## Running L3STER applications

L3STER follows the MPI+X paradigm (hybrid parallelism).
It uses TBB for multithreading, and MPI for multiprocessing.
It is recommended that you launch one MPI rank per CPU, not CPU *core*.

> Trilinos uses OpenMP for multithreading. L3STER and Trilinos parallel regions never overlap, so oversubscription is not an issue.

> L3STER uses Hwloc to detect your machine's topology and limits the SMT parallelism where appropriate. You don't need to worry about hyperthreading, L3STER will just do the right thing.

### Desktops

On desktops, where presumably you have only one CPU, you can launch your application directly.
Parallelism is achieved via multithreading on a single MPI rank.

```bash
./my-l3ster-app
```

Note that you still need to build with MPI (sorry).

### Clusters

Example slurm script demonstrating L3STER usage:

```bash
#SBATCH -N [number of nodes you'd like to run on]
#SBATCH -n [number of nodes you'd like to run on multiplied by number of sockets/node (often 2)]
#SBATCH -c [number of cores per socket]
#SBATCH --ntasks-per-socket 1 

# Set up environment via spack and/or system modules

cd /my/project/dir
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpic++ .. || exit 1
cmake --build .                                                 || exit 1
srun my-l3ster-app
```

## Usage

We are working on fully documenting the L3STER library.
For the time being, please refer to the examples.