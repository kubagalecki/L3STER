# L3STER prerequisites

### Compiler

L3STER is written entirely in C++. To compile it, you'll need a compiler which supports the C++17 standard.

### L3STER depends on the following libraries:

- Eigen for dense serial algebra (it is included as a module in L3STER, no installation required)
- METIS for mesh partitioning
- MPI for distributed communication between processes
- Trilinos for scalable linear solvers. The following packages ae required:
    - Tpetra for sparse algebra
    - MueLu for AMG preconditioners
    - Kokkos for intra-node parallelism

