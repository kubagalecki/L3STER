# `algsys`

This directory contains facilities for managing the algebraic problems corresponding to the supplied PDEs and
mesh.
This includes:

- creating the sparsity graph which describes the problem structure
- computing the local system (in the domain and on the boundary)
- assembling the global sparse system
- static condensation
- matrix-free facilities