# `algsys`

This directory contains facilities for assembling the sparse algebraic problems corresponding to the supplied PDEs and mesh.
This includes:

- creating the sparsity graph which describes the problem structure
- assembling the local system (in the domain and on the boundary)
- scattering the local systems into the global one (FE assembly)
- static condensation