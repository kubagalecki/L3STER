[![Tests](https://github.com/kubagalecki/L3STER/workflows/Tests/badge.svg)](https://github.com/kubagalecki/L3STER/actions)
[![codecov](https://codecov.io/gh/kubagalecki/L3STER/branch/master/graph/badge.svg?token=6VT1TVS7FG)](https://codecov.io/gh/kubagalecki/L3STER)

# L3STER

**Currently under construction.**

The name L3STER is derived from "Least-Squares Scalable SpecTral Element fRamework".
The goal of the project is to develop, based on the least-squares spectral *h/p* element method, a scalable C++ framework for the numerical solution of partial differential equations.
Although the aim is to develop a generic framework, the author's intended classes of applications are as follows:
- incompressible flow with stabilized boundary conditions
- optimal control problems involving fluid-structure interaction

Some attractive features of the code include:
- utilization of state of the art algebraic solvers available in the Trilinos library
- modern implementation leveraging features from C++20
- linear scaling with problem size thanks to the synergistic combination of the least-squares method with a multigrid solver (theoretically)
- L3STER is header-only, allowing you to take full advantage of your compiler's optimizer
- a low entry barrier - given a set of PDEs and a mesh, you should be able to set up your first simulation within an afternoon!
