# 2D flow past a cylinder in a channel

*This example is somewhat advanced.*
*If you are just starting out with L3STER, you may wish to have a look at the advection example first.*

- Fully coupled 2D Navier-Stokes equations in the velocity-vorticity-pressure formulation [1]
- Dirichlet BC at the inlet - parabolic profile
- Dirichlet BC at the walls - no-slip
- Pseudo-traction (open) boundary condition at the outlet.
- Implicit 2nd order time discretization scheme (BDF2)
- Linearized using Newton's method
- Steady-state solution used as the initial condition

Note: the mesh file `karman.msh` is assumed to reside in the directory above the directory in which the program is run.
If you'd like to structure the program differently, pass the path to the mesh file as the first program argument.

[1] Jiang, B. (1998). The Least-Squares Finite Element Method. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-03740-9