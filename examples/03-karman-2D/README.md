# 2D flow past a cylinder in a channel

*This example is somewhat advanced.*
*If you are just starting out with L3STER, you may want to have a look at the advection example first.*

- Fully coupled 2D Navier-Stokes equations in the velocity-vorticity-pressure formulation [1]
- Momentum equation is scaled by the time step to improve mass conservation [2]
- Dirichlet BC at the inlet - parabolic profile
- Dirichlet BC at the walls - no-slip
- Pseudo-traction (open) boundary condition at the outlet
- Implicit 2nd order time discretization scheme (BDF2)
- Convective term linearized using Newton's method
- Steady-state solution used as the initial condition

Note: the mesh file `karman.msh` is assumed to reside in the directory above the directory in which the program is run.
If you'd like to use a different setup, pass the path to the mesh file as the first program argument.

[1] Jiang, B. (1998). The Least-Squares Finite Element Method. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-03740-9

[2] Pontaza, J. P. (2006). A least-squares finite element formulation for unsteady incompressible flows with improved velocity–pressure coupling. Journal of Computational Physics, 217(2), 563–588. https://doi.org/10.1016/j.jcp.2006.01.013