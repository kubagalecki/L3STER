# 2D advection

In this example, we will show how to solve a 2D advection problem in a channel. The advection velocity field
follows a parabolic profile (Poiseuille flow). The initial species concentration is 0 in the entire domain. We impose a
Dirichlet boundary condition on the species concentration at the inflow boundary, prescribed as a constant.

```math
\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = 0
```

```math
\mathbf{u} = \begin{bmatrix} 0 \\ 1 - \left( 2y - 1 \right)^2 \end{bmatrix}
```