# 2D Advection

In this example, we will show how to solve a 2D advection problem in a channel.

$$ \begin{align}
\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi &= 0 &\mathrm{in} \; \Omega=[0,3] \times [0,1] \\
\phi &= \phi_D & \mathrm{on} \; \partial \Omega_D
\end{align} $$

The advection velocity field follows a parabolic profile (Poiseuille flow).

$$ \mathbf{u} = \begin{bmatrix} 1 - \left( 2y - 1 \right)^2 \\
0 \end{bmatrix} $$

The initial species concentration is 0 in the entire domain.

$$ \phi \big|_{t=0} = 0 $$

We impose a constant Dirichlet boundary condition on the species concentration at the channel inlet.

$$ \phi \big|_{x=0} = 1 $$