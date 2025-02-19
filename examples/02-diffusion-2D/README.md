# 2D Diffusion

In this example, we will solve a 2D diffusion problem with a constant source term and Robin boundary condition.
The domain will be the unit square: $\Omega=[0,1] \times [0,1]$.
This setup represents a uniformly heated plate with convectively cooled edges.
Assuming unit coefficients, the equations describing these physics are given by

$$\begin{align}
-\Delta \phi &= 1 \quad&\mathrm{in} \; \Omega\\
\frac{\partial\phi}{\partial n} &= -\phi \quad&\mathrm{on} \; \partial\Omega
\end{align}$$

Because L3STER (or rather, the underlying least-squares formulation) requires the equations be of the first order, we will first need to recast them as such.
We can achieve this by introducing $\mathbf{q}$, the gradient of $\phi$, as an auxiliary variable.
This yields the following system:

$$\begin{align}
\nabla \cdot \mathbf{q} &= 1 \quad &\\
\mathbf{q} - \nabla\phi &= \mathbf{0} &in \; \Omega\\
\nabla \times \mathbf{q} &= \mathbf{0}&\\
\phi + \mathbf{q} \cdot \mathbf{n} &= 0 &on \; \partial\Omega
\end{align}$$

The third equation is a curl constraint (see [1] for details).
Note that this system has more equations than unknowns.
This is not a problem, as these parameters are independent in L3STER.

[1] Jiang, B. (1998). The Least-Squares Finite Element Method. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-03740-9