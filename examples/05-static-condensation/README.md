# 2D Advection in periodic domain

In this example, we will show how to solve a 2D advection problem in a periodic rectangular domain.
The domain is periodic in the x direction.
Homogenous Neuman conditions are imposed at the top and bottom.

$$ \begin{align}
\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi &= 0 &\mathrm{in} \; \Omega=[-1,1] \times [0,1] \\
\frac{\partial\phi}{\partial n} &= 0 & \mathrm{on} \; \partial \Omega_N
\end{align} $$

The advection velocity is constant and horizontal

$$ \mathbf{u} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} $$

The initial species concentration follows a Gaussian curve

$$ \phi(t=0,x,y) = \exp\left(-kx^2\right) $$

This is essentially the 1D advection equation with constant unit advection velocity, which possesses an analytical solution given by

$$ \phi(t,x,y) = \phi(0,x-t,y) = \exp\left(-k(x-t)^2\right) $$

The time derivative is discretized using the 3rd order backward differentiation formula (BDF3):

$$ \frac{\partial u_{n+1}}{\partial t} \approx \frac{1}{\Delta t} \left( \frac{11}{6}u_{n+1} - 3u_{n} + \frac{3}{2}u_{n-1} - \frac{1}{3}u_{n-2} \right)$$

### Static condensation

The problem description in this example is the same as in example #4.
The difference lies in the approach to assembling the algebraic system.
Here, we statically condense [1] the degrees of freedom corresponding to internal element nodes.
Compared to example #4, we can notice the following:

- Reduced algebraic system size
- Faster solution time
- Only 2 lines of code chaged (you can use `diff` to see the difference)

Note that static condensation will not be beneficial (in fact, it might be detrimental) for low-order meshes.

[1] Guyan, R. J. (1965). Reduction of stiffness and mass matrices. AIAA Journal, 3(2), 380–380. https://doi.org/10.2514/3.2874