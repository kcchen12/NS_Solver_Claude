# Mathematical Formulas

This note collects the main equations used in this project and writes them in Markdown-friendly LaTeX.

## 1. Incompressible Navier-Stokes Equations

The solver advances the 2-D incompressible Navier-Stokes equations:

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
\frac{\partial \mathbf{u}}{\partial t}
+ (\mathbf{u} \cdot \nabla)\mathbf{u}
= -\nabla p + \nu \nabla^2 \mathbf{u}
$$

where

$$
\mathbf{u} = \begin{bmatrix} u \\ v \end{bmatrix}
$$

and:

$$
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}
$$

$$
\nabla^2 \mathbf{u} =
\begin{bmatrix}
\nabla^2 u \\
\nabla^2 v
\end{bmatrix}
$$

## 2. Reynolds Number

The Reynolds number is defined as

$$
Re = \frac{U_\infty L}{\nu}
$$

where:

- $U_\infty$ is the reference or inflow velocity
- $L$ is the characteristic length
- $\nu$ is the kinematic viscosity

For the nondimensional unit-scale case used by the solver,

$$
\nu = \frac{1}{Re}
$$

## 3. Fractional-Step / Projection Method

The solver uses a fractional-step method with SSP-RK3 for the convection-diffusion part.

### 3.1 Intermediate Velocity

Ignoring pressure during the Runge-Kutta stage:

$$
\mathbf{u}^* = \mathbf{u}^n + \Delta t \, \mathrm{RK3}\!\left[-(\mathbf{u}\cdot\nabla)\mathbf{u} + \nu \nabla^2 \mathbf{u}\right]
$$

### 3.2 Pressure Poisson Equation

The pressure correction $\phi$ satisfies

$$
\nabla^2 \phi = \frac{\nabla \cdot \mathbf{u}^*}{\Delta t}
$$

### 3.3 Velocity Correction

The corrected velocity is

$$
\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t \, \nabla \phi
$$

In component form:

$$
u^{n+1}_{i,j} = u^*_{i,j} - \Delta t \, \frac{\phi_{i,j} - \phi_{i-1,j}}{\Delta x}
$$

$$
v^{n+1}_{i,j} = v^*_{i,j} - \Delta t \, \frac{\phi_{i,j} - \phi_{i,j-1}}{\Delta y}
$$

### 3.4 Pressure Update

$$
p^{n+1} = p^n + \phi
$$

## 4. CFL Estimate

The solver estimates the convective CFL number as

$$
\mathrm{CFL}_x = \frac{\max |u| \, \Delta t}{\Delta x}
$$

$$
\mathrm{CFL}_y = \frac{\max |v| \, \Delta t}{\Delta y}
$$

$$
\mathrm{CFL} = \max(\mathrm{CFL}_x,\mathrm{CFL}_y)
$$

## 5. Cylinder Pressure Forces

For a cylinder, the pressure force on the body is computed from the surface integral

$$
\mathbf{F} = - \int_{\partial \Omega_b} p \, \mathbf{n} \, ds
$$

with components

$$
F_x = - \int_{\partial \Omega_b} p \, n_x \, ds
$$

$$
F_y = - \int_{\partial \Omega_b} p \, n_y \, ds
$$

## 6. Drag and Lift Coefficients

Using dynamic pressure

$$
q = \frac{1}{2}\rho U^2
$$

the drag and lift coefficients are

$$
C_D = \frac{2F_x}{qA}
$$

$$
C_L = \frac{2F_y}{qA}
$$

In this project, the reference area is taken as the characteristic length in 2-D:

$$
A = L
$$

so equivalently

$$
C_D = \frac{2F_x}{\tfrac{1}{2}\rho U^2 L}
$$

$$
C_L = \frac{2F_y}{\tfrac{1}{2}\rho U^2 L}
$$

## 7. Strouhal Number

The Strouhal number is computed from the dominant shedding frequency:

$$
St = \frac{fL}{U}
$$

where:

- $f$ is the dominant oscillation frequency
- $L$ is the characteristic length
- $U$ is the reference velocity

## 8. Vorticity

The 2-D scalar vorticity used in plotting is

$$
\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
$$
