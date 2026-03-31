"""
NS_Solver: 2D incompressible Navier-Stokes solver.

Components
----------
grid        : Cartesian MAC staggered grid (2D / 3D capable)
boundary    : Boundary condition types and application
operators   : Finite-volume spatial operators
poisson     : Pressure Poisson solver
ibm         : Immersed boundary method (direct forcing)
rk3         : 3rd-order SSP Runge-Kutta time integrator
solver      : Fractional-step (pressure-correction) solver
parallel    : MPI parallel utilities
io_utils    : I/O helpers (HDF5 / NumPy output)
"""
