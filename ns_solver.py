"""
2D Incompressible Navier-Stokes Solver

Implements the projection (fractional step) method on a uniform collocated
finite-difference grid.  Two boundary-condition modes are supported:

* ``'no_slip'``  – Dirichlet no-slip walls; pressure Poisson equation solved
  with a sparse direct solver (suitable for lid-driven cavity flow).
* ``'periodic'`` – Periodic boundaries; pressure Poisson equation solved with
  a 2-D FFT (suitable for Taylor-Green vortex validation).

The non-dimensional governing equations are:

    ∂u/∂t + (u·∇)u = −∇p + (1/Re)∇²u
    ∇·u = 0

Reference:
    Chorin, A.J. (1968). Numerical solution of the Navier-Stokes equations.
    Mathematics of Computation, 22(104), 745-762.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class NSSolver2D:
    """2D Incompressible Navier-Stokes Solver.

    Uses the explicit-Euler projection method on a collocated uniform grid
    with second-order central finite differences.

    Parameters
    ----------
    nx : int
        Number of grid points (no-slip) or cells (periodic) in x-direction.
    ny : int
        Number of grid points (no-slip) or cells (periodic) in y-direction.
    Re : float
        Reynolds number (Re = U L / ν).
    dt : float
        Time step size.  Must satisfy the CFL and diffusive stability
        conditions for the chosen resolution and Reynolds number.
    Lx : float, optional
        Domain length in x.  Default ``1.0``.
    Ly : float, optional
        Domain length in y.  Default ``1.0``.
    bc : {'no_slip', 'periodic'}, optional
        Boundary condition type.  Default ``'no_slip'``.

    Attributes
    ----------
    u : ndarray, shape (ny, nx)
        x-component of velocity.
    v : ndarray, shape (ny, nx)
        y-component of velocity.
    p : ndarray, shape (ny, nx)
        Pressure field.
    X, Y : ndarray, shape (ny, nx)
        Coordinate meshgrids.

    Examples
    --------
    Lid-driven cavity at Re = 100:

    >>> solver = NSSolver2D(nx=33, ny=33, Re=100, dt=0.001)
    >>> solver.set_lid_driven_cavity(u_lid=1.0)
    >>> u, v, p = solver.run(T=1.0)

    Taylor-Green vortex:

    >>> import numpy as np
    >>> solver = NSSolver2D(nx=32, ny=32, Re=10, dt=0.001,
    ...                    Lx=2*np.pi, Ly=2*np.pi, bc='periodic')
    >>> solver.set_initial_condition(
    ...     np.sin(solver.X) * np.cos(solver.Y),
    ...     -np.cos(solver.X) * np.sin(solver.Y))
    >>> u, v, p = solver.run(T=0.5)
    """

    def __init__(self, nx, ny, Re, dt, Lx=1.0, Ly=1.0, bc='no_slip'):
        if bc not in ('no_slip', 'periodic'):
            raise ValueError(f"bc must be 'no_slip' or 'periodic', got '{bc}'")

        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.dt = dt
        self.Lx = Lx
        self.Ly = Ly
        self.bc = bc

        if bc == 'periodic':
            self.dx = Lx / nx
            self.dy = Ly / ny
            x = np.arange(nx) * self.dx
            y = np.arange(ny) * self.dy
        else:
            self.dx = Lx / (nx - 1)
            self.dy = Ly / (ny - 1)
            x = np.linspace(0, Lx, nx)
            y = np.linspace(0, Ly, ny)

        self.x = x
        self.y = y
        self.X, self.Y = np.meshgrid(x, y)

        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

        self._u_lid = 0.0

        if bc == 'periodic':
            self._setup_periodic_solver()
        else:
            self._setup_dirichlet_solver()

    # ------------------------------------------------------------------
    # Solver setup
    # ------------------------------------------------------------------

    def _setup_periodic_solver(self):
        """Precompute FFT eigenvalues for the periodic Poisson solver."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        kx = np.fft.fftfreq(nx) * nx   # integer wavenumbers
        ky = np.fft.fftfreq(ny) * ny
        KX, KY = np.meshgrid(kx, ky)

        # Eigenvalues of the 2-D discrete Laplacian with periodic BCs
        self._lap_eig = (
            2 * (np.cos(2 * np.pi * KX / nx) - 1) / dx ** 2
            + 2 * (np.cos(2 * np.pi * KY / ny) - 1) / dy ** 2
        )
        # Avoid division by zero at (0,0); p_hat[0,0] = 0 enforces zero mean
        self._lap_eig[0, 0] = 1.0

    def _setup_dirichlet_solver(self):
        """Build sparse Laplacian matrix for the pressure Poisson equation.

        Neumann (dp/dn = 0) boundary conditions are applied by using
        one-sided stencils at domain boundaries.  One pressure value is
        pinned at the corner to remove the null-space singularity.
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        N = nx * ny

        A = lil_matrix((N, N))

        for j in range(ny):
            for i in range(nx):
                row = j * nx + i
                diag = 0.0

                if i > 0:
                    A[row, row - 1] = 1.0 / dx ** 2
                    diag -= 1.0 / dx ** 2
                if i < nx - 1:
                    A[row, row + 1] = 1.0 / dx ** 2
                    diag -= 1.0 / dx ** 2
                if j > 0:
                    A[row, row - nx] = 1.0 / dy ** 2
                    diag -= 1.0 / dy ** 2
                if j < ny - 1:
                    A[row, row + nx] = 1.0 / dy ** 2
                    diag -= 1.0 / dy ** 2

                A[row, row] = diag

        # Pin pressure at (0, 0) to remove the constant null-space
        A[0, :] = 0
        A[0, 0] = 1.0

        self._A = csr_matrix(A)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_initial_condition(self, u0, v0):
        """Set the initial velocity field.

        Parameters
        ----------
        u0 : array_like, shape (ny, nx)
            Initial x-velocity.
        v0 : array_like, shape (ny, nx)
            Initial y-velocity.

        Raises
        ------
        ValueError
            If the shape of ``u0`` or ``v0`` does not match the grid.
        """
        u0 = np.asarray(u0, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        expected = (self.ny, self.nx)
        if u0.shape != expected or v0.shape != expected:
            raise ValueError(
                f"Expected shape {expected}, got u0={u0.shape}, v0={v0.shape}"
            )
        self.u = u0.copy()
        self.v = v0.copy()

    def set_lid_driven_cavity(self, u_lid=1.0):
        """Configure the lid-driven cavity problem.

        The top wall moves at ``u_lid`` in the x-direction.  All other
        walls have zero-velocity (no-slip) conditions.

        Parameters
        ----------
        u_lid : float, optional
            Lid velocity.  Default ``1.0``.

        Raises
        ------
        ValueError
            If the solver was not created with ``bc='no_slip'``.
        """
        if self.bc != 'no_slip':
            raise ValueError("set_lid_driven_cavity requires bc='no_slip'")
        self._u_lid = u_lid
        self._apply_bc()

    def step(self):
        """Advance the solution by one time step.

        Algorithm (Chorin projection method):

        1. **Predictor** – compute intermediate velocity ``u*`` from the
           advection-diffusion equation, ignoring the pressure gradient.
        2. **Pressure solve** – solve the Poisson equation
           ``∇²p = (1/dt) ∇·u*``.
        3. **Corrector** – project ``u*`` onto the divergence-free space:
           ``u = u* − dt ∇p``.

        Returns
        -------
        u, v, p : ndarray, shape (ny, nx)
            Updated velocity and pressure fields.
        """
        dt = self.dt
        dx, dy = self.dx, self.dy

        # Step 1: Intermediate velocity (explicit Euler, no pressure)
        Fu, Fv = self._rhs()
        u_star = self.u + dt * Fu
        v_star = self.v + dt * Fv
        self.u, self.v = u_star, v_star
        self._apply_bc()
        u_star, v_star = self.u.copy(), self.v.copy()

        # Step 2: Divergence of u* → pressure Poisson
        div = self._divergence_of(u_star, v_star)
        p_new = self._solve_poisson(div / dt)

        # Step 3: Velocity correction
        dp_dx, dp_dy = self._gradient(p_new)
        if self.bc == 'periodic':
            self.u = u_star - dt * dp_dx
            self.v = v_star - dt * dp_dy
        else:
            nx, ny = self.nx, self.ny
            i, j = slice(1, nx - 1), slice(1, ny - 1)
            self.u = u_star.copy()
            self.v = v_star.copy()
            self.u[j, i] -= dt * dp_dx[j, i]
            self.v[j, i] -= dt * dp_dy[j, i]

        self.p = p_new
        self._apply_bc()
        return self.u, self.v, self.p

    def run(self, T):
        """Run the solver for a total time ``T``.

        Parameters
        ----------
        T : float
            Total simulation time.

        Returns
        -------
        u, v, p : ndarray
            Final velocity and pressure fields.
        """
        nt = int(T / self.dt)
        for _ in range(nt):
            self.step()
        return self.u, self.v, self.p

    def divergence(self):
        """Compute the divergence ∇·u of the current velocity field.

        For incompressible flow this should be near zero everywhere after
        each projection step.

        Returns
        -------
        div : ndarray, shape (ny, nx)
        """
        return self._divergence_of(self.u, self.v)

    def kinetic_energy(self):
        """Compute the total kinetic energy ½ ∫∫ (u² + v²) dA.

        Returns
        -------
        float
        """
        return 0.5 * np.sum(self.u ** 2 + self.v ** 2) * self.dx * self.dy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_bc(self):
        """Apply boundary conditions to the velocity field."""
        if self.bc == 'no_slip':
            self.u[0, :] = 0.0       # bottom – no-slip
            self.v[0, :] = 0.0
            self.u[-1, :] = self._u_lid   # top – moving lid
            self.v[-1, :] = 0.0
            self.u[:, 0] = 0.0       # left – no-slip
            self.v[:, 0] = 0.0
            self.u[:, -1] = 0.0      # right – no-slip
            self.v[:, -1] = 0.0
        # For periodic BC, np.roll handles wrap-around; no explicit BC needed.

    def _rhs(self):
        """Compute the advection-diffusion RHS of the momentum equations.

        Returns
        -------
        Fu, Fv : ndarray
            Right-hand side for u and v.  For no-slip BCs only interior
            points are non-zero.
        """
        u, v = self.u, self.v
        dx, dy, Re = self.dx, self.dy, self.Re

        if self.bc == 'periodic':
            u_xp = np.roll(u, -1, axis=1)
            u_xm = np.roll(u, 1, axis=1)
            u_yp = np.roll(u, -1, axis=0)
            u_ym = np.roll(u, 1, axis=0)
            v_xp = np.roll(v, -1, axis=1)
            v_xm = np.roll(v, 1, axis=1)
            v_yp = np.roll(v, -1, axis=0)
            v_ym = np.roll(v, 1, axis=0)

            du_dx = (u_xp - u_xm) / (2 * dx)
            du_dy = (u_yp - u_ym) / (2 * dy)
            dv_dx = (v_xp - v_xm) / (2 * dx)
            dv_dy = (v_yp - v_ym) / (2 * dy)

            d2u = (u_xp - 2 * u + u_xm) / dx ** 2 + (u_yp - 2 * u + u_ym) / dy ** 2
            d2v = (v_xp - 2 * v + v_xm) / dx ** 2 + (v_yp - 2 * v + v_ym) / dy ** 2

            Fu = -u * du_dx - v * du_dy + d2u / Re
            Fv = -u * dv_dx - v * dv_dy + d2v / Re
        else:
            nx, ny = self.nx, self.ny
            Fu = np.zeros_like(u)
            Fv = np.zeros_like(v)
            i, j = slice(1, nx - 1), slice(1, ny - 1)

            du_dx = (u[j, 2:] - u[j, :-2]) / (2 * dx)
            du_dy = (u[2:, i] - u[:-2, i]) / (2 * dy)
            dv_dx = (v[j, 2:] - v[j, :-2]) / (2 * dx)
            dv_dy = (v[2:, i] - v[:-2, i]) / (2 * dy)

            d2u = (
                (u[j, 2:] - 2 * u[j, i] + u[j, :-2]) / dx ** 2
                + (u[2:, i] - 2 * u[j, i] + u[:-2, i]) / dy ** 2
            )
            d2v = (
                (v[j, 2:] - 2 * v[j, i] + v[j, :-2]) / dx ** 2
                + (v[2:, i] - 2 * v[j, i] + v[:-2, i]) / dy ** 2
            )

            Fu[j, i] = -u[j, i] * du_dx - v[j, i] * du_dy + d2u / Re
            Fv[j, i] = -u[j, i] * dv_dx - v[j, i] * dv_dy + d2v / Re

        return Fu, Fv

    def _divergence_of(self, u, v):
        """Compute ∇·(u, v) using central differences."""
        dx, dy = self.dx, self.dy
        nx, ny = self.nx, self.ny

        if self.bc == 'periodic':
            du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
            dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
            return du_dx + dv_dy
        else:
            div = np.zeros((ny, nx))
            i, j = slice(1, nx - 1), slice(1, ny - 1)
            div[j, i] = (
                (u[j, 2:] - u[j, :-2]) / (2 * dx)
                + (v[2:, i] - v[:-2, i]) / (2 * dy)
            )
            return div

    def _gradient(self, p):
        """Compute the gradient (∂p/∂x, ∂p/∂y) using central differences."""
        dx, dy = self.dx, self.dy
        if self.bc == 'periodic':
            dp_dx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dx)
            dp_dy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dy)
        else:
            nx, ny = self.nx, self.ny
            dp_dx = np.zeros_like(p)
            dp_dy = np.zeros_like(p)
            i, j = slice(1, nx - 1), slice(1, ny - 1)
            dp_dx[j, i] = (p[j, 2:] - p[j, :-2]) / (2 * dx)
            dp_dy[j, i] = (p[2:, i] - p[:-2, i]) / (2 * dy)
        return dp_dx, dp_dy

    def _solve_poisson(self, rhs):
        """Solve ∇²p = rhs for the pressure field.

        Parameters
        ----------
        rhs : ndarray, shape (ny, nx)
            Right-hand side of the Poisson equation.

        Returns
        -------
        p : ndarray, shape (ny, nx)
        """
        if self.bc == 'periodic':
            rhs_hat = np.fft.fft2(rhs)
            p_hat = rhs_hat / self._lap_eig
            p_hat[0, 0] = 0.0    # zero mean pressure
            return np.real(np.fft.ifft2(p_hat))
        else:
            b = rhs.ravel().copy()
            b[0] = 0.0           # pressure pin at corner
            return spsolve(self._A, b).reshape((self.ny, self.nx))
