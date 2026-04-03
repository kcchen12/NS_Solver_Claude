"""
Fractional-step (projection) solver for the 2-D incompressible
Navier-Stokes equations.

Algorithm
---------
Given (uⁿ, vⁿ, pⁿ) advance one time step Δt:

1. **Intermediate velocity** — apply SSP-RK3 to the convection-diffusion
   operator *without* pressure:

       u* = uⁿ + Δt · RK3[ -(u·∇)u + ν ∇²u ]

   Boundary conditions are enforced after each RK sub-stage.
   IBM direct forcing is applied after each sub-stage (if bodies present).

2. **Pressure Poisson** — solve for the pressure correction φ:

       ∇²φ = ∇·u* / Δt

   with Neumann BCs (∂φ/∂n = 0) at solid/inflow/far-field faces and
   Dirichlet (φ = 0) at outflow faces.

3. **Velocity correction**:

       uⁿ⁺¹[i,j] = u*[i,j] − Δt · (φ[i,j] − φ[i-1,j]) / dx   (interior x-faces)
       vⁿ⁺¹[i,j] = v*[i,j] − Δt · (φ[i,j] − φ[i,j-1]) / dy   (interior y-faces)

4. **Pressure update**:

       pⁿ⁺¹ = pⁿ + φ

5. Apply boundary conditions to uⁿ⁺¹, vⁿ⁺¹.

Reference
---------
Kim & Moin (1985), "Application of a fractional-step method to
incompressible Navier-Stokes equations", J. Comput. Phys. 59, 308–323.
"""

import numpy as np

from src.grid import CartesianGrid
from src.boundary import BoundaryConfig, apply_velocity_bc, apply_pressure_bc, \
    apply_post_correction_bc
from src.operators import rhs_u, rhs_v, divergence, laplacian_u, laplacian_v
from src.poisson import PoissonSolver
from src.rk3 import ssp_rk3_step
from src.ibm import ImmersedBoundary


class FractionalStepSolver:
    """
    Incompressible Navier-Stokes solver using the fractional-step method
    with SSP-RK3 time integration and finite-volume spatial discretisation.

    Parameters
    ----------
    grid : CartesianGrid
    bc   : BoundaryConfig
    nu   : float
        Kinematic viscosity (ν = 1/Re for unit velocity/length scales).
    ibm  : ImmersedBoundary, optional
        Immersed-boundary object.  Pass ``None`` (default) for no solid body.
    """

    def __init__(self, grid: CartesianGrid, bc: BoundaryConfig,
                 nu: float,
                 ibm: ImmersedBoundary = None):
        self.grid = grid
        self.bc = bc
        self.nu = nu
        self.ibm = ibm
        self._poisson = PoissonSolver(grid, bc)

        # Field arrays (initialised by caller via init_fields)
        self.u = grid.zeros_u()
        self.v = grid.zeros_v()
        self.p = grid.zeros_p()
        self.t = 0.0
        self.last_ibm_force_x = 0.0
        self.last_ibm_force_y = 0.0

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def init_fields(self, u0: float = 1.0, v0: float = 0.0,
                    p0: float = 0.0) -> None:
        """Initialise all fields to uniform values."""
        self.u[:] = u0
        self.v[:] = v0
        self.p[:] = p0
        # Enforce BCs on initial state
        apply_velocity_bc(self.u, self.v, self.grid, self.bc)
        self.t = 0.0
        self.last_ibm_force_x = 0.0
        self.last_ibm_force_y = 0.0

    # ------------------------------------------------------------------
    # Single time-step advance
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """
        Advance the solution by one time step *dt*.

        Modifies ``self.u``, ``self.v``, ``self.p``, ``self.t`` in-place.
        """
        grid = self.grid
        bc = self.bc
        nu = self.nu
        self.last_ibm_force_x = 0.0
        self.last_ibm_force_y = 0.0
        # SOFT_FIX_NONUNIFORM_IBM (search token)
        # Stabilization knobs used only for non-uniform IBM runs.
        # If future method improvements are added, replace/remove this block.
        ibm_relax = 0.3 if (
            self.ibm is not None and self.ibm.has_solid and grid.is_nonuniform) else 1.0
        nu_stab_floor = 0.03

        def _expand_mask(mask: np.ndarray) -> np.ndarray:
            # SOFT_FIX_NONUNIFORM_IBM: one-ring mask expansion.
            out = mask.copy()
            out[:-1, :] |= mask[1:, :]
            out[1:, :] |= mask[:-1, :]
            out[:, :-1] |= mask[:, 1:]
            out[:, 1:] |= mask[:, :-1]
            return out

        # ----------------------------------------------------------------
        # Step 1 — SSP-RK3 for convection-diffusion (no pressure)
        # ----------------------------------------------------------------
        def _rhs(u, v):
            """RHS callable for the RK integrator."""
            apply_velocity_bc(u, v, grid, bc, dt=dt)
            if self.ibm is not None and self.ibm.has_solid:
                self.ibm.apply(u, v, relax=ibm_relax)
            Lu = rhs_u(u, v, grid, bc, nu)
            Lv = rhs_v(u, v, grid, bc, nu)
            if self.ibm is not None and self.ibm.has_solid:
                if grid.is_nonuniform and nu < nu_stab_floor:
                    # SOFT_FIX_NONUNIFORM_IBM: supplemental diffusion floor.
                    # Extra dissipation stabilises IBM on stretched grids.
                    nu_extra = nu_stab_floor - nu
                    Lu += nu_extra * laplacian_u(u, grid, bc)
                    Lv += nu_extra * laplacian_v(v, grid, bc)
                # Do not advance momentum inside the immersed solid region.
                Lu[self.ibm.mask_u[1:-1, :]] = 0.0
                Lv[self.ibm.mask_v[:, 1:-1]] = 0.0
                if grid.is_nonuniform:
                    # SOFT_FIX_NONUNIFORM_IBM: damping ring around IBM interface.
                    # Extra damping in a one-cell ring around IBM masks to
                    # stabilise steep metric/forcing transitions.
                    mu = _expand_mask(self.ibm.mask_u[1:-1, :])
                    mv = _expand_mask(self.ibm.mask_v[:, 1:-1])
                    Lu[mu] = 0.0
                    Lv[mv] = 0.0
            return Lu, Lv

        u_star, v_star = ssp_rk3_step(_rhs, self.u, self.v, dt)

        # Apply BCs and IBM to the intermediate velocity
        apply_velocity_bc(u_star, v_star, grid, bc, dt=dt)
        if self.ibm is not None and self.ibm.has_solid:
            self.last_ibm_force_x, self.last_ibm_force_y = self.ibm.apply(
                u_star, v_star, dt=dt, relax=ibm_relax
            )

        # ----------------------------------------------------------------
        # Step 2 — Solve pressure Poisson:  ∇²φ = ∇·u* / dt
        # ----------------------------------------------------------------
        div_star = divergence(u_star, v_star, grid)
        rhs_poisson = div_star / dt

        phi = self._poisson.solve(rhs_poisson)
        apply_pressure_bc(phi, grid, bc)

        # ----------------------------------------------------------------
        # Step 3 — Velocity correction
        # ----------------------------------------------------------------
        # Interior x-faces: i = 1 .. nx-1
        u_new = u_star.copy()
        u_new[1:-1, :] -= dt * (phi[1:, :] - phi[:-1, :]) / grid.dx_c[:, None]

        # Interior y-faces: j = 1 .. ny-1
        v_new = v_star.copy()
        v_new[:, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / grid.dy_c[None, :]

        # ----------------------------------------------------------------
        # Step 4 — Pressure update
        # ----------------------------------------------------------------
        self.p += phi

        # ----------------------------------------------------------------
        # Step 5 — Apply only essential BCs to corrected velocity
        # (wall no-penetration + outflow; NOT inflow tangential v, which
        #  would override the pressure-corrected values)
        # ----------------------------------------------------------------
        apply_post_correction_bc(u_new, v_new, grid, bc, dt=dt)
        if self.ibm is not None and self.ibm.has_solid:
            self.ibm.apply(u_new, v_new, relax=ibm_relax)

        self.u = u_new
        self.v = v_new
        self.t += dt

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def divergence(self) -> np.ndarray:
        """Return ∇·u at all cell centres (should be ≈ 0)."""
        return divergence(self.u, self.v, self.grid)

    def cfl(self, dt: float) -> float:
        """Return the maximum CFL number over all cell centres."""
        grid = self.grid
        # Interpolate u, v to cell centres for CFL estimate
        u_c = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        v_c = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        cfl_local = (
            np.abs(u_c) / grid.dx_f[:, None] + np.abs(v_c) / grid.dy_f[None, :]) * dt
        return float(np.max(cfl_local))

    def suggest_dt(self, cfl_target: float = 0.5, dt_max: float = 0.1) -> float:
        """Suggest a stable time step based on convective CFL and diffusion."""
        grid = self.grid
        u_max = max(np.max(np.abs(self.u)), 1e-12)
        v_max = max(np.max(np.abs(self.v)), 1e-12)
        inv_dt_conv = np.max(np.abs(u_max) / grid.dx_f) + \
            np.max(np.abs(v_max) / grid.dy_f)
        dt_conv = cfl_target / max(inv_dt_conv, 1e-12)
        dt_diff = 0.5 * cfl_target / \
            (self.nu * (1.0 / grid.dx_min**2 + 1.0 / grid.dy_min**2))
        return min(dt_conv, dt_diff, dt_max)
