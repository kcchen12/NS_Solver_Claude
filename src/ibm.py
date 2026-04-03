"""
Immersed boundary method (IBM) — direct forcing approach.

Theory
------
The immersed boundary method represents solid geometry *inside* the
computational domain by adding a body-force term f to the momentum
equations:

    ∂u/∂t + (u·∇)u = -∇p + ν∇²u + f

where f is chosen to enforce the desired velocity (typically zero) at
solid-body points.

Direct-forcing approach
~~~~~~~~~~~~~~~~~~~~~~~
At each time step, after computing the intermediate velocity u*, we
correct it so that solid-body points match the prescribed velocity:

    u*[solid] = u_body[solid]

This is equivalent to an infinite forcing that drives u to u_body
instantaneously.  While first-order in time at the boundary, it is
simple to implement, robust, and widely used in the literature
(Mohd-Yusof 1997, Fadlun et al. 2000).

Geometry
--------
Solid regions are specified as level-set / mask arrays:

    mask_u[i,j] = 1  if x-face (xf[i], yc[j]) is inside solid
    mask_v[i,j] = 1  if y-face (xc[i], yf[j]) is inside solid

Helper methods allow adding:
    - Circular cylinders
    - Rectangular blocks
    - Arbitrary masks (load from array)
"""

import numpy as np
from src.grid import CartesianGrid


class ImmersedBoundary:
    """
    Manages solid-body masks for the IBM direct-forcing approach.

    Parameters
    ----------
    grid : CartesianGrid
    """

    def __init__(self, grid: CartesianGrid):
        self.grid = grid
        # Binary masks: 1 = solid, 0 = fluid
        self.mask_u = np.zeros(grid.u_shape, dtype=bool)
        self.mask_v = np.zeros(grid.v_shape, dtype=bool)

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def add_circle(self, cx: float, cy: float, radius: float,
                   u_body: float = 0.0, v_body: float = 0.0) -> None:
        """
        Mark all MAC faces inside a circular cylinder as solid.

        Parameters
        ----------
        cx, cy : float
            Centre of the cylinder in physical coordinates.
        radius : float
            Cylinder radius.
        u_body, v_body : float
            Prescribed velocity on the body surface (0 for stationary wall).
        """
        grid = self.grid
        # x-faces  (xf[i], yc[j])
        for i in range(grid.nx + 1):
            for j in range(grid.ny):
                r2 = (grid.xf[i] - cx)**2 + (grid.yc[j] - cy)**2
                if r2 <= radius**2:
                    self.mask_u[i, j] = True

        # y-faces  (xc[i], yf[j])
        for i in range(grid.nx):
            for j in range(grid.ny + 1):
                r2 = (grid.xc[i] - cx)**2 + (grid.yf[j] - cy)**2
                if r2 <= radius**2:
                    self.mask_v[i, j] = True

    def add_rectangle(self, x0: float, x1: float,
                      y0: float, y1: float,
                      u_body: float = 0.0, v_body: float = 0.0) -> None:
        """
        Mark all MAC faces inside axis-aligned rectangle [x0,x1]×[y0,y1].
        """
        grid = self.grid
        for i in range(grid.nx + 1):
            for j in range(grid.ny):
                if x0 <= grid.xf[i] <= x1 and y0 <= grid.yc[j] <= y1:
                    self.mask_u[i, j] = True
        for i in range(grid.nx):
            for j in range(grid.ny + 1):
                if x0 <= grid.xc[i] <= x1 and y0 <= grid.yf[j] <= y1:
                    self.mask_v[i, j] = True

    def add_mask(self, mask_u: np.ndarray, mask_v: np.ndarray) -> None:
        """
        Directly supply boolean masks for u and v faces.
        """
        assert mask_u.shape == self.grid.u_shape
        assert mask_v.shape == self.grid.v_shape
        self.mask_u |= mask_u
        self.mask_v |= mask_v

    # ------------------------------------------------------------------
    # Forcing
    # ------------------------------------------------------------------

    def apply(self, u: np.ndarray, v: np.ndarray,
              u_body: float = 0.0, v_body: float = 0.0,
              dt: float = None, rho: float = 1.0,
              relax: float = 1.0):
        """
        Apply direct forcing **in-place**: set solid-face velocities to the
        prescribed body velocity (default: 0 for stationary body).

        Call this *after* computing the intermediate velocity u* and
        *before* the pressure-correction step.
        """
        force_x = 0.0
        force_y = 0.0
        # SOFT_FIX_NONUNIFORM_IBM (search token)
        # Relaxed direct forcing for stretched-grid stability.
        # 1.0 = classic hard forcing, <1.0 = blended forcing.
        relax = float(np.clip(relax, 0.0, 1.0))
        if dt is not None and dt > 0.0:
            face_area = self.grid.dx * self.grid.dy
            force_x = rho * face_area * \
                float(np.sum(u[self.mask_u] - u_body)) / dt
            force_y = rho * face_area * \
                float(np.sum(v[self.mask_v] - v_body)) / dt

        if relax >= 1.0 - 1e-12:
            u[self.mask_u] = u_body
            v[self.mask_v] = v_body
        else:
            u[self.mask_u] = (1.0 - relax) * u[self.mask_u] + relax * u_body
            v[self.mask_v] = (1.0 - relax) * v[self.mask_v] + relax * v_body
        return force_x, force_y

    @property
    def has_solid(self) -> bool:
        """True if any solid cells are defined."""
        return bool(self.mask_u.any() or self.mask_v.any())
