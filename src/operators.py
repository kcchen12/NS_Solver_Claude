"""
Finite-volume spatial operators for the MAC staggered grid.

Variable layout (2-D)
---------------------
  p[i, j]   pressure     at cell centre (xc[i], yc[j])    shape (nx,   ny  )
  u[i, j]   x-velocity   at x-face     (xf[i], yc[j])    shape (nx+1, ny  )
  v[i, j]   y-velocity   at y-face     (xc[i], yf[j])    shape (nx,   ny+1)

All operators return arrays whose shape matches their natural support.

Ghost-cell convention for wall BCs
-----------------------------------
When a velocity component is tangential to a no-slip wall its value at
the face is zero.  Since the component is stored at the *cell centre* in
that direction (not at the face), we use a ghost-cell reflection:

    u_ghost = 2 * u_wall - u_interior

with u_wall = 0 for no-slip.  The ghost value is needed only for the
second-order Laplacian stencil.
"""

import numpy as np
from src.grid import CartesianGrid
from src.boundary import BoundaryConfig, BCType


# ---------------------------------------------------------------------------
# Ghost-cell helpers (tangential BCs for Laplacian)
# ---------------------------------------------------------------------------

def _u_ghost_bottom(u: np.ndarray, bc: BoundaryConfig) -> np.ndarray:
    """Ghost row of u below y = 0 (j = -1)."""
    if bc.bottom == BCType.WALL:
        return -u[:, 0]              # no-slip: average = 0 at face
    elif bc.bottom in (BCType.INFLOW, BCType.FARFIELD):
        return 2.0 * bc.u_inf - u[:, 0]
    else:
        return u[:, 0]               # Neumann (zero gradient)


def _u_ghost_top(u: np.ndarray, bc: BoundaryConfig, grid: CartesianGrid) -> np.ndarray:
    """Ghost row of u above y = Ly (j = ny)."""
    if bc.top == BCType.WALL:
        return -u[:, -1]
    elif bc.top in (BCType.INFLOW, BCType.FARFIELD):
        return 2.0 * bc.u_inf - u[:, -1]
    else:
        return u[:, -1]


def _v_ghost_left(v: np.ndarray, bc: BoundaryConfig) -> np.ndarray:
    """Ghost column of v left of x = 0 (i = -1)."""
    if bc.left == BCType.WALL:
        return -v[0, :]
    elif bc.left in (BCType.INFLOW, BCType.FARFIELD):
        return 2.0 * bc.v_inf - v[0, :]
    else:
        return v[0, :]


def _v_ghost_right(v: np.ndarray, bc: BoundaryConfig, grid: CartesianGrid) -> np.ndarray:
    """Ghost column of v right of x = Lx (i = nx)."""
    if bc.right == BCType.WALL:
        return -v[-1, :]
    elif bc.right in (BCType.INFLOW, BCType.FARFIELD):
        return 2.0 * bc.v_inf - v[-1, :]
    elif bc.right == BCType.OUTFLOW:
        return v[-1, :]              # Neumann
    else:
        return v[-1, :]


# ---------------------------------------------------------------------------
# Divergence  ∇·u
# ---------------------------------------------------------------------------

def divergence(u: np.ndarray, v: np.ndarray,
               grid: CartesianGrid) -> np.ndarray:
    """
    Compute ∇·u at all cell centres.

    Returns array of shape (nx, ny).
    """
    dx, dy = grid.dx, grid.dy
    # u[i+1, j] - u[i, j] for i=0..nx-1
    div = (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy
    return div


# ---------------------------------------------------------------------------
# Gradient of pressure
# ---------------------------------------------------------------------------

def grad_p_x(p: np.ndarray, grid: CartesianGrid) -> np.ndarray:
    """
    x-component of ∇p at interior x-faces.

    Returns shape (nx-1, ny)  (faces i=1..nx-1).
    """
    return (p[1:, :] - p[:-1, :]) / grid.dx


def grad_p_y(p: np.ndarray, grid: CartesianGrid) -> np.ndarray:
    """
    y-component of ∇p at interior y-faces.

    Returns shape (nx, ny-1)  (faces j=1..ny-1).
    """
    return (p[:, 1:] - p[:, :-1]) / grid.dy


# ---------------------------------------------------------------------------
# Laplacian of velocity
# ---------------------------------------------------------------------------

def laplacian_u(u: np.ndarray, grid: CartesianGrid,
                bc: BoundaryConfig) -> np.ndarray:
    """
    ∇²u at all interior x-faces: i=1..nx-1, j=0..ny-1.

    Returns shape (nx-1, ny).
    """
    dx, dy = grid.dx, grid.dy

    # x-direction: d²u/dx²
    # Interior in x: i=1..nx-1 → use u[i-1], u[i], u[i+1]
    u_xm = u[:-2, :]   # u[i-1, j]
    u_xp = u[2:,  :]   # u[i+1, j]
    u_0  = u[1:-1, :]  # u[i,   j]
    d2u_dx2 = (u_xp - 2.0 * u_0 + u_xm) / dx**2

    # y-direction: d²u/dy²
    # Need ghost rows at j=-1 and j=ny
    u_ghost_bot = _u_ghost_bottom(u[1:-1, :], bc)   # shape (nx-1,)
    u_ghost_top = _u_ghost_top(u[1:-1, :], bc, grid)

    u_ym = np.empty_like(u_0)
    u_yp = np.empty_like(u_0)
    u_ym[:, 1:]  = u_0[:, :-1]    # u[i, j-1] for j>=1
    u_ym[:, 0]   = u_ghost_bot    # ghost at j=-1
    u_yp[:, :-1] = u_0[:, 1:]     # u[i, j+1] for j<=ny-2
    u_yp[:, -1]  = u_ghost_top    # ghost at j=ny

    d2u_dy2 = (u_yp - 2.0 * u_0 + u_ym) / dy**2

    return d2u_dx2 + d2u_dy2


def laplacian_v(v: np.ndarray, grid: CartesianGrid,
                bc: BoundaryConfig) -> np.ndarray:
    """
    ∇²v at all interior y-faces: i=0..nx-1, j=1..ny-1.

    Returns shape (nx, ny-1).
    """
    dx, dy = grid.dx, grid.dy

    # y-direction
    v_ym = v[:, :-2]   # v[i, j-1]
    v_yp = v[:, 2:]    # v[i, j+1]
    v_0  = v[:, 1:-1]  # v[i, j]
    d2v_dy2 = (v_yp - 2.0 * v_0 + v_ym) / dy**2

    # x-direction
    v_ghost_lft = _v_ghost_left(v[:, 1:-1], bc)
    v_ghost_rgt = _v_ghost_right(v[:, 1:-1], bc, grid)

    v_xm = np.empty_like(v_0)
    v_xp = np.empty_like(v_0)
    v_xm[1:,  :] = v_0[:-1, :]
    v_xm[0,   :] = v_ghost_lft
    v_xp[:-1, :] = v_0[1:, :]
    v_xp[-1,  :] = v_ghost_rgt

    d2v_dx2 = (v_xp - 2.0 * v_0 + v_xm) / dx**2

    return d2v_dx2 + d2v_dy2


# ---------------------------------------------------------------------------
# Convective fluxes (2nd-order central)
# ---------------------------------------------------------------------------

def convection_u(u: np.ndarray, v: np.ndarray,
                 grid: CartesianGrid, bc: BoundaryConfig) -> np.ndarray:
    """
    Convective term  -(u·∇)u  at interior x-faces (i=1..nx-1, j=0..ny-1).

    Returns shape (nx-1, ny).

    Uses 2nd-order central differencing.
    """
    dx, dy = grid.dx, grid.dy
    nx, ny = grid.nx, grid.ny

    u_int = u[1:-1, :]    # interior u faces, shape (nx-1, ny)

    # --- d(uu)/dx ---
    # u at right face of control volume (at xc[i])  i=1..nx-1:
    #   u_e = (u[i, j] + u[i+1, j]) / 2
    # u at left face (at xc[i-1]):
    #   u_w = (u[i-1, j] + u[i, j]) / 2
    u_e = 0.5 * (u[1:-1, :] + u[2:, :])     # shape (nx-1, ny)
    u_w = 0.5 * (u[:-2,  :] + u[1:-1, :])   # shape (nx-1, ny)
    duu_dx = (u_e**2 - u_w**2) / dx

    # --- d(vu)/dy ---
    # v at top face of u-control-volume  (y = yf[j+1]):
    #   v_n = (v[i-1, j+1] + v[i, j+1]) / 2   (v is at xc, so interpolate)
    # u at top face:
    #   u_n = (u[i, j] + u[i, j+1]) / 2

    # Build v at x=xf[i] (i=1..nx-1) by averaging v[i-1, :] and v[i, :]
    # v has shape (nx, ny+1); indices 1..nx-1 → v[0..nx-2] and v[1..nx-1]
    v_at_uf_x = 0.5 * (v[:-1, :] + v[1:, :])   # shape (nx-1, ny+1)

    # Ghost rows for u in y
    u_ghost_bot = _u_ghost_bottom(u_int, bc)   # shape (nx-1,)
    u_ghost_top = _u_ghost_top(u_int, bc, grid)

    # u_n = (u[i, j] + u[i, j+1]) / 2
    u_n = np.empty((nx - 1, ny))
    u_n[:, :-1] = 0.5 * (u_int[:, :-1] + u_int[:, 1:])
    u_n[:, -1]  = 0.5 * (u_int[:, -1] + u_ghost_top)

    # u_s = (u[i, j-1] + u[i, j]) / 2
    u_s = np.empty((nx - 1, ny))
    u_s[:, 1:] = 0.5 * (u_int[:, :-1] + u_int[:, 1:])
    u_s[:, 0]  = 0.5 * (u_ghost_bot + u_int[:, 0])

    # v_n  (at y-face j+1)
    v_n = v_at_uf_x[:, 1:]    # shape (nx-1, ny)
    # v_s  (at y-face j)
    v_s = v_at_uf_x[:, :-1]   # shape (nx-1, ny)

    dvu_dy = (v_n * u_n - v_s * u_s) / dy

    return -(duu_dx + dvu_dy)


def convection_v(u: np.ndarray, v: np.ndarray,
                 grid: CartesianGrid, bc: BoundaryConfig) -> np.ndarray:
    """
    Convective term  -(u·∇)v  at interior y-faces (i=0..nx-1, j=1..ny-1).

    Returns shape (nx, ny-1).
    """
    dx, dy = grid.dx, grid.dy
    nx, ny = grid.nx, grid.ny

    v_int = v[:, 1:-1]   # interior v faces, shape (nx, ny-1)

    # --- d(uv)/dx ---
    # u at y = yf[j]  (j=1..ny-1) — interpolate u at x-faces to xc:
    #   u at right face of v-control-vol (at xf[i+1]):  u[i+1, j-1+1] & u[i+1, j]
    # v at right face:
    #   v_e = (v[i, j] + v[i+1, j]) / 2

    # u at y-face yf[j]: u is at yc, so need to interpolate to yf[j]
    # u_at_yf[i, j] = (u[i, j-1] + u[i, j]) / 2  for j=1..ny-1
    #   u shape (nx+1, ny); u_at_yf shape (nx+1, ny-1)
    u_at_yf = 0.5 * (u[:, :-1] + u[:, 1:])   # shape (nx+1, ny-1)

    # u at right face of v-cell (xf[i+1], yf[j]):
    u_e = u_at_yf[1:, :]   # shape (nx, ny-1)   [i+1, j] i=0..nx-1
    # u at left face of v-cell (xf[i], yf[j]):
    u_w = u_at_yf[:-1, :]  # shape (nx, ny-1)   [i,   j]

    # v_e = (v[i, j] + v[i+1, j]) / 2
    v_ghost_lft = _v_ghost_left(v_int, bc)     # shape (ny-1,)
    v_ghost_rgt = _v_ghost_right(v_int, bc, grid)

    v_e = np.empty((nx, ny - 1))
    v_e[:-1, :] = 0.5 * (v_int[:-1, :] + v_int[1:, :])
    v_e[-1,  :] = 0.5 * (v_int[-1, :] + v_ghost_rgt)

    v_w = np.empty((nx, ny - 1))
    v_w[1:,  :] = 0.5 * (v_int[:-1, :] + v_int[1:, :])
    v_w[0,   :] = 0.5 * (v_ghost_lft + v_int[0, :])

    duv_dx = (u_e * v_e - u_w * v_w) / dx

    # --- d(vv)/dy ---
    v_n = 0.5 * (v_int + v[:, 2:])    # (v[j] + v[j+1]) / 2
    v_s = 0.5 * (v[:, :-2] + v_int)   # (v[j-1] + v[j]) / 2

    dvv_dy = (v_n**2 - v_s**2) / dy

    return -(duv_dx + dvv_dy)


# ---------------------------------------------------------------------------
# Full momentum RHS  (convection + diffusion)
# ---------------------------------------------------------------------------

def rhs_u(u: np.ndarray, v: np.ndarray,
          grid: CartesianGrid, bc: BoundaryConfig,
          nu: float) -> np.ndarray:
    """
    Full RHS for u-momentum at interior x-faces (i=1..nx-1, j=0..ny-1).

    Returns shape (nx-1, ny):  -(u·∇)u + ν ∇²u
    (pressure term added separately in the fractional-step method).
    """
    return convection_u(u, v, grid, bc) + nu * laplacian_u(u, grid, bc)


def rhs_v(u: np.ndarray, v: np.ndarray,
          grid: CartesianGrid, bc: BoundaryConfig,
          nu: float) -> np.ndarray:
    """
    Full RHS for v-momentum at interior y-faces (i=0..nx-1, j=1..ny-1).

    Returns shape (nx, ny-1):  -(u·∇)v + ν ∇²v
    """
    return convection_v(u, v, grid, bc) + nu * laplacian_v(v, grid, bc)
