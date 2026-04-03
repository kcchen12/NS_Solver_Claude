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


def _edge_order(coords: np.ndarray) -> int:
    return 2 if coords.size >= 3 else 1


def _first_derivative(arr: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    return np.gradient(arr, coords, axis=axis, edge_order=_edge_order(coords))


def _second_derivative(arr: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    d1 = _first_derivative(arr, coords, axis=axis)
    return np.gradient(d1, coords, axis=axis, edge_order=_edge_order(coords))


def _three_point_second(fm: np.ndarray, f0: np.ndarray, fp: np.ndarray,
                        hm: np.ndarray, hp: np.ndarray) -> np.ndarray:
    return 2.0 * ((fp - f0) / hp - (f0 - fm) / hm) / (hp + hm)


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
    div = (
        (u[1:, :] - u[:-1, :]) / grid.dx_f[:, None]
        + (v[:, 1:] - v[:, :-1]) / grid.dy_f[None, :]
    )
    return div


# ---------------------------------------------------------------------------
# Gradient of pressure
# ---------------------------------------------------------------------------

def grad_p_x(p: np.ndarray, grid: CartesianGrid) -> np.ndarray:
    """
    x-component of ∇p at interior x-faces.

    Returns shape (nx-1, ny)  (faces i=1..nx-1).
    """
    return (p[1:, :] - p[:-1, :]) / grid.dx_c[:, None]


def grad_p_y(p: np.ndarray, grid: CartesianGrid) -> np.ndarray:
    """
    y-component of ∇p at interior y-faces.

    Returns shape (nx, ny-1)  (faces j=1..ny-1).
    """
    return (p[:, 1:] - p[:, :-1]) / grid.dy_c[None, :]


# ---------------------------------------------------------------------------
# Laplacian of velocity
# ---------------------------------------------------------------------------

def laplacian_u(u: np.ndarray, grid: CartesianGrid,
                bc: BoundaryConfig) -> np.ndarray:
    """
    ∇²u at all interior x-faces: i=1..nx-1, j=0..ny-1.

    Returns shape (nx-1, ny).
    """
    u_0 = u[1:-1, :]

    # x-direction at interior u-faces using local face spacing.
    hm_x = grid.dx_f[:-1][:, None]
    hp_x = grid.dx_f[1:][:, None]
    d2u_dx2 = _three_point_second(u[:-2, :], u_0, u[2:, :], hm_x, hp_x)

    # y-direction with ghost rows for tangential-wall handling.
    u_ghost_bot = _u_ghost_bottom(u_0, bc)
    u_ghost_top = _u_ghost_top(u_0, bc, grid)

    u_ym = np.empty_like(u_0)
    u_yp = np.empty_like(u_0)
    u_ym[:, 1:] = u_0[:, :-1]
    u_ym[:, 0] = u_ghost_bot
    u_yp[:, :-1] = u_0[:, 1:]
    u_yp[:, -1] = u_ghost_top

    hm_y = np.empty(grid.ny)
    hp_y = np.empty(grid.ny)
    hm_y[0] = grid.dy_f[0]
    hp_y[-1] = grid.dy_f[-1]
    if grid.ny > 1:
        hp_y[:-1] = grid.dy_c
        hm_y[1:] = grid.dy_c

    d2u_dy2 = _three_point_second(
        u_ym, u_0, u_yp, hm_y[None, :], hp_y[None, :])

    return d2u_dx2 + d2u_dy2


def laplacian_v(v: np.ndarray, grid: CartesianGrid,
                bc: BoundaryConfig) -> np.ndarray:
    """
    ∇²v at all interior y-faces: i=0..nx-1, j=1..ny-1.

    Returns shape (nx, ny-1).
    """
    v_0 = v[:, 1:-1]

    # y-direction at interior v-faces using local face spacing.
    hm_y = grid.dy_f[:-1][None, :]
    hp_y = grid.dy_f[1:][None, :]
    d2v_dy2 = _three_point_second(v[:, :-2], v_0, v[:, 2:], hm_y, hp_y)

    # x-direction with ghost columns for tangential-wall handling.
    v_ghost_lft = _v_ghost_left(v_0, bc)
    v_ghost_rgt = _v_ghost_right(v_0, bc, grid)

    v_xm = np.empty_like(v_0)
    v_xp = np.empty_like(v_0)
    v_xm[1:, :] = v_0[:-1, :]
    v_xm[0, :] = v_ghost_lft
    v_xp[:-1, :] = v_0[1:, :]
    v_xp[-1, :] = v_ghost_rgt

    hm_x = np.empty(grid.nx)
    hp_x = np.empty(grid.nx)
    hm_x[0] = grid.dx_f[0]
    hp_x[-1] = grid.dx_f[-1]
    if grid.nx > 1:
        hp_x[:-1] = grid.dx_c
        hm_x[1:] = grid.dx_c

    d2v_dx2 = _three_point_second(
        v_xm, v_0, v_xp, hm_x[:, None], hp_x[:, None])

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
    if not grid.is_nonuniform:
        dx, dy = grid.dx, grid.dy
        nx, ny = grid.nx, grid.ny

        u_int = u[1:-1, :]

        u_e = 0.5 * (u[1:-1, :] + u[2:, :])
        u_w = 0.5 * (u[:-2, :] + u[1:-1, :])
        duu_dx = (u_e**2 - u_w**2) / dx

        v_at_uf_x = 0.5 * (v[:-1, :] + v[1:, :])

        u_ghost_bot = _u_ghost_bottom(u_int, bc)
        u_ghost_top = _u_ghost_top(u_int, bc, grid)

        u_n = np.empty((nx - 1, ny))
        u_n[:, :-1] = 0.5 * (u_int[:, :-1] + u_int[:, 1:])
        u_n[:, -1] = 0.5 * (u_int[:, -1] + u_ghost_top)

        u_s = np.empty((nx - 1, ny))
        u_s[:, 1:] = 0.5 * (u_int[:, :-1] + u_int[:, 1:])
        u_s[:, 0] = 0.5 * (u_ghost_bot + u_int[:, 0])

        v_n = v_at_uf_x[:, 1:]
        v_s = v_at_uf_x[:, :-1]
        dvu_dy = (v_n * u_n - v_s * u_s) / dy

        return -(duu_dx + dvu_dy)

    u_int = u[1:-1, :]
    dudx = _first_derivative(u_int, grid.xf[1:-1], axis=0)
    dudy = _first_derivative(u_int, grid.yc, axis=1)

    # Interpolate v to u locations (xf interior, yc).
    v_at_uf_x = 0.5 * (v[:-1, :] + v[1:, :])
    v_at_u = 0.5 * (v_at_uf_x[:, :-1] + v_at_uf_x[:, 1:])

    return -(u_int * dudx + v_at_u * dudy)


def convection_v(u: np.ndarray, v: np.ndarray,
                 grid: CartesianGrid, bc: BoundaryConfig) -> np.ndarray:
    """
    Convective term  -(u·∇)v  at interior y-faces (i=0..nx-1, j=1..ny-1).

    Returns shape (nx, ny-1).
    """
    if not grid.is_nonuniform:
        dx, dy = grid.dx, grid.dy
        nx, ny = grid.nx, grid.ny

        v_int = v[:, 1:-1]

        u_at_yf = 0.5 * (u[:, :-1] + u[:, 1:])
        u_e = u_at_yf[1:, :]
        u_w = u_at_yf[:-1, :]

        v_ghost_lft = _v_ghost_left(v_int, bc)
        v_ghost_rgt = _v_ghost_right(v_int, bc, grid)

        v_e = np.empty((nx, ny - 1))
        v_e[:-1, :] = 0.5 * (v_int[:-1, :] + v_int[1:, :])
        v_e[-1, :] = 0.5 * (v_int[-1, :] + v_ghost_rgt)

        v_w = np.empty((nx, ny - 1))
        v_w[1:, :] = 0.5 * (v_int[:-1, :] + v_int[1:, :])
        v_w[0, :] = 0.5 * (v_ghost_lft + v_int[0, :])

        duv_dx = (u_e * v_e - u_w * v_w) / dx

        v_n = 0.5 * (v_int + v[:, 2:])
        v_s = 0.5 * (v[:, :-2] + v_int)
        dvv_dy = (v_n**2 - v_s**2) / dy

        return -(duv_dx + dvv_dy)

    v_int = v[:, 1:-1]
    dvdx = _first_derivative(v_int, grid.xc, axis=0)
    dvdy = _first_derivative(v_int, grid.yf[1:-1], axis=1)

    # Interpolate u to v locations (xc, yf interior).
    u_at_yf = 0.5 * (u[:, :-1] + u[:, 1:])
    u_at_v = 0.5 * (u_at_yf[:-1, :] + u_at_yf[1:, :])

    return -(u_at_v * dvdx + v_int * dvdy)


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
