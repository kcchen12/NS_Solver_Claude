"""
Boundary condition types and enforcement for MAC staggered grids.

Supported conditions
--------------------
INFLOW          : prescribed velocity (u=U_inf, v=V_inf)
FARFIELD        : free-stream velocity (same as inflow but typically
                  used on lateral/top boundaries)
OUTFLOW         : first-order convective outflow  ∂φ/∂t + c ∂φ/∂x = 0
WALL            : no-slip, no-penetration solid wall
PERIODIC        : periodic (handled externally via ghost-cell exchange)

The domain has four faces in 2-D (six in 3-D):
    LEFT   (x = 0),    RIGHT (x = Lx)
    BOTTOM (y = 0),    TOP   (y = Ly)
    FRONT  (z = 0),    BACK  (z = Lz)   ← 3-D only

Ghost-cell technique
--------------------
Where the velocity component is *not* co-located with a domain face we
use a ghost-cell reflection so that the wall value is interpolated to
exactly 0 (no-slip) or U_inf (inflow):

    u_ghost = 2 * u_wall - u_interior

This keeps 2nd-order accuracy for diffusion and does not require any
special treatment in the operator code.
"""

import numpy as np
from src.grid import CartesianGrid


# ---------------------------------------------------------------------------
# BC type constants
# ---------------------------------------------------------------------------

class BCType:
    INFLOW = "inflow"
    FARFIELD = "farfield"
    OUTFLOW = "outflow"   # convective outflow
    WALL = "wall"      # no-slip / no-penetration
    PERIODIC = "periodic"


class WallSlipMode:
    NO_SLIP = "no-slip"
    FREE_SLIP = "free-slip"


class OutflowMode:
    CONVECTIVE = "convective"
    ZERO_GRADIENT = "zero-gradient"


# ---------------------------------------------------------------------------
# BC configuration container
# ---------------------------------------------------------------------------

class BoundaryConfig:
    """Stores boundary-condition type for each domain face.

    Parameters
    ----------
    left, right, bottom, top : str
        BCType constant for each 2-D face.
    front, back : str, optional
        BCType constant for z-faces (3-D only).
    u_inf, v_inf, w_inf : float
        Free-stream / inflow velocity components.
    """

    def __init__(self, left: str, right: str, bottom: str, top: str,
                 front: str = BCType.WALL, back: str = BCType.WALL,
                 u_inf: float = 1.0, v_inf: float = 0.0, w_inf: float = 0.0,
                 wall_slip_mode: str = WallSlipMode.NO_SLIP,
                 wall_penetration: bool = False,
                 wall_normal_velocity: float = 0.0,
                 outflow_mode: str = OutflowMode.CONVECTIVE,
                 outflow_speed: float = None):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.front = front    # z = 0  (3-D)
        self.back = back     # z = Lz (3-D)
        self.u_inf = u_inf
        self.v_inf = v_inf
        self.w_inf = w_inf
        self.wall_slip_mode = wall_slip_mode
        self.wall_penetration = wall_penetration
        self.wall_normal_velocity = wall_normal_velocity
        self.outflow_mode = outflow_mode
        self.outflow_speed = outflow_speed


def _is_free_slip(bc: BoundaryConfig) -> bool:
    return str(bc.wall_slip_mode).lower() == WallSlipMode.FREE_SLIP


def _wall_normal_velocity(bc: BoundaryConfig) -> float:
    if bc.wall_penetration:
        return float(bc.wall_normal_velocity)
    return 0.0


def _outflow_convective_speed(bc: BoundaryConfig, normal_reference: float) -> float:
    if bc.outflow_speed is not None:
        return max(abs(float(bc.outflow_speed)), 1e-10)
    return max(abs(float(normal_reference)), 1e-10)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_face_u(u: np.ndarray, j_slice, value: float):
    """Set u ghost cells in y-direction so wall average equals *value*."""
    # u[:, j_slice] is the first/last interior row; ghost = 2*value - interior
    pass  # handled inline below


def _ghost_no_slip(interior_row: np.ndarray, wall_value: float) -> np.ndarray:
    """Return ghost-cell values that enforce wall_value at the boundary face."""
    return 2.0 * wall_value - interior_row


# ---------------------------------------------------------------------------
# Main BC enforcement functions
# ---------------------------------------------------------------------------

def apply_velocity_bc(u: np.ndarray, v: np.ndarray,
                      grid: CartesianGrid, bc: BoundaryConfig,
                      dt: float = None,
                      w: np.ndarray = None) -> None:
    """
    Apply boundary conditions to velocity fields **in-place**.

    u : shape (nx+1, ny)    — x-velocity at x-faces
    v : shape (nx, ny+1)    — y-velocity at y-faces
    w : shape (nx, ny, nz+1) or None — z-velocity (3-D)

    Notes
    -----
    Boundary face values of u and v are set directly.  Ghost-cell rows
    used for the diffusion stencil near solid walls are **not** stored in
    the main arrays; instead, ``operators.py`` reads them via the helper
    function :func:`ghost_u_bottom` etc.  The *only* ghost values stored
    here are the far-field / inflow / outflow conditions applied to the
    outermost layer of the velocity arrays.

    For a no-slip wall at y=0 the enforcement is:
        v[:, 0] = 0   (no penetration through the face)
    The tangential component u[:,0..] is at yc[0] = dy/2 and is driven
    to zero through the ghost-cell mechanism in the Laplacian stencil
    (see operators.py).  No modification of u interior rows is done here.
    """
    nx, ny = grid.nx, grid.ny

    # ---------------------------------------------------------------
    # LEFT boundary  (u[0, :] is the left domain face)
    # ---------------------------------------------------------------
    free_slip = _is_free_slip(bc)
    wall_vn = _wall_normal_velocity(bc)

    if bc.left in (BCType.INFLOW, BCType.FARFIELD):
        u[0, :] = bc.u_inf
        # v at left domain edge – set via linear extrapolation / Dirichlet
        v[0, :] = bc.v_inf        # v-face at i=0 matches free stream

    elif bc.left == BCType.WALL:
        u[0, :] = wall_vn
        if free_slip:
            v[0, :] = v[1, :]
        else:
            v[0, :] = _ghost_no_slip(v[1, :], 0.0)   # ghost in x for v

    elif bc.left == BCType.PERIODIC:
        pass  # handled externally

    # ---------------------------------------------------------------
    # RIGHT boundary  (u[nx, :] is the right domain face)
    # ---------------------------------------------------------------
    if bc.right == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Uc = _outflow_convective_speed(bc, bc.u_inf)
            cfl = Uc * dt / grid.dx_east
            u[nx, :] = u[nx, :] - cfl * (u[nx, :] - u[nx - 1, :])
        else:
            u[nx, :] = u[nx - 1, :]    # zero-gradient fallback
        # v at right boundary – zero-gradient in x
        v[nx - 1, :] = v[nx - 2, :]

    elif bc.right in (BCType.INFLOW, BCType.FARFIELD):
        u[nx, :] = bc.u_inf
        v[nx - 1, :] = bc.v_inf

    elif bc.right == BCType.WALL:
        u[nx, :] = wall_vn
        if free_slip:
            v[nx - 1, :] = v[nx - 2, :]
        else:
            v[nx - 1, :] = _ghost_no_slip(v[nx - 2, :], 0.0)

    elif bc.right == BCType.PERIODIC:
        pass

    # ---------------------------------------------------------------
    # BOTTOM boundary  (v[:, 0] is the bottom domain face)
    # ---------------------------------------------------------------
    if bc.bottom == BCType.WALL:
        v[:, 0] = wall_vn
        if free_slip:
            u[1:-1, 0] = u[1:-1, 1]
        # no-slip tangential component enforcement handled via ghost cells in operators

    elif bc.bottom in (BCType.INFLOW, BCType.FARFIELD):
        v[:, 0] = bc.v_inf
        u[1:-1, 0] = bc.u_inf   # interior x-faces at j=0

    elif bc.bottom == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Vc = _outflow_convective_speed(bc, bc.v_inf)
            cfl = abs(Vc) * dt / grid.dy_south
            v[:, 0] = v[:, 0] - cfl * (v[:, 0] - v[:, 1])
        else:
            v[:, 0] = v[:, 1]
        u[1:-1, 0] = u[1:-1, 1]

    elif bc.bottom == BCType.PERIODIC:
        pass

    # ---------------------------------------------------------------
    # TOP boundary  (v[:, ny] is the top domain face)
    # ---------------------------------------------------------------
    if bc.top == BCType.WALL:
        v[:, ny] = wall_vn
        if free_slip:
            u[1:-1, ny - 1] = u[1:-1, ny - 2]

    elif bc.top in (BCType.INFLOW, BCType.FARFIELD):
        v[:, ny] = bc.v_inf
        u[1:-1, ny - 1] = bc.u_inf

    elif bc.top == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Vc = _outflow_convective_speed(bc, bc.v_inf)
            cfl = Vc * dt / grid.dy_north
            v[:, ny] = v[:, ny] - cfl * (v[:, ny] - v[:, ny - 1])
        else:
            v[:, ny] = v[:, ny - 1]
        u[1:-1, ny - 1] = u[1:-1, ny - 2]

    elif bc.top == BCType.PERIODIC:
        pass

    # ---------------------------------------------------------------
    # 3-D: FRONT / BACK  (w[:, :, 0] and w[:, :, nz])
    # ---------------------------------------------------------------
    if grid.is_3d and w is not None:
        nz = grid.nz
        if bc.front == BCType.WALL:
            w[:, :, 0] = 0.0
        elif bc.front in (BCType.INFLOW, BCType.FARFIELD):
            w[:, :, 0] = bc.w_inf

        if bc.back == BCType.WALL:
            w[:, :, nz] = 0.0
        elif bc.back in (BCType.INFLOW, BCType.FARFIELD):
            w[:, :, nz] = bc.w_inf


def apply_post_correction_bc(u: np.ndarray, v: np.ndarray,
                             grid: CartesianGrid, bc: BoundaryConfig,
                             dt: float = None,
                             w: np.ndarray = None) -> None:
    """
    Apply only the *essential* boundary conditions after the pressure-
    correction step.

    After correcting the velocity with the pressure gradient, only
    no-penetration wall conditions and the outflow convective update need
    to be re-enforced.  Re-applying the inflow/far-field conditions to
    *tangential* velocity components would override the pressure-corrected
    values and destroy the divergence-free condition.

    Specifically:
    - No-penetration at solid walls  (v[:, 0]=0, v[:, ny]=0, etc.)
    - Inflow normal velocity         (u[0,:] = U_inf)
    - Convective outflow update      (u[nx,:] = convective update)
    - Zero-gradient for tangential v at outflow  (v[nx-1,:] = v[nx-2,:])
    """
    nx, ny = grid.nx, grid.ny

    # LEFT: enforce normal inflow velocity only; don't touch v[0, :]
    free_slip = _is_free_slip(bc)
    wall_vn = _wall_normal_velocity(bc)

    if bc.left in (BCType.INFLOW, BCType.FARFIELD):
        u[0, :] = bc.u_inf

    elif bc.left == BCType.WALL:
        u[0, :] = wall_vn
        if free_slip:
            v[0, :] = v[1, :]

    # RIGHT: convective outflow or prescribed
    if bc.right == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Uc = _outflow_convective_speed(bc, bc.u_inf)
            cfl = Uc * dt / grid.dx_east
            u[nx, :] = u[nx, :] - cfl * (u[nx, :] - u[nx - 1, :])
        else:
            u[nx, :] = u[nx - 1, :]
        v[nx - 1, :] = v[nx - 2, :]
    elif bc.right in (BCType.INFLOW, BCType.FARFIELD):
        u[nx, :] = bc.u_inf
    elif bc.right == BCType.WALL:
        u[nx, :] = wall_vn
        if free_slip:
            v[nx - 1, :] = v[nx - 2, :]

    # BOTTOM: no-penetration wall (v[:, 0] = 0)
    if bc.bottom == BCType.WALL:
        v[:, 0] = wall_vn
        if free_slip:
            u[1:-1, 0] = u[1:-1, 1]
    elif bc.bottom == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Vc = _outflow_convective_speed(bc, bc.v_inf)
            cfl = abs(Vc) * dt / grid.dy_south
            v[:, 0] = v[:, 0] - cfl * (v[:, 0] - v[:, 1])
        else:
            v[:, 0] = v[:, 1]

    # TOP: no-penetration wall (v[:, ny] = 0)
    if bc.top == BCType.WALL:
        v[:, ny] = wall_vn
        if free_slip:
            u[1:-1, ny - 1] = u[1:-1, ny - 2]
    elif bc.top == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Vc = _outflow_convective_speed(bc, bc.v_inf)
            cfl = Vc * dt / grid.dy_north
            v[:, ny] = v[:, ny] - cfl * (v[:, ny] - v[:, ny - 1])
        else:
            v[:, ny] = v[:, ny - 1]

    # 3D: front/back walls
    if grid.is_3d and w is not None:
        nz = grid.nz
        if bc.front == BCType.WALL:
            w[:, :, 0] = 0.0
        if bc.back == BCType.WALL:
            w[:, :, nz] = 0.0


def apply_pressure_bc(phi: np.ndarray,
                      grid: CartesianGrid, bc: BoundaryConfig) -> None:
    """
    Apply Neumann (zero-gradient) pressure boundary conditions **in-place**.

    The pressure correction φ satisfies ∂φ/∂n = 0 on all solid/inflow/
    far-field faces.  At outflow (Dirichlet φ = 0) the pressure is set
    to zero.

    phi : shape (nx, ny) or (nx, ny, nz)
    """
    nx, ny = grid.nx, grid.ny

    # LEFT
    if bc.left == BCType.OUTFLOW:
        phi[0, ...] = 0.0
    else:
        phi[0, ...] = phi[1, ...]     # Neumann

    # RIGHT
    if bc.right == BCType.OUTFLOW:
        phi[nx - 1, ...] = 0.0
    else:
        phi[nx - 1, ...] = phi[nx - 2, ...]  # Neumann

    # BOTTOM
    if bc.bottom == BCType.OUTFLOW:
        phi[:, 0, ...] = 0.0
    else:
        phi[:, 0, ...] = phi[:, 1, ...]   # Neumann

    # TOP
    if bc.top == BCType.OUTFLOW:
        phi[:, ny - 1, ...] = 0.0
    else:
        phi[:, ny - 1, ...] = phi[:, ny - 2, ...]  # Neumann

    if grid.is_3d:
        nz = grid.nz
        phi[..., 0] = phi[..., 1]
        phi[..., nz - 1] = phi[..., nz - 2]
