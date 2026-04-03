"""
Cartesian MAC staggered grid.

Staggered variable placement (2-D shown; 3-D adds z-faces for w):

    p  : cell centres          shape (nx, ny [, nz])
    u  : x-face centres        shape (nx+1, ny [, nz])
    v  : y-face centres        shape (nx, ny+1 [, nz])
    w  : z-face centres (3-D)  shape (nx, ny, nz+1)

Index convention: first index → x, second → y, third → z.

Face/node indices run from 0 to nx (inclusive) for x-faces, etc.
Cell-centre indices run from 0 to nx-1.

Example (nx=4, ny=3, 2-D):

    y
    ^
    |  [0,2][1,2][2,2][3,2]   ← cell centres (p)
    |  [0,1][1,1][2,1][3,1]
    |  [0,0][1,0][2,0][3,0]
    +-----------------------------> x

    u-faces (4+1=5 in x, 3 in y):
       i=0  i=1  i=2  i=3  i=4

    v-faces (4 in x, 3+1=4 in y):
       j=0(bottom boundary) … j=3(top boundary)
"""

import numpy as np
from typing import Optional


def _build_faces(n: int, length: float, spacing: str,
                 focus: Optional[float], stretch: float) -> np.ndarray:
    """Build monotone face coordinates on [0, length]."""
    mode = str(spacing).strip().lower()
    if mode == "uniform":
        return np.linspace(0.0, length, n + 1)
    if mode != "stretched":
        raise ValueError(
            f"Unsupported spacing mode '{spacing}'. Use 'uniform' or 'stretched'.")

    s = np.linspace(0.0, 1.0, n + 1)
    if focus is None:
        centre = 0.5
    else:
        centre = float(focus) / float(length)
    centre = float(np.clip(centre, 1e-6, 1.0 - 1e-6))

    a = max(float(stretch), 0.0)
    if a < 1e-12:
        return np.linspace(0.0, length, n + 1)

    q = 1.0 + a
    eta = np.empty_like(s)
    left = s <= centre
    right = ~left

    r_left = np.zeros_like(s[left]) if centre <= 1e-12 else s[left] / centre
    eta[left] = centre * (1.0 - (1.0 - r_left) ** q)

    span_r = 1.0 - centre
    r_right = np.zeros_like(
        s[right]) if span_r <= 1e-12 else (s[right] - centre) / span_r
    eta[right] = centre + span_r * (r_right ** q)

    # Blend with uniform spacing to avoid extremely tiny cells that can
    # collapse the explicit time step (dt -> 0) at high stretch values.
    blend = a / (1.0 + a)
    eta = (1.0 - blend) * s + blend * eta

    eta[0] = 0.0
    eta[-1] = 1.0
    return length * eta


class CartesianGrid:
    """Uniform Cartesian grid supporting both 2-D and 3-D problems."""

    def __init__(self, nx: int, ny: int, nz: int = 1,
                 lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
                 x_spacing: str = "uniform", y_spacing: str = "uniform",
                 x_focus: Optional[float] = None, y_focus: Optional[float] = None,
                 x_stretch: float = 3.0, y_stretch: float = 3.0):
        """
        Parameters
        ----------
        nx, ny, nz : int
            Number of cells in each direction. ``nz=1`` activates 2-D mode.
        lx, ly, lz : float
            Physical domain lengths.
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.lx, self.ly, self.lz = lx, ly, lz
        self.x_spacing = str(x_spacing).strip().lower()
        self.y_spacing = str(y_spacing).strip().lower()
        self.x_focus = x_focus
        self.y_focus = y_focus
        self.x_stretch = float(x_stretch)
        self.y_stretch = float(y_stretch)

        # Representative spacings retained for backwards compatibility.
        self.dx = lx / nx
        self.dy = ly / ny
        self.dz = lz / nz if nz > 1 else lz

        self.is_3d: bool = nz > 1

        # Cell-face (node) coordinates
        self.xf = _build_faces(nx, lx, self.x_spacing,
                               self.x_focus, self.x_stretch)
        self.yf = _build_faces(ny, ly, self.y_spacing,
                               self.y_focus, self.y_stretch)
        self.zf = np.linspace(0.0, lz, nz + 1)   # z-face positions

        # Cell-centre coordinates
        self.xc = 0.5 * (self.xf[:-1] + self.xf[1:])
        self.yc = 0.5 * (self.yf[:-1] + self.yf[1:])
        self.zc = 0.5 * (self.zf[:-1] + self.zf[1:])

        # Local spacings used by non-uniform operators.
        self.dx_f = np.diff(self.xf)          # cell widths in x (nx)
        self.dy_f = np.diff(self.yf)          # cell widths in y (ny)
        # centre-to-centre x distances (nx-1)
        self.dx_c = np.diff(self.xc)
        # centre-to-centre y distances (ny-1)
        self.dy_c = np.diff(self.yc)

        self.dx_min = float(np.min(self.dx_f))
        self.dy_min = float(np.min(self.dy_f))
        self.dx_west = float(self.dx_f[0])
        self.dx_east = float(self.dx_f[-1])
        self.dy_south = float(self.dy_f[0])
        self.dy_north = float(self.dy_f[-1])
        self.is_nonuniform = (self.x_spacing != "uniform") or (
            self.y_spacing != "uniform")

    # ------------------------------------------------------------------
    # Array shape helpers
    # ------------------------------------------------------------------

    @property
    def p_shape(self) -> tuple:
        """Shape of pressure array."""
        return (self.nx, self.ny, self.nz) if self.is_3d else (self.nx, self.ny)

    @property
    def u_shape(self) -> tuple:
        """Shape of x-velocity array (staggered in x)."""
        return (self.nx + 1, self.ny, self.nz) if self.is_3d else (self.nx + 1, self.ny)

    @property
    def v_shape(self) -> tuple:
        """Shape of y-velocity array (staggered in y)."""
        return (self.nx, self.ny + 1, self.nz) if self.is_3d else (self.nx, self.ny + 1)

    @property
    def w_shape(self) -> tuple:
        """Shape of z-velocity array (staggered in z, 3-D only)."""
        if self.is_3d:
            return (self.nx, self.ny, self.nz + 1)
        return None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    def zeros_p(self) -> np.ndarray:
        return np.zeros(self.p_shape)

    def zeros_u(self) -> np.ndarray:
        return np.zeros(self.u_shape)

    def zeros_v(self) -> np.ndarray:
        return np.zeros(self.v_shape)

    def zeros_w(self) -> np.ndarray:
        if not self.is_3d:
            raise RuntimeError("w-velocity only exists in 3-D mode.")
        return np.zeros(self.w_shape)

    def __repr__(self) -> str:
        dim = "3D" if self.is_3d else "2D"
        return (f"CartesianGrid({dim}, nx={self.nx}, ny={self.ny}, nz={self.nz}, "
                f"dx={self.dx:.4g}, dy={self.dy:.4g}, dz={self.dz:.4g}, "
                f"x_spacing={self.x_spacing}, y_spacing={self.y_spacing})")
