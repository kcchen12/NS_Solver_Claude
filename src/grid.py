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


class CartesianGrid:
    """Cartesian grid supporting uniform and non-uniform 2-D layouts."""

    def __init__(self, nx: int, ny: int, nz: int = 1,
                 lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
                 xf: np.ndarray | None = None,
                 yf: np.ndarray | None = None,
                 zf: np.ndarray | None = None):
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

        self.is_3d: bool = nz > 1

        # Cell-face (node) coordinates
        self.xf = np.linspace(
            0.0, lx, nx + 1) if xf is None else np.asarray(xf, dtype=float)
        self.yf = np.linspace(
            0.0, ly, ny + 1) if yf is None else np.asarray(yf, dtype=float)
        self.zf = np.linspace(
            0.0, lz, nz + 1) if zf is None else np.asarray(zf, dtype=float)

        if self.xf.shape != (nx + 1,):
            raise ValueError(f"xf must have shape ({nx + 1},)")
        if self.yf.shape != (ny + 1,):
            raise ValueError(f"yf must have shape ({ny + 1},)")
        if self.zf.shape != (nz + 1,):
            raise ValueError(f"zf must have shape ({nz + 1},)")
        if not np.all(np.diff(self.xf) > 0.0):
            raise ValueError("xf must be strictly increasing")
        if not np.all(np.diff(self.yf) > 0.0):
            raise ValueError("yf must be strictly increasing")
        if not np.all(np.diff(self.zf) > 0.0):
            raise ValueError("zf must be strictly increasing")

        # Cell-centre coordinates
        self.xc = 0.5 * (self.xf[:-1] + self.xf[1:])
        self.yc = 0.5 * (self.yf[:-1] + self.yf[1:])
        self.zc = 0.5 * (self.zf[:-1] + self.zf[1:])

        self.dx_cells = np.diff(self.xf)
        self.dy_cells = np.diff(self.yf)
        self.dz_cells = np.diff(self.zf)

        self.dx_min = float(np.min(self.dx_cells))
        self.dy_min = float(np.min(self.dy_cells))
        self.dz_min = float(np.min(self.dz_cells))
        self.dx_max = float(np.max(self.dx_cells))
        self.dy_max = float(np.max(self.dy_cells))
        self.dz_max = float(np.max(self.dz_cells))

        # Backward-compatible scalar spacings used in legacy code paths.
        self.dx = self.dx_min
        self.dy = self.dy_min
        self.dz = self.dz_min if nz > 1 else lz

        self.is_uniform = bool(
            np.allclose(self.dx_cells, self.dx_cells[0]) and
            np.allclose(self.dy_cells, self.dy_cells[0]) and
            np.allclose(self.dz_cells, self.dz_cells[0])
        )

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

    def to_metadata(self) -> dict:
        """Return serializable metadata describing the prepared grid."""
        return {
            "grid_type": "uniform" if self.is_uniform else "nonuniform",
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "lx": self.lx,
            "ly": self.ly,
            "lz": self.lz,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "dx_min": self.dx_min,
            "dy_min": self.dy_min,
            "dz_min": self.dz_min,
            "dx_max": self.dx_max,
            "dy_max": self.dy_max,
            "dz_max": self.dz_max,
            "is_3d": self.is_3d,
            "xf": self.xf.copy(),
            "yf": self.yf.copy(),
            "zf": self.zf.copy(),
            "xc": self.xc.copy(),
            "yc": self.yc.copy(),
            "zc": self.zc.copy(),
            "dx_cells": self.dx_cells.copy(),
            "dy_cells": self.dy_cells.copy(),
            "dz_cells": self.dz_cells.copy(),
            "is_uniform": self.is_uniform,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "CartesianGrid":
        """Rebuild a prepared grid from saved metadata."""
        xf = metadata.get("xf", None)
        yf = metadata.get("yf", None)
        zf = metadata.get("zf", None)
        return cls(
            nx=int(metadata["nx"]),
            ny=int(metadata["ny"]),
            nz=int(metadata.get("nz", 1)),
            lx=float(metadata["lx"]),
            ly=float(metadata["ly"]),
            lz=float(metadata.get("lz", 1.0)),
            xf=xf,
            yf=yf,
            zf=zf,
        )

    def __repr__(self) -> str:
        dim = "3D" if self.is_3d else "2D"
        grid_kind = "uniform" if self.is_uniform else "nonuniform"
        return (f"CartesianGrid({dim}, nx={self.nx}, ny={self.ny}, nz={self.nz}, "
                f"type={grid_kind}, dx_min={self.dx_min:.4g}, dy_min={self.dy_min:.4g}, dz={self.dz:.4g})")


def stretched_faces_tanh(n: int, length: float, beta: float) -> np.ndarray:
    """Build monotonic face coordinates in [0, length] using tanh stretching.

    beta <= 0 returns a uniform distribution.
    beta > 0 clusters cells near both boundaries and coarsens near the center.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")

    xi = np.linspace(0.0, 1.0, n + 1)
    if beta <= 0.0:
        return length * xi

    eta = 2.0 * xi - 1.0
    s = 0.5 * (1.0 + np.tanh(beta * eta) / np.tanh(beta))
    return length * s


def _cluster_to_upper_end(n: int, length: float, beta: float) -> np.ndarray:
    """Faces on [0, length] clustered near the upper end x=length."""
    xi = np.linspace(0.0, 1.0, n + 1)
    if beta <= 0.0:
        return length * xi
    s = np.tanh(beta * xi) / np.tanh(beta)
    return length * s


def _cluster_to_lower_end(n: int, length: float, beta: float) -> np.ndarray:
    """Faces on [0, length] clustered near the lower end x=0."""
    xi = np.linspace(0.0, 1.0, n + 1)
    if beta <= 0.0:
        return length * xi
    s = 1.0 - (np.tanh(beta * (1.0 - xi)) / np.tanh(beta))
    return length * s


def stretched_faces_piecewise_focus(
    n: int,
    length: float,
    beta: float,
    focus: float,
) -> np.ndarray:
    """Piecewise monotonic faces clustered around an interior focus location.

    The interval [0, length] is split at ``focus``. The left segment is
    clustered toward ``focus`` from below, and the right segment is clustered
    toward ``focus`` from above.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")

    if beta <= 0.0:
        return stretched_faces_tanh(n, length, beta=0.0)

    eps = 1e-12 * max(1.0, length)
    focus_clamped = float(np.clip(focus, eps, length - eps))

    n_left = int(round(n * (focus_clamped / length)))
    n_left = int(np.clip(n_left, 1, n - 1))
    n_right = n - n_left

    left_faces = _cluster_to_upper_end(n_left, focus_clamped, beta)
    right_local = _cluster_to_lower_end(n_right, length - focus_clamped, beta)
    right_faces = focus_clamped + right_local

    faces = np.concatenate([left_faces[:-1], right_faces])
    faces[0] = 0.0
    faces[-1] = length
    return faces


def build_nonuniform_grid_metadata(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    beta_x: float,
    beta_y: float,
    focus_x: float | None = None,
    focus_y: float | None = None,
) -> dict:
    """Build serializable metadata for a 2-D non-uniform Cartesian grid.

    Non-uniform spacing is clustered around the provided focus point.
    """
    fx = lx / 4.0 if focus_x is None else float(focus_x)
    fy = ly / 2.0 if focus_y is None else float(focus_y)
    xf = stretched_faces_piecewise_focus(nx, lx, beta_x, focus=fx)
    yf = stretched_faces_piecewise_focus(ny, ly, beta_y, focus=fy)

    grid = CartesianGrid(nx=nx, ny=ny, lx=lx, ly=ly, xf=xf, yf=yf)
    metadata = grid.to_metadata()
    metadata["grid_type"] = "nonuniform"
    metadata["beta_x"] = float(beta_x)
    metadata["beta_y"] = float(beta_y)
    metadata["nonuniform_mode"] = "piecewise-cylinder"
    metadata["focus_x"] = float(fx)
    metadata["focus_y"] = float(fy)
    metadata["dx"] = grid.dx_cells.copy()
    metadata["dy"] = grid.dy_cells.copy()
    return metadata
