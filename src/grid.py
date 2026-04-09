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


def _allocate_piecewise_cells(
    n: int,
    lengths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Allocate integer cell counts across piecewise segments."""
    lengths = np.asarray(lengths, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = lengths > 1e-14

    counts = np.zeros_like(lengths, dtype=int)
    num_positive = int(np.count_nonzero(positive))
    if num_positive == 0:
        raise ValueError("At least one segment must have positive length")

    if n <= num_positive:
        ranked = np.argsort(-(lengths * weights))
        counts[ranked[:n]] = 1
        return counts

    counts[positive] = 1
    remaining = n - num_positive
    effective = lengths[positive] * weights[positive]
    total = float(np.sum(effective))
    if total <= 0.0:
        counts[positive] += remaining // num_positive
        counts[np.flatnonzero(positive)[:remaining % num_positive]] += 1
        return counts

    raw = remaining * effective / total
    add = np.floor(raw).astype(int)
    counts[np.flatnonzero(positive)] += add

    shortfall = remaining - int(np.sum(add))
    if shortfall > 0:
        frac = raw - add
        order = np.argsort(-frac)
        positive_idx = np.flatnonzero(positive)
        for idx in order[:shortfall]:
            counts[positive_idx[idx]] += 1

    return counts


def stretched_faces_center_band(
    n: int,
    length: float,
    beta: float,
    center: float | None = None,
    band_fraction: float = 1.0 / 3.0,
) -> tuple[np.ndarray, float, float]:
    """Faces concentrated smoothly toward the middle of a central band.

    Outside the band, the target point density stays near the uniform-grid
    baseline. Inside the band, a raised-cosine boost increases density
    smoothly toward the band center, so spacing decreases gently as you move
    inward rather than collapsing at the band edges.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")

    if beta <= 0.0:
        return stretched_faces_tanh(n, length, beta=0.0), 0.0, length

    frac = float(np.clip(band_fraction, 1e-3, 1.0))
    band_width = frac * length
    half_width = 0.5 * band_width
    eps = 1e-12 * max(1.0, length)

    center_default = 0.5 * length if center is None else float(center)
    center_clamped = float(np.clip(center_default, half_width + eps, length - half_width - eps))
    band_start = max(0.0, center_clamped - half_width)
    band_end = min(length, center_clamped + half_width)

    dense_points = max(2001, 50 * n + 1)
    x_dense = np.linspace(0.0, length, dense_points)
    density = np.ones_like(x_dense)

    inside = (x_dense >= band_start) & (x_dense <= band_end)
    if np.any(inside):
        local = (x_dense[inside] - center_clamped) / max(half_width, eps)
        # Raised-cosine bump: density=1 at band edges and density=1+beta at center.
        density[inside] += beta * 0.5 * (1.0 + np.cos(np.pi * local))

    cumulative = np.zeros_like(x_dense)
    cumulative[1:] = np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(x_dense))
    targets = np.linspace(0.0, cumulative[-1], n + 1)
    faces = np.interp(targets, cumulative, x_dense)
    faces[0] = 0.0
    faces[-1] = length
    return faces, band_start, band_end


def build_nonuniform_grid_metadata(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    beta_x: float,
    beta_y: float,
    focus_x: float | None = None,
    focus_y: float | None = None,
    band_fraction_x: float = 1.0 / 3.0,
    band_fraction_y: float = 1.0 / 3.0,
) -> dict:
    """Build serializable metadata for a 2-D non-uniform Cartesian grid.

    Non-uniform spacing is concentrated inside a rectangular interior band.
    """
    center_x = 0.5 * lx if focus_x is None else float(focus_x)
    center_y = 0.5 * ly if focus_y is None else float(focus_y)
    xf, band_start_x, band_end_x = stretched_faces_center_band(
        nx, lx, beta_x, center=center_x, band_fraction=band_fraction_x
    )
    yf, band_start_y, band_end_y = stretched_faces_center_band(
        ny, ly, beta_y, center=center_y, band_fraction=band_fraction_y
    )

    grid = CartesianGrid(nx=nx, ny=ny, lx=lx, ly=ly, xf=xf, yf=yf)
    metadata = grid.to_metadata()
    metadata["grid_type"] = "nonuniform"
    metadata["beta_x"] = float(beta_x)
    metadata["beta_y"] = float(beta_y)
    metadata["nonuniform_mode"] = "center-band"
    metadata["focus_x"] = float(center_x)
    metadata["focus_y"] = float(center_y)
    metadata["band_fraction_x"] = float(band_fraction_x)
    metadata["band_fraction_y"] = float(band_fraction_y)
    metadata["band_start_x"] = float(band_start_x)
    metadata["band_end_x"] = float(band_end_x)
    metadata["band_start_y"] = float(band_start_y)
    metadata["band_end_y"] = float(band_end_y)
    metadata["dx"] = grid.dx_cells.copy()
    metadata["dy"] = grid.dy_cells.copy()
    return metadata
