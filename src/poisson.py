"""
Pressure Poisson solver for the fractional-step method.

Problem statement
-----------------
After computing an intermediate velocity u* (divergence-free correction
not yet applied), we need to find the pressure correction phi such that:

    laplacian(phi) = div(u*) / dt

with boundary conditions:
    - dphi/dn = 0  (Neumann) at inflow, far-field, and solid-wall faces
    - phi = 0      (Dirichlet) at outflow faces

For purely Neumann systems (no outflow) the matrix is singular, so one
reference pressure value is pinned to zero.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.boundary import BCType, BoundaryConfig
from src.grid import CartesianGrid


def _cell_index(i: int, j: int, nx: int) -> int:
    """Flat index of cell (i, j) in row-major order (j-major)."""
    return j * nx + i


def _has_outflow_dirichlet(bc: BoundaryConfig) -> bool:
    """True when any face imposes Dirichlet pressure at outflow."""
    return any(t == BCType.OUTFLOW for t in [bc.left, bc.right, bc.bottom, bc.top])


def _regularize_neumann_matrix(A: sp.csr_matrix) -> sp.csr_matrix:
    """Pin one pressure value so a purely Neumann system becomes solvable."""
    A_reg = A.tolil()
    A_reg[0, :] = 0.0
    A_reg[0, 0] = 1.0
    return A_reg.tocsr()


def build_poisson_matrix(grid: CartesianGrid, bc: BoundaryConfig):
    """
    Build the sparse Laplacian matrix A and a boolean mask ``dirich_mask``
    indicating Dirichlet cells (where phi = 0).

    Returns
    -------
    A : (nx*ny) x (nx*ny) CSR matrix
    dirich_mask : np.ndarray of bool, shape (nx*ny,)
        True where b must be overridden to 0 (Dirichlet BC).
    """
    nx, ny = grid.nx, grid.ny
    N = nx * ny

    rows, cols, vals = [], [], []
    dirich_mask = np.zeros(N, dtype=bool)

    def add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    dx_cells = grid.dx_cells
    dy_cells = grid.dy_cells
    dxc = grid.dxc
    dyc = grid.dyc

    for j in range(ny):
        for i in range(nx):
            idx = _cell_index(i, j, nx)

            dirich_left = (i == 0) and (bc.left == BCType.OUTFLOW)
            dirich_right = (i == nx - 1) and (bc.right == BCType.OUTFLOW)
            dirich_bottom = (j == 0) and (bc.bottom == BCType.OUTFLOW)
            dirich_top = (j == ny - 1) and (bc.top == BCType.OUTFLOW)

            if dirich_left or dirich_right or dirich_bottom or dirich_top:
                add(idx, idx, 1.0)
                dirich_mask[idx] = True
                continue

            diag = 0.0
            dxp = dx_cells[i]
            dyp = dy_cells[j]

            if i > 0:
                left_dirich = (i - 1 == 0) and (bc.left == BCType.OUTFLOW)
                a_w = 1.0 / (dxp * dxc[i - 1])
                if not left_dirich:
                    add(idx, _cell_index(i - 1, j, nx), a_w)
                diag -= a_w

            if i < nx - 1:
                right_dirich = (i + 1 == nx - 1) and (bc.right == BCType.OUTFLOW)
                a_e = 1.0 / (dxp * dxc[i])
                if not right_dirich:
                    add(idx, _cell_index(i + 1, j, nx), a_e)
                diag -= a_e

            if j > 0:
                bot_dirich = (j - 1 == 0) and (bc.bottom == BCType.OUTFLOW)
                a_s = 1.0 / (dyp * dyc[j - 1])
                if not bot_dirich:
                    add(idx, _cell_index(i, j - 1, nx), a_s)
                diag -= a_s

            if j < ny - 1:
                top_dirich = (j + 1 == ny - 1) and (bc.top == BCType.OUTFLOW)
                a_n = 1.0 / (dyp * dyc[j])
                if not top_dirich:
                    add(idx, _cell_index(i, j + 1, nx), a_n)
                diag -= a_n

            add(idx, idx, diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, dirich_mask


def solve_poisson(
    rhs: np.ndarray,
    grid: CartesianGrid,
    bc: BoundaryConfig,
    A: sp.csr_matrix = None,
    dirich_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Solve the pressure Poisson equation ``laplacian(phi) = rhs``.

    Returns
    -------
    phi : np.ndarray, shape (nx, ny)
    """
    nx, ny = grid.nx, grid.ny
    if rhs.shape != (nx, ny):
        raise ValueError(f"rhs must have shape {(nx, ny)}, got {rhs.shape}")

    if A is None:
        A, dirich_mask = build_poisson_matrix(grid, bc)

    b = rhs.ravel(order="F").copy()
    if dirich_mask is not None:
        b[dirich_mask] = 0.0

    if not _has_outflow_dirichlet(bc):
        A = _regularize_neumann_matrix(A)
        b[0] = 0.0

    phi_flat = spla.spsolve(A, b)
    return phi_flat.reshape((nx, ny), order="F")


class PoissonSolver:
    """
    Caching wrapper around :func:`solve_poisson`.

    Builds and factorizes the sparse matrix once so repeated solves are cheap.
    """

    def __init__(self, grid: CartesianGrid, bc: BoundaryConfig):
        self.grid = grid
        self.bc = bc
        self._A, self._dirich_mask = build_poisson_matrix(grid, bc)

        if not _has_outflow_dirichlet(bc):
            self._A = _regularize_neumann_matrix(self._A)
            self._regularised = True
        else:
            self._regularised = False

        self._solve_linear = spla.factorized(self._A.tocsc())

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``laplacian(phi) = rhs``, returning phi of shape (nx, ny)."""
        nx, ny = self.grid.nx, self.grid.ny
        if rhs.shape != (nx, ny):
            raise ValueError(f"rhs must have shape {(nx, ny)}, got {rhs.shape}")

        b = rhs.ravel(order="F").copy()
        b[self._dirich_mask] = 0.0
        if self._regularised:
            b[0] = 0.0

        phi_flat = self._solve_linear(b)
        return phi_flat.reshape((nx, ny), order="F")
