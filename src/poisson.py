"""
Pressure Poisson solver for the fractional-step method.

Problem statement
-----------------
After computing an intermediate velocity u* (divergence-free correction
not yet applied), we need to find the pressure correction φ such that:

    ∇²φ = ∇·u* / Δt

with boundary conditions:
    - ∂φ/∂n = 0  (Neumann) at inflow, far-field, and solid-wall faces
    - φ = 0       (Dirichlet) at outflow faces

Implementation
--------------
We build a sparse matrix representation of the 2-D Laplacian on the
cell-centred grid using the standard 5-point finite-difference stencil:

    (φ[i+1,j] - 2φ[i,j] + φ[i-1,j]) / dx²
  + (φ[i,j+1] - 2φ[i,j] + φ[i,j-1]) / dy²
  = rhs[i,j]

Neumann BCs are enforced via ghost-cell approach (ghost = interior →
coefficient of the off-domain neighbour is added to the diagonal term).
Dirichlet BCs (outflow) pin φ = 0 on that face.

For purely Neumann systems (no outflow) the matrix is singular.
We regularise by pinning φ[0,0] = 0 (setting one reference pressure).

The linear system is solved with ``scipy.sparse.linalg.spsolve``.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from src.grid import CartesianGrid
from src.boundary import BoundaryConfig, BCType


def _cell_index(i: int, j: int, nx: int) -> int:
    """Flat index of cell (i, j) in row-major order (j-major)."""
    return j * nx + i


def build_poisson_matrix(grid: CartesianGrid,
                         bc: BoundaryConfig):
    """
    Build the sparse Laplacian matrix A and a boolean mask ``dirich_mask``
    indicating Dirichlet cells (where φ = 0).

    Returns
    -------
    A : (nx*ny) × (nx*ny) CSR matrix
    dirich_mask : np.ndarray of bool, shape (nx*ny,)
        True where b must be overridden to 0 (Dirichlet BC).
    """
    nx, ny = grid.nx, grid.ny
    N = nx * ny

    rows, cols, vals = [], [], []
    dirich_mask = np.zeros(N, dtype=bool)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for j in range(ny):
        for i in range(nx):
            idx = _cell_index(i, j, nx)

            # ---- determine which faces are Dirichlet (outflow) ----
            dirich_left = (i == 0) and (bc.left == BCType.OUTFLOW)
            dirich_right = (i == nx - 1) and (bc.right == BCType.OUTFLOW)
            dirich_bottom = (j == 0) and (bc.bottom == BCType.OUTFLOW)
            dirich_top = (j == ny - 1) and (bc.top == BCType.OUTFLOW)

            if dirich_left or dirich_right or dirich_bottom or dirich_top:
                # Dirichlet cell: φ = 0  →  1·φ[idx] = 0
                add(idx, idx, 1.0)
                dirich_mask[idx] = True
                continue

            diag = 0.0
            dx_cell = grid.dx_f[i]
            dy_cell = grid.dy_f[j]

            # ---- x-direction neighbours ----
            if i > 0:
                left_dirich = (i - 1 == 0) and (bc.left == BCType.OUTFLOW)
                a_w = 1.0 / (grid.dx_c[i - 1] * dx_cell)
                if not left_dirich:
                    add(idx, _cell_index(i - 1, j, nx), a_w)
                diag -= a_w
            else:
                # Left boundary face: Neumann → ghost = interior
                pass  # results in diag = -ax (one-sided stencil)

            if i < nx - 1:
                right_dirich = (
                    i + 1 == nx - 1) and (bc.right == BCType.OUTFLOW)
                a_e = 1.0 / (grid.dx_c[i] * dx_cell)
                if not right_dirich:
                    add(idx, _cell_index(i + 1, j, nx), a_e)
                diag -= a_e
            else:
                # Right boundary: Neumann
                pass

            # ---- y-direction neighbours ----
            if j > 0:
                bot_dirich = (j - 1 == 0) and (bc.bottom == BCType.OUTFLOW)
                a_s = 1.0 / (grid.dy_c[j - 1] * dy_cell)
                if not bot_dirich:
                    add(idx, _cell_index(i, j - 1, nx), a_s)
                diag -= a_s
            else:
                pass  # Neumann at bottom

            if j < ny - 1:
                top_dirich = (j + 1 == ny - 1) and (bc.top == BCType.OUTFLOW)
                a_n = 1.0 / (grid.dy_c[j] * dy_cell)
                if not top_dirich:
                    add(idx, _cell_index(i, j + 1, nx), a_n)
                diag -= a_n
            else:
                pass  # Neumann at top

            add(idx, idx, diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, dirich_mask


def solve_poisson(rhs: np.ndarray, grid: CartesianGrid,
                  bc: BoundaryConfig,
                  A: sp.csr_matrix = None,
                  dirich_mask: np.ndarray = None) -> np.ndarray:
    """
    Solve the pressure Poisson equation  ∇²φ = rhs.

    Parameters
    ----------
    rhs : np.ndarray, shape (nx, ny)
        Right-hand side  ∇·u* / Δt.
    grid : CartesianGrid
    bc : BoundaryConfig
    A : sparse matrix, optional
        Pre-built Poisson matrix.
    dirich_mask : np.ndarray of bool, optional
        Mask of Dirichlet cell positions (where b must be 0).

    Returns
    -------
    phi : np.ndarray, shape (nx, ny)
    """
    nx, ny = grid.nx, grid.ny

    if A is None:
        A, dirich_mask = build_poisson_matrix(grid, bc)

    # Fortran (column-major) order: flat[i + j*nx] = phi[i, j]
    b = rhs.ravel(order='F').copy()

    # Zero Dirichlet entries (φ = 0 there)
    if dirich_mask is not None:
        b[dirich_mask] = 0.0

    # Regularise for purely Neumann systems: pin φ[0,0] = 0
    has_dirichlet = any(t == BCType.OUTFLOW for t in
                        [bc.left, bc.right, bc.bottom, bc.top])
    if not has_dirichlet:
        A = A.tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A = A.tocsr()
        b[0] = 0.0

    phi_flat = spla.spsolve(A, b)
    return phi_flat.reshape((nx, ny), order='F')


class PoissonSolver:
    """
    Caching wrapper around :func:`solve_poisson`.

    Builds the sparse matrix once and reuses it across time steps.
    """

    def __init__(self, grid: CartesianGrid, bc: BoundaryConfig):
        self.grid = grid
        self.bc = bc
        self._A, self._dirich_mask = build_poisson_matrix(grid, bc)

        # Regularise for purely Neumann systems (no outflow Dirichlet BC)
        has_dirichlet = any(t == BCType.OUTFLOW for t in
                            [bc.left, bc.right, bc.bottom, bc.top])
        if not has_dirichlet:
            A_reg = self._A.tolil()
            A_reg[0, :] = 0.0
            A_reg[0, 0] = 1.0
            self._A = A_reg.tocsr()
            self._regularised = True
        else:
            self._regularised = False

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ∇²φ = rhs, returning φ of shape (nx, ny)."""
        nx, ny = self.grid.nx, self.grid.ny
        # Fortran order: flat[i+j*nx] = rhs[i,j]
        b = rhs.ravel(order='F').copy()
        # Zero Dirichlet entries and regularisation pin
        b[self._dirich_mask] = 0.0
        if self._regularised:
            b[0] = 0.0
        phi_flat = spla.spsolve(self._A, b)
        return phi_flat.reshape((nx, ny), order='F')
