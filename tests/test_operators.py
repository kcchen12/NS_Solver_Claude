"""Tests for FVM spatial operators."""
import numpy as np
import pytest
from src.grid      import CartesianGrid
from src.boundary  import BoundaryConfig, BCType
from src.operators import divergence, laplacian_u, laplacian_v, rhs_u, rhs_v


# ---------------------------------------------------------------------------
# Helper: uniform-flow configuration
# ---------------------------------------------------------------------------

def uniform_bc(u_inf=1.0):
    return BoundaryConfig(
        left=BCType.INFLOW, right=BCType.OUTFLOW,
        bottom=BCType.WALL, top=BCType.WALL,
        u_inf=u_inf, v_inf=0.0,
    )


def make_uniform_fields(grid, u_inf=1.0):
    """Create uniform flow: u=u_inf, v=0 everywhere."""
    u = np.full(grid.u_shape, u_inf)
    v = np.zeros(grid.v_shape)
    return u, v


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

class TestDivergence:
    def test_uniform_flow_zero_div(self):
        """Divergence of a uniform field must be zero."""
        g = CartesianGrid(8, 6, lx=2.0, ly=1.0)
        u, v = make_uniform_fields(g, u_inf=1.0)
        div = divergence(u, v, g)
        assert div.shape == g.p_shape
        assert np.allclose(div, 0.0)

    def test_linear_x_velocity(self):
        """u = x → ∇·u = du/dx = 1/lx per cell."""
        g = CartesianGrid(4, 3, lx=1.0, ly=1.0)
        u = np.zeros(g.u_shape)
        for i in range(g.nx + 1):
            u[i, :] = g.xf[i]          # u[i,j] = xf[i]
        v = np.zeros(g.v_shape)
        div = divergence(u, v, g)
        # du/dx = 1 everywhere
        assert np.allclose(div, 1.0)


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

class TestLaplacianU:
    def test_uniform_field_zero_laplacian(self):
        """∇²u = 0 for a uniform u field (interior y-rows only).

        Near WALL boundaries the no-slip ghost cell creates a non-zero
        Laplacian; we check only j=1..ny-2 where no ghost is needed.
        """
        g = CartesianGrid(8, 6, lx=2.0, ly=1.0)
        bc = uniform_bc()
        u = np.ones(g.u_shape)
        lap = laplacian_u(u, g, bc)
        assert lap.shape == (g.nx - 1, g.ny)
        # Interior y-rows are unaffected by wall ghost cells
        assert np.allclose(lap[:, 1:-1], 0.0, atol=1e-12)

    def test_quadratic_x(self):
        """u = x² → d²u/dx² = 2 (interior y-rows only)."""
        g = CartesianGrid(8, 4, lx=1.0, ly=1.0)
        bc = uniform_bc()
        u = np.zeros(g.u_shape)
        for i in range(g.nx + 1):
            u[i, :] = g.xf[i]**2
        lap = laplacian_u(u, g, bc)
        # Check d²u/dx² = 2 at interior y rows only (wall ghosts affect j=0,j=ny-1)
        assert np.allclose(lap[:, 1:-1], 2.0, atol=1e-10)


class TestLaplacianV:
    def test_uniform_field_zero_laplacian(self):
        """∇²v = 0 for a uniform v field (interior x-columns only).

        Near INFLOW/OUTFLOW boundaries the ghost cells create non-zero
        Laplacian at i=0; we check only i=1..nx-2.
        """
        g = CartesianGrid(8, 6, lx=2.0, ly=1.0)
        bc = uniform_bc()
        v = np.ones(g.v_shape)
        lap = laplacian_v(v, g, bc)
        assert lap.shape == (g.nx, g.ny - 1)
        # Interior x-columns are unaffected by left/right BC ghost cells
        assert np.allclose(lap[1:-1, :], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Convection (uniform flow → zero convective acceleration)
# ---------------------------------------------------------------------------

class TestRHS:
    def test_uniform_rhs_u_zero_convection(self):
        """For uniform u=const, v=0: convective term -(u·∇)u = 0."""
        g = CartesianGrid(8, 6, lx=2.0, ly=1.0)
        bc = uniform_bc(u_inf=1.0)
        u, v = make_uniform_fields(g, u_inf=1.0)
        conv = rhs_u(u, v, g, bc, nu=0.0)  # nu=0 → pure convection
        assert np.allclose(conv, 0.0, atol=1e-12)

    def test_uniform_rhs_v_zero_convection(self):
        """For uniform flow, v-convection must also be zero."""
        g = CartesianGrid(8, 6, lx=2.0, ly=1.0)
        bc = uniform_bc(u_inf=1.0)
        u, v = make_uniform_fields(g, u_inf=1.0)
        conv = rhs_v(u, v, g, bc, nu=0.0)
        assert np.allclose(conv, 0.0, atol=1e-12)

    def test_rhs_shapes(self):
        g = CartesianGrid(10, 8)
        bc = uniform_bc()
        u, v = make_uniform_fields(g)
        Lu = rhs_u(u, v, g, bc, nu=0.01)
        Lv = rhs_v(u, v, g, bc, nu=0.01)
        assert Lu.shape == (g.nx - 1, g.ny)
        assert Lv.shape == (g.nx, g.ny - 1)
