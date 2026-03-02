"""Tests for boundary condition enforcement."""
import numpy as np
import pytest
from src.grid     import CartesianGrid
from src.boundary import BoundaryConfig, BCType, apply_velocity_bc, apply_pressure_bc


def make_grid(nx=8, ny=6):
    return CartesianGrid(nx, ny, lx=2.0, ly=1.0)


class TestInflowBC:
    def test_left_inflow_sets_u(self):
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL,
                            u_inf=1.5, v_inf=0.0)
        u = np.zeros(g.u_shape)
        v = np.zeros(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        assert np.allclose(u[0, :], 1.5)

    def test_left_inflow_sets_v(self):
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL,
                            u_inf=1.0, v_inf=0.2)
        u = np.zeros(g.u_shape)
        v = np.zeros(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        # Wall BCs override the corners (j=0 and j=ny); check interior faces only
        assert np.allclose(v[0, 1:-1], 0.2)


class TestWallBC:
    def test_bottom_wall_no_penetration(self):
        """v[:, 0] = 0 at solid bottom wall."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL)
        u = np.ones(g.u_shape)
        v = np.ones(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        assert np.allclose(v[:, 0], 0.0)

    def test_top_wall_no_penetration(self):
        """v[:, ny] = 0 at solid top wall."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL)
        u = np.ones(g.u_shape)
        v = np.ones(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        assert np.allclose(v[:, g.ny], 0.0)


class TestConvectiveOutflow:
    def test_outflow_zero_gradient_fallback(self):
        """Without dt, outflow uses zero-gradient extrapolation."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL,
                            u_inf=1.0)
        u = np.random.rand(*g.u_shape)
        v = np.zeros(g.v_shape)
        u_before = u[g.nx - 1, :].copy()
        apply_velocity_bc(u, v, g, bc, dt=None)
        assert np.allclose(u[g.nx, :], u_before)

    def test_outflow_with_dt(self):
        """With dt provided, outflow applies convective update."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL,
                            u_inf=1.0)
        u = np.ones(g.u_shape) * 1.0
        v = np.zeros(g.v_shape)
        dt = 0.01
        apply_velocity_bc(u, v, g, bc, dt=dt)
        # For uniform u=1, the convective update gives no change
        assert np.allclose(u[g.nx, :], 1.0, atol=1e-12)


class TestPressureBC:
    def test_neumann_left(self):
        """Left Neumann: phi[0] = phi[1] for interior j."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL)
        phi = np.random.rand(*g.p_shape)
        phi_interior = phi[1, :].copy()
        apply_pressure_bc(phi, g, bc)
        # Bottom/top Neumann BCs also modify phi[0,0] and phi[0,ny-1],
        # so check only interior j indices
        assert np.allclose(phi[0, 1:-1], phi_interior[1:-1])

    def test_dirichlet_right_outflow(self):
        """Right outflow (Dirichlet): phi[-1] = 0."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL)
        phi = np.random.rand(*g.p_shape) + 1.0
        apply_pressure_bc(phi, g, bc)
        assert np.allclose(phi[g.nx - 1, :], 0.0)
