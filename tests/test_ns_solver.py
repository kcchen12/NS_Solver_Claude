"""Tests for the 2D incompressible Navier-Stokes solver (ns_solver.py)."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ns_solver import NSSolver2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _taylor_green_ic(solver):
    """Return Taylor-Green vortex initial condition arrays."""
    X, Y = solver.X, solver.Y
    u0 = np.sin(X) * np.cos(Y)
    v0 = -np.cos(X) * np.sin(Y)
    return u0, v0


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_no_slip_shapes(self):
        s = NSSolver2D(nx=11, ny=11, Re=100, dt=0.001)
        assert s.u.shape == (11, 11)
        assert s.v.shape == (11, 11)
        assert s.p.shape == (11, 11)

    def test_periodic_shapes(self):
        s = NSSolver2D(nx=16, ny=16, Re=100, dt=0.001, bc='periodic')
        assert s.u.shape == (16, 16)
        assert s.v.shape == (16, 16)
        assert s.p.shape == (16, 16)

    def test_no_slip_grid_spacing(self):
        s = NSSolver2D(nx=11, ny=11, Re=100, dt=0.001, Lx=2.0, Ly=1.0)
        assert s.dx == pytest.approx(0.2)
        assert s.dy == pytest.approx(0.1)

    def test_periodic_grid_spacing(self):
        s = NSSolver2D(nx=10, ny=10, Re=100, dt=0.001, Lx=1.0, Ly=1.0,
                       bc='periodic')
        assert s.dx == pytest.approx(0.1)
        assert s.dy == pytest.approx(0.1)

    def test_invalid_bc_raises(self):
        with pytest.raises(ValueError, match="bc must be"):
            NSSolver2D(nx=11, ny=11, Re=100, dt=0.001, bc='invalid')

    def test_default_zero_velocity(self):
        s = NSSolver2D(nx=11, ny=11, Re=100, dt=0.001)
        assert np.all(s.u == 0)
        assert np.all(s.v == 0)

    def test_set_initial_condition(self):
        s = NSSolver2D(nx=8, ny=8, Re=100, dt=0.001)
        u0 = np.ones((8, 8))
        v0 = np.zeros((8, 8))
        s.set_initial_condition(u0, v0)
        np.testing.assert_array_equal(s.u, u0)
        np.testing.assert_array_equal(s.v, v0)

    def test_set_initial_condition_wrong_shape(self):
        s = NSSolver2D(nx=8, ny=8, Re=100, dt=0.001)
        with pytest.raises(ValueError, match="Expected shape"):
            s.set_initial_condition(np.ones((5, 5)), np.ones((5, 5)))

    def test_meshgrid_shape(self):
        s = NSSolver2D(nx=9, ny=7, Re=100, dt=0.001)
        assert s.X.shape == (7, 9)
        assert s.Y.shape == (7, 9)


# ---------------------------------------------------------------------------
# Lid-driven cavity (no-slip BCs)
# ---------------------------------------------------------------------------

class TestLidDrivenCavity:
    def _solver(self, nx=13, ny=13, Re=100, dt=0.001, u_lid=1.0):
        s = NSSolver2D(nx=nx, ny=ny, Re=Re, dt=dt)
        s.set_lid_driven_cavity(u_lid=u_lid)
        return s

    def test_lid_bc_top_wall(self):
        s = self._solver()
        # Corner nodes are shared with the side walls (u=0 there); check
        # only the interior portion of the lid.
        np.testing.assert_array_equal(s.u[-1, 1:-1], 1.0)
        np.testing.assert_array_equal(s.v[-1, :], 0.0)

    def test_lid_bc_other_walls(self):
        s = self._solver()
        np.testing.assert_array_equal(s.u[0, :], 0.0)
        np.testing.assert_array_equal(s.u[:, 0], 0.0)
        np.testing.assert_array_equal(s.u[:, -1], 0.0)
        np.testing.assert_array_equal(s.v[0, :], 0.0)

    def test_lid_bc_wrong_mode(self):
        s = NSSolver2D(nx=11, ny=11, Re=100, dt=0.001, bc='periodic')
        with pytest.raises(ValueError, match="bc='no_slip'"):
            s.set_lid_driven_cavity()

    def test_step_return_shapes(self):
        s = self._solver()
        u, v, p = s.step()
        assert u.shape == (13, 13)
        assert v.shape == (13, 13)
        assert p.shape == (13, 13)

    def test_bc_preserved_after_step(self):
        s = self._solver()
        s.step()
        # Interior lid nodes stay at u_lid; corners belong to the side walls.
        np.testing.assert_array_equal(s.u[-1, 1:-1], 1.0)
        np.testing.assert_array_equal(s.u[0, :], 0.0)

    def test_flow_develops(self):
        """Interior velocity must become non-zero after the lid starts moving."""
        s = self._solver()
        s.run(T=0.01)
        assert np.any(s.u[1:-1, 1:-1] != 0)

    def test_divergence_small_after_run(self):
        """Projection must keep ∇·u small.

        On a collocated grid the div-grad and Laplacian operators differ by
        O(dx²), so the residual divergence after projection is O(dt·dx²)
        rather than machine precision.  We test interior cells away from the
        moving lid where the boundary-layer divergence is larger.
        """
        s = self._solver()
        s.run(T=0.005)
        div = s.divergence()
        # Exclude the single row adjacent to the moving lid (j = ny-2)
        # where the projection error is dominated by the Dirichlet BC jump.
        assert np.max(np.abs(div[1:-2, 1:-1])) < 1e-2

    def test_kinetic_energy_increases_from_rest(self):
        """Energy should increase from zero when the lid is set in motion."""
        s = self._solver()
        s.run(T=0.01)
        assert s.kinetic_energy() > 0


# ---------------------------------------------------------------------------
# Periodic BCs – Taylor-Green vortex
# ---------------------------------------------------------------------------

class TestPeriodicTaylorGreen:
    def _solver(self, n=32, Re=10.0, dt=0.001):
        s = NSSolver2D(nx=n, ny=n, Re=Re, dt=dt,
                       Lx=2 * np.pi, Ly=2 * np.pi, bc='periodic')
        u0, v0 = _taylor_green_ic(s)
        s.set_initial_condition(u0, v0)
        return s

    def test_initial_divergence_free(self):
        """Taylor-Green IC is analytically div-free; FD should give ~0 on a
        square grid because the central-difference truncation errors cancel."""
        s = self._solver()
        div = s.divergence()
        assert np.max(np.abs(div)) < 1e-10

    def test_divergence_free_after_run(self):
        """Projection step must maintain the divergence-free constraint.

        On a collocated grid the div-grad and Laplacian operators differ by
        O(dx²), giving a residual divergence of O(dt·dx²) per step.
        For dt=0.001 and dx=2π/32 ≈ 0.196 the expected magnitude is ~4e-5.
        """
        s = self._solver()
        s.run(T=0.01)
        div = s.divergence()
        assert np.max(np.abs(div)) < 1e-3

    def test_energy_decays_with_viscosity(self):
        """For Re < ∞ the kinetic energy must decrease."""
        s = self._solver(Re=10.0)
        E0 = s.kinetic_energy()
        s.run(T=0.1)
        assert s.kinetic_energy() < E0

    def test_analytical_solution(self):
        """Velocity should match the analytical Taylor-Green decay within a
        loose tolerance (first-order time, second-order space)."""
        Re = 10.0
        T = 0.1
        s = self._solver(Re=Re)
        s.run(T=T)

        decay = np.exp(-2.0 * T / Re)
        u_exact = np.sin(s.X) * np.cos(s.Y) * decay
        v_exact = -np.cos(s.X) * np.sin(s.Y) * decay

        assert np.max(np.abs(s.u - u_exact)) < 0.05
        assert np.max(np.abs(s.v - v_exact)) < 0.05

    def test_step_returns_correct_shapes(self):
        s = self._solver()
        u, v, p = s.step()
        assert u.shape == (32, 32)
        assert v.shape == (32, 32)
        assert p.shape == (32, 32)


# ---------------------------------------------------------------------------
# Divergence utility
# ---------------------------------------------------------------------------

class TestDivergence:
    def test_zero_velocity_zero_divergence_noslip(self):
        s = NSSolver2D(nx=11, ny=11, Re=100, dt=0.001)
        div = s.divergence()
        assert np.all(div == 0)

    def test_zero_velocity_zero_divergence_periodic(self):
        s = NSSolver2D(nx=16, ny=16, Re=100, dt=0.001, bc='periodic')
        div = s.divergence()
        assert np.all(div == 0)


# ---------------------------------------------------------------------------
# Kinetic energy utility
# ---------------------------------------------------------------------------

class TestKineticEnergy:
    def test_zero_energy_for_zero_velocity(self):
        s = NSSolver2D(nx=11, ny=11, Re=100, dt=0.001)
        assert s.kinetic_energy() == 0.0

    def test_energy_positive_after_lid_run(self):
        s = NSSolver2D(nx=11, ny=11, Re=100, dt=0.001)
        s.set_lid_driven_cavity(u_lid=1.0)
        s.run(T=0.01)
        assert s.kinetic_energy() > 0

    def test_energy_scales_with_velocity(self):
        """Doubling velocity should quadruple kinetic energy."""
        s = NSSolver2D(nx=11, ny=11, Re=100, dt=0.001, bc='periodic')
        s.set_initial_condition(np.ones((11, 11)), np.zeros((11, 11)))
        E1 = s.kinetic_energy()
        s.set_initial_condition(2 * np.ones((11, 11)), np.zeros((11, 11)))
        E2 = s.kinetic_energy()
        assert E2 == pytest.approx(4 * E1)
