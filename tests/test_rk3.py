"""Tests for the SSP-RK3 integrator."""
import numpy as np
import pytest
from src.rk3 import ssp_rk3_step


# ---------------------------------------------------------------------------
# Simple ODE test:  du/dt = -u  →  u(t) = u0 * exp(-t)
# ---------------------------------------------------------------------------

def make_simple_decay_rhs(lam=1.0):
    """Return an RHS function for du/dt = -lam*u (using dummy v=0)."""
    def rhs(u, v):
        # Only interior values (shape (nx-1, ny) for u)
        Lu = -lam * u[1:-1, :]
        Lv = np.zeros((v.shape[0], v.shape[1] - 2))
        return Lu, Lv
    return rhs


class TestSSPRK3:
    def test_order_of_accuracy(self):
        """SSP-RK3 should be 3rd-order accurate for a simple ODE."""
        # du/dt = -u,  u(0) = 1  →  u(1) = exp(-1)
        u_exact = np.exp(-1.0)

        errors = []
        for n_steps in [10, 20, 40]:
            dt = 1.0 / n_steps
            # Minimal 1D problem: nx=2 (one interior u-face), ny=1
            u = np.ones((3, 1))   # u shape (nx+1, ny) = (3, 1)
            v = np.zeros((2, 2))  # v shape (nx, ny+1) = (2, 2)
            rhs = make_simple_decay_rhs(lam=1.0)
            for _ in range(n_steps):
                u, v = ssp_rk3_step(rhs, u, v, dt)
            errors.append(abs(u[1, 0] - u_exact))

        # Check convergence rate ≈ 3
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 6.0, f"Expected ≥3rd-order but ratio={ratio_1:.2f}"
        assert ratio_2 > 6.0, f"Expected ≥3rd-order but ratio={ratio_2:.2f}"

    def test_zero_rhs_no_change(self):
        """If rhs = 0 the solution should not change."""
        def zero_rhs(u, v):
            return np.zeros((u.shape[0] - 2, u.shape[1])), \
                   np.zeros((v.shape[0], v.shape[1] - 2))

        u0 = np.random.rand(5, 4)
        v0 = np.random.rand(4, 5)
        u_new, v_new = ssp_rk3_step(zero_rhs, u0, v0, dt=0.1)
        # Interior values unchanged
        assert np.allclose(u_new[1:-1, :], u0[1:-1, :])
        assert np.allclose(v_new[:, 1:-1], v0[:, 1:-1])

    def test_boundary_values_unchanged(self):
        """Boundary face values of u and v must not be touched by RK3."""
        def small_rhs(u, v):
            return np.ones((u.shape[0] - 2, u.shape[1])), \
                   np.ones((v.shape[0], v.shape[1] - 2))

        u0 = np.random.rand(7, 5)
        v0 = np.random.rand(6, 6)
        u_new, v_new = ssp_rk3_step(small_rhs, u0, v0, dt=0.05)
        # Boundary faces (i=0, i=nx) and (j=0, j=ny) untouched
        assert np.allclose(u_new[0, :],  u0[0, :])
        assert np.allclose(u_new[-1, :], u0[-1, :])
        assert np.allclose(v_new[:, 0],  v0[:, 0])
        assert np.allclose(v_new[:, -1], v0[:, -1])
