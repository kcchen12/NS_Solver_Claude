"""
3rd-order Strong-Stability-Preserving Runge-Kutta integrator (SSP-RK3).

The integrator advances the interior velocity degrees of freedom while
leaving boundary faces untouched. Boundary conditions are enforced by the
caller between stages.
"""

from typing import Callable, Tuple

import numpy as np


ArrayPair = Tuple[np.ndarray, np.ndarray]


def ssp_rk3_step(
    rhs_fn: Callable[[np.ndarray, np.ndarray], ArrayPair],
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
) -> ArrayPair:
    """
    Advance (u, v) by one time step using the Shu-Osher SSP-RK3 scheme.

    The RHS function must return arrays with the interior shapes:
        Lu : shape (nx-1, ny)
        Lv : shape (nx, ny-1)
    """
    u_int = np.s_[1:-1, :]
    v_int = np.s_[:, 1:-1]
    u_shape = u[u_int].shape
    v_shape = v[v_int].shape

    def _validate_shapes(Lu: np.ndarray, Lv: np.ndarray, stage: str) -> None:
        if Lu.shape != u_shape:
            raise ValueError(f"{stage}: Lu must have shape {u_shape}, got {Lu.shape}")
        if Lv.shape != v_shape:
            raise ValueError(f"{stage}: Lv must have shape {v_shape}, got {Lv.shape}")

    Lu1, Lv1 = rhs_fn(u, v)
    _validate_shapes(Lu1, Lv1, "stage 1")
    u1 = u.copy()
    v1 = v.copy()
    u1[u_int] = u[u_int] + dt * Lu1
    v1[v_int] = v[v_int] + dt * Lv1

    Lu2, Lv2 = rhs_fn(u1, v1)
    _validate_shapes(Lu2, Lv2, "stage 2")
    u2 = u.copy()
    v2 = v.copy()
    u2[u_int] = 0.75 * u[u_int] + 0.25 * (u1[u_int] + dt * Lu2)
    v2[v_int] = 0.75 * v[v_int] + 0.25 * (v1[v_int] + dt * Lv2)

    Lu3, Lv3 = rhs_fn(u2, v2)
    _validate_shapes(Lu3, Lv3, "stage 3")
    u_new = u.copy()
    v_new = v.copy()
    u_new[u_int] = (1.0 / 3.0) * u[u_int] + (2.0 / 3.0) * (u2[u_int] + dt * Lu3)
    v_new[v_int] = (1.0 / 3.0) * v[v_int] + (2.0 / 3.0) * (v2[v_int] + dt * Lv3)
    return u_new, v_new
