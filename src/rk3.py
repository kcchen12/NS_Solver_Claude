"""
3rd-order Strong-Stability-Preserving Runge-Kutta integrator (SSP-RK3).

The Shu-Osher (1988) three-stage scheme reads:

    u¹  =  uⁿ + Δt · L(uⁿ)
    u²  =  ¾ uⁿ  + ¼ (u¹ + Δt · L(u¹))
    uⁿ⁺¹ = ⅓ uⁿ  + ⅔ (u² + Δt · L(u²))

where L denotes the spatial operator (convection + diffusion).

This is equivalent to the classic RK3 but expressed in Shu-Osher form
that preserves the TVD / SSP property, making it well-suited for
hyperbolic-dominated problems.

Usage with the fractional-step solver
--------------------------------------
The integrator is given a callable ``rhs_fn(u, v) → (Lu, Lv)`` that
evaluates the convection-diffusion right-hand side **excluding pressure**.
Pressure correction is applied once, after all RK stages, by the
:class:`~src.solver.FractionalStepSolver`.
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
    Advance (u, v) by one time step Δt using SSP-RK3.

    Parameters
    ----------
    rhs_fn : callable
        ``rhs_fn(u, v)`` → ``(Lu, Lv)`` where Lu and Lv are the
        convection-diffusion right-hand sides for the interior faces.
        The returned arrays must match the *interior* shapes:
            Lu : shape (nx-1, ny)
            Lv : shape (nx, ny-1)
    u, v : np.ndarray
        Current velocity fields (full arrays including boundary faces).
    dt : float
        Time-step size.

    Returns
    -------
    u_new, v_new : np.ndarray
        Updated interior values.  Boundary values are *not* modified
        here; the caller (:class:`~src.solver.FractionalStepSolver`)
        applies BCs after each stage.
    """
    # Stage 1
    Lu1, Lv1 = rhs_fn(u, v)
    u1 = u.copy()
    v1 = v.copy()
    u1[1:-1, :] = u[1:-1, :] + dt * Lu1
    v1[:, 1:-1] = v[:, 1:-1] + dt * Lv1

    # Stage 2
    Lu2, Lv2 = rhs_fn(u1, v1)
    u2 = u.copy()
    v2 = v.copy()
    u2[1:-1, :] = 0.75 * u[1:-1, :] + 0.25 * (u1[1:-1, :] + dt * Lu2)
    v2[:, 1:-1] = 0.75 * v[:, 1:-1] + 0.25 * (v1[:, 1:-1] + dt * Lv2)

    # Stage 3  →  result
    Lu3, Lv3 = rhs_fn(u2, v2)
    u_new = u.copy()
    v_new = v.copy()
    u_new[1:-1, :] = (1.0 / 3.0) * u[1:-1, :] + (2.0 / 3.0) * (u2[1:-1, :] + dt * Lu3)
    v_new[:, 1:-1] = (1.0 / 3.0) * v[:, 1:-1] + (2.0 / 3.0) * (v2[:, 1:-1] + dt * Lv3)

    return u_new, v_new
