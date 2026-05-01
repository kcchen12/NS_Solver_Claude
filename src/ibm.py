"""
Immersed boundary method (IBM) — direct forcing approach.

Theory
------
The immersed boundary method represents solid geometry *inside* the
computational domain by adding a body-force term f to the momentum
equations:

    ∂u/∂t + (u·∇)u = -∇p + ν∇²u + f

where f is chosen to enforce the desired velocity (typically zero) at
solid-body points.

Direct-forcing approach
~~~~~~~~~~~~~~~~~~~~~~~
At each time step, after computing the intermediate velocity u*, we
correct it so that solid-body points match the prescribed velocity:

    u*[solid] = u_body[solid]

This is equivalent to an infinite forcing that drives u to u_body
instantaneously.  While first-order in time at the boundary, it is
simple to implement, robust, and widely used in the literature
(Mohd-Yusof 1997, Fadlun et al. 2000).

Geometry
--------
Solid regions are specified as level-set / mask arrays:

    mask_u[i,j] = 1  if x-face (xf[i], yc[j]) is inside solid
    mask_v[i,j] = 1  if y-face (xc[i], yf[j]) is inside solid

Helper methods allow adding:
    - Circular cylinders
    - Rectangular blocks
    - Arbitrary masks (load from array)
"""

from dataclasses import dataclass

import numpy as np
from src.grid import CartesianGrid


@dataclass
class RotatingCircleSpec:
    cx: float
    cy: float
    radius: float
    omega_amplitude: float
    frequency: float
    phase: float
    mask_u: np.ndarray
    mask_v: np.ndarray

    def angular_velocity(self, time: float) -> float:
        return float(
            self.omega_amplitude * np.sin(2.0 * np.pi * self.frequency * time + self.phase)
        )


@dataclass
class SweepingJetSpec:
    cx: float
    cy: float
    radius: float
    jet_speed: float
    slot_center_angle: float
    slot_width_angle: float
    slot_depth: float
    sweep_amplitude: float
    frequency: float
    phase: float
    mask_u: np.ndarray
    mask_v: np.ndarray

    def sweep_angle(self, time: float) -> float:
        return float(
            self.sweep_amplitude * np.sin(2.0 * np.pi * self.frequency * time + self.phase)
        )


class ImmersedBoundary:
    """
    Manages solid-body masks for the IBM direct-forcing approach.

    Parameters
    ----------
    grid : CartesianGrid
    """

    def __init__(self, grid: CartesianGrid):
        self.grid = grid
        # Binary masks: 1 = solid, 0 = fluid
        self.mask_u = np.zeros(grid.u_shape, dtype=bool)
        self.mask_v = np.zeros(grid.v_shape, dtype=bool)
        self.rotating_circles: list[RotatingCircleSpec] = []
        self.sweeping_jets: list[SweepingJetSpec] = []

    @staticmethod
    def _circle_mask(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> np.ndarray:
        radius_sq = float(radius) ** 2
        return ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) <= radius_sq

    @staticmethod
    def _top_indent_mask(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        indent_width: float,
        indent_depth: float,
    ) -> np.ndarray:
        if indent_width <= 0.0 or indent_depth <= 0.0:
            return np.zeros_like(x_coords, dtype=bool)

        half_width = 0.5 * float(indent_width)
        y0 = float(cy + radius - indent_depth)
        y1 = float(cy + radius)
        return (
            (x_coords >= cx - half_width)
            & (x_coords <= cx + half_width)
            & (y_coords >= y0)
            & (y_coords <= y1)
        )

    def _circle_with_top_indent_masks(
        self,
        cx: float,
        cy: float,
        radius: float,
        indent_width: float,
        indent_depth: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = self.grid

        xf = grid.xf[:, np.newaxis]
        yc = grid.yc[np.newaxis, :]
        mask_u = self._circle_mask(xf, yc, cx, cy, radius)
        mask_u &= ~self._top_indent_mask(
            xf, yc, cx, cy, radius, indent_width, indent_depth
        )

        xc = grid.xc[:, np.newaxis]
        yf = grid.yf[np.newaxis, :]
        mask_v = self._circle_mask(xc, yf, cx, cy, radius)
        mask_v &= ~self._top_indent_mask(
            xc, yf, cx, cy, radius, indent_width, indent_depth
        )
        return mask_u, mask_v

    @staticmethod
    def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _circular_slot_masks(
        self,
        cx: float,
        cy: float,
        radius: float,
        slot_center_angle: float,
        slot_width_angle: float,
        slot_depth: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = self.grid

        xf = grid.xf[:, np.newaxis]
        yc = grid.yc[np.newaxis, :]
        r_u = np.sqrt((xf - cx) ** 2 + (yc - cy) ** 2)
        theta_u = np.arctan2(yc - cy, xf - cx)
        mask_u = (
            (r_u <= radius)
            & (r_u >= radius - slot_depth)
            & (np.abs(self._wrap_angle(theta_u - slot_center_angle)) <= 0.5 * slot_width_angle)
        )

        xc = grid.xc[:, np.newaxis]
        yf = grid.yf[np.newaxis, :]
        r_v = np.sqrt((xc - cx) ** 2 + (yf - cy) ** 2)
        theta_v = np.arctan2(yf - cy, xc - cx)
        mask_v = (
            (r_v <= radius)
            & (r_v >= radius - slot_depth)
            & (np.abs(self._wrap_angle(theta_v - slot_center_angle)) <= 0.5 * slot_width_angle)
        )
        return mask_u, mask_v

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def add_circle(self, cx: float, cy: float, radius: float,
                   u_body: float = 0.0, v_body: float = 0.0) -> None:
        """
        Mark all MAC faces inside a circular cylinder as solid.

        Parameters
        ----------
        cx, cy : float
            Centre of the cylinder in physical coordinates.
        radius : float
            Cylinder radius.
        u_body, v_body : float
            Prescribed velocity on the body surface (0 for stationary wall).
        """
        del u_body, v_body
        grid = self.grid
        xf = grid.xf[:, np.newaxis]
        yc = grid.yc[np.newaxis, :]
        self.mask_u |= self._circle_mask(xf, yc, cx, cy, radius)

        xc = grid.xc[:, np.newaxis]
        yf = grid.yf[np.newaxis, :]
        self.mask_v |= self._circle_mask(xc, yf, cx, cy, radius)

    def add_circle_with_top_indent(
        self,
        cx: float,
        cy: float,
        radius: float,
        indent_width: float,
        indent_depth: float,
        u_body: float = 0.0,
        v_body: float = 0.0,
    ) -> None:
        """
        Mark a circular body with a rectangular cut-out at the top as solid.

        The indent is centered at x=cx and removes the region
        [cx-indent_width/2, cx+indent_width/2] x [cy+radius-indent_depth, cy+radius].
        """
        del u_body, v_body
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if indent_width <= 0.0 or indent_depth <= 0.0:
            raise ValueError("indent_width and indent_depth must be positive")
        if indent_width >= 2.0 * radius:
            raise ValueError("indent_width must be smaller than the cylinder diameter")
        if indent_depth >= 2.0 * radius:
            raise ValueError("indent_depth must be smaller than the cylinder diameter")

        mask_u, mask_v = self._circle_with_top_indent_masks(
            cx=cx,
            cy=cy,
            radius=radius,
            indent_width=indent_width,
            indent_depth=indent_depth,
        )
        self.mask_u |= mask_u
        self.mask_v |= mask_v

    def add_rotating_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        omega_amplitude: float,
        frequency: float,
        phase: float = 0.0,
    ) -> None:
        """Add a circular cylinder with sinusoidal back-and-forth rotation.

        The imposed angular velocity is

            omega(t) = omega_amplitude * sin(2*pi*frequency*t + phase)

        with tangential wall velocity

            u = -omega(t) * (y - cy)
            v =  omega(t) * (x - cx)
        """
        grid = self.grid

        xf = grid.xf[:, np.newaxis]
        yc = grid.yc[np.newaxis, :]
        mask_u = self._circle_mask(xf, yc, cx, cy, radius)

        xc = grid.xc[:, np.newaxis]
        yf = grid.yf[np.newaxis, :]
        mask_v = self._circle_mask(xc, yf, cx, cy, radius)

        self.mask_u |= mask_u
        self.mask_v |= mask_v
        self.rotating_circles.append(
            RotatingCircleSpec(
                cx=float(cx),
                cy=float(cy),
                radius=float(radius),
                omega_amplitude=float(omega_amplitude),
                frequency=float(frequency),
                phase=float(phase),
                mask_u=mask_u,
                mask_v=mask_v,
            )
        )

    def add_constant_rotating_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        omega: float,
    ) -> None:
        """Add a circular cylinder with constant one-direction rotation.

        The imposed angular velocity is

            omega(t) = omega

        with tangential wall velocity

            u = -omega * (y - cy)
            v =  omega * (x - cx)
        """
        self.add_rotating_circle(
            cx=cx,
            cy=cy,
            radius=radius,
            omega_amplitude=float(omega),
            frequency=0.0,
            phase=0.5 * np.pi,
        )

    def add_sweeping_jet_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        jet_speed: float,
        slot_center_angle_deg: float = 90.0,
        slot_width_angle_deg: float = 18.0,
        slot_depth: float = 0.0,
        sweep_amplitude_deg: float = 25.0,
        frequency: float = 0.0,
        phase: float = 0.0,
    ) -> None:
        """
        Add a finite-width sweeping jet outlet on a circular IBM body.

        The outlet location is fixed on the body surface, while the jet
        direction oscillates relative to the local outward normal:

            u_jet = U_j * (cos(alpha(t)) * n_hat + sin(alpha(t)) * t_hat)

        where alpha(t) = sweep_amplitude * sin(2*pi*frequency*t + phase).
        """
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if jet_speed < 0.0:
            raise ValueError("jet_speed must be non-negative")

        slot_width_angle = np.deg2rad(float(slot_width_angle_deg))
        if slot_width_angle <= 0.0 or slot_width_angle >= 2.0 * np.pi:
            raise ValueError("slot_width_angle_deg must be in (0, 360)")

        if slot_depth <= 0.0:
            slot_depth = 0.15 * radius
        if slot_depth >= radius:
            raise ValueError("slot_depth must be smaller than the cylinder radius")

        mask_u, mask_v = self._circular_slot_masks(
            cx=cx,
            cy=cy,
            radius=radius,
            slot_center_angle=np.deg2rad(float(slot_center_angle_deg)),
            slot_width_angle=slot_width_angle,
            slot_depth=float(slot_depth),
        )

        self.add_circle(cx, cy, radius)
        self.sweeping_jets.append(
            SweepingJetSpec(
                cx=float(cx),
                cy=float(cy),
                radius=float(radius),
                jet_speed=float(jet_speed),
                slot_center_angle=np.deg2rad(float(slot_center_angle_deg)),
                slot_width_angle=slot_width_angle,
                slot_depth=float(slot_depth),
                sweep_amplitude=np.deg2rad(float(sweep_amplitude_deg)),
                frequency=float(frequency),
                phase=float(phase),
                mask_u=mask_u,
                mask_v=mask_v,
            )
        )

    def add_rectangle(self, x0: float, x1: float,
                      y0: float, y1: float,
                      u_body: float = 0.0, v_body: float = 0.0) -> None:
        """
        Mark all MAC faces inside axis-aligned rectangle [x0,x1]×[y0,y1].
        """
        del u_body, v_body
        grid = self.grid
        self.mask_u |= (
            (grid.xf[:, np.newaxis] >= x0)
            & (grid.xf[:, np.newaxis] <= x1)
            & (grid.yc[np.newaxis, :] >= y0)
            & (grid.yc[np.newaxis, :] <= y1)
        )
        self.mask_v |= (
            (grid.xc[:, np.newaxis] >= x0)
            & (grid.xc[:, np.newaxis] <= x1)
            & (grid.yf[np.newaxis, :] >= y0)
            & (grid.yf[np.newaxis, :] <= y1)
        )

    def add_mask(self, mask_u: np.ndarray, mask_v: np.ndarray) -> None:
        """
        Directly supply boolean masks for u and v faces.
        """
        assert mask_u.shape == self.grid.u_shape
        assert mask_v.shape == self.grid.v_shape
        self.mask_u |= mask_u
        self.mask_v |= mask_v

    # ------------------------------------------------------------------
    # Forcing
    # ------------------------------------------------------------------

    def apply(self, u: np.ndarray, v: np.ndarray,
              u_body: float = 0.0, v_body: float = 0.0,
              dt: float = None, rho: float = 1.0,
              return_face_forcing: bool = False,
              time: float = 0.0):
        """
        Apply direct forcing **in-place**: set solid-face velocities to the
        prescribed body velocity (default: 0 for stationary body).

        Call this *after* computing the intermediate velocity u* and
        *before* the pressure-correction step.
        """
        force_x = 0.0
        force_y = 0.0
        forcing_u = None
        forcing_v = None
        u_target = np.full_like(u, float(u_body))
        v_target = np.full_like(v, float(v_body))

        if self.rotating_circles:
            u_y = np.broadcast_to(
                self.grid.yc[np.newaxis, :], self.grid.u_shape)
            v_x = np.broadcast_to(
                self.grid.xc[:, np.newaxis], self.grid.v_shape)
            for spec in self.rotating_circles:
                omega = spec.angular_velocity(time)
                u_target[spec.mask_u] = -omega * (u_y[spec.mask_u] - spec.cy)
                v_target[spec.mask_v] = omega * (v_x[spec.mask_v] - spec.cx)

        if self.sweeping_jets:
            u_x = np.broadcast_to(self.grid.xf[:, np.newaxis], self.grid.u_shape)
            u_y = np.broadcast_to(self.grid.yc[np.newaxis, :], self.grid.u_shape)
            v_x = np.broadcast_to(self.grid.xc[:, np.newaxis], self.grid.v_shape)
            v_y = np.broadcast_to(self.grid.yf[np.newaxis, :], self.grid.v_shape)
            for spec in self.sweeping_jets:
                alpha = spec.sweep_angle(time)

                ru = np.sqrt((u_x - spec.cx) ** 2 + (u_y - spec.cy) ** 2)
                rv = np.sqrt((v_x - spec.cx) ** 2 + (v_y - spec.cy) ** 2)
                ru = np.where(ru > 1e-14, ru, 1.0)
                rv = np.where(rv > 1e-14, rv, 1.0)

                nu_x = (u_x - spec.cx) / ru
                nu_y = (u_y - spec.cy) / ru
                nv_x = (v_x - spec.cx) / rv
                nv_y = (v_y - spec.cy) / rv

                tu_x = -nu_y
                tu_y = nu_x
                tv_x = -nv_y
                tv_y = nv_x

                u_target[spec.mask_u] = spec.jet_speed * (
                    np.cos(alpha) * nu_x[spec.mask_u] + np.sin(alpha) * tu_x[spec.mask_u]
                )
                v_target[spec.mask_v] = spec.jet_speed * (
                    np.cos(alpha) * nv_y[spec.mask_v] + np.sin(alpha) * tv_y[spec.mask_v]
                )

        if dt is not None and dt > 0.0:
            face_area = self.grid.mean_cell_area
            force_x = rho * face_area * \
                float(np.sum(u[self.mask_u] - u_target[self.mask_u])) / dt
            force_y = rho * face_area * \
                float(np.sum(v[self.mask_v] - v_target[self.mask_v])) / dt
            if return_face_forcing:
                forcing_u = np.zeros_like(u)
                forcing_v = np.zeros_like(v)
                forcing_u[self.mask_u] = (
                    u_target[self.mask_u] - u[self.mask_u]) / dt
                forcing_v[self.mask_v] = (
                    v_target[self.mask_v] - v[self.mask_v]) / dt
        elif return_face_forcing:
            forcing_u = np.zeros_like(u)
            forcing_v = np.zeros_like(v)

        u[self.mask_u] = u_target[self.mask_u]
        v[self.mask_v] = v_target[self.mask_v]
        if return_face_forcing:
            return force_x, force_y, forcing_u, forcing_v
        return force_x, force_y

    @property
    def has_solid(self) -> bool:
        """True if any solid cells are defined."""
        return bool(self.mask_u.any() or self.mask_v.any())
