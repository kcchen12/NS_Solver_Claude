"""Tests for the immersed boundary method."""
import numpy as np
import pytest
from src.grid import CartesianGrid
from src.ibm  import ImmersedBoundary


class TestIBMCircle:
    def setup_method(self):
        self.g = CartesianGrid(16, 12, lx=2.0, ly=1.5)
        self.ibm = ImmersedBoundary(self.g)

    def test_no_solid_by_default(self):
        assert not self.ibm.has_solid

    def test_add_circle_marks_solid(self):
        self.ibm.add_circle(1.0, 0.75, 0.2)
        assert self.ibm.has_solid

    def test_circle_centre_is_solid(self):
        cx, cy, r = 1.0, 0.75, 0.3
        self.ibm.add_circle(cx, cy, r)
        g = self.g
        # Find u-face index closest to centre
        i_c = int(np.argmin(np.abs(g.xf - cx)))
        j_c = int(np.argmin(np.abs(g.yc - cy)))
        assert self.ibm.mask_u[i_c, j_c]

    def test_apply_zeros_solid_cells(self):
        self.ibm.add_circle(1.0, 0.75, 0.2)
        u = np.ones(self.g.u_shape)
        v = np.ones(self.g.v_shape)
        self.ibm.apply(u, v)
        # All solid u-faces must be 0
        assert np.all(u[self.ibm.mask_u] == 0.0)
        assert np.all(v[self.ibm.mask_v] == 0.0)

    def test_fluid_cells_unchanged(self):
        self.ibm.add_circle(1.0, 0.75, 0.1)
        u = np.ones(self.g.u_shape) * 2.0
        v = np.ones(self.g.v_shape) * 3.0
        self.ibm.apply(u, v)
        # Fluid u-faces must remain at original value
        assert np.all(u[~self.ibm.mask_u] == 2.0)
        assert np.all(v[~self.ibm.mask_v] == 3.0)

    def test_rotating_circle_imposes_tangential_velocity(self):
        cx, cy, r = 1.0, 0.75, 0.3
        omega = 2.5
        self.ibm.add_rotating_circle(
            cx, cy, r, omega_amplitude=omega, frequency=0.0, phase=np.pi / 2.0
        )
        u = np.zeros(self.g.u_shape)
        v = np.zeros(self.g.v_shape)
        self.ibm.apply(u, v, time=0.0)

        i_u = int(np.argmin(np.abs(self.g.xf - cx)))
        j_u = int(np.argmin(np.abs(self.g.yc - (cy + 0.1))))
        assert self.ibm.mask_u[i_u, j_u]
        assert np.isclose(u[i_u, j_u], -omega * (self.g.yc[j_u] - cy))

        i_v = int(np.argmin(np.abs(self.g.xc - (cx + 0.1))))
        j_v = int(np.argmin(np.abs(self.g.yf - cy)))
        assert self.ibm.mask_v[i_v, j_v]
        assert np.isclose(v[i_v, j_v], omega * (self.g.xc[i_v] - cx))

    def test_constant_rotating_circle_imposes_tangential_velocity(self):
        cx, cy, r = 1.0, 0.75, 0.3
        omega = 1.75
        self.ibm.add_constant_rotating_circle(cx, cy, r, omega=omega)
        u = np.zeros(self.g.u_shape)
        v = np.zeros(self.g.v_shape)
        self.ibm.apply(u, v, time=2.0)

        i_u = int(np.argmin(np.abs(self.g.xf - cx)))
        j_u = int(np.argmin(np.abs(self.g.yc - (cy + 0.1))))
        assert self.ibm.mask_u[i_u, j_u]
        assert np.isclose(u[i_u, j_u], -omega * (self.g.yc[j_u] - cy))

        i_v = int(np.argmin(np.abs(self.g.xc - (cx + 0.1))))
        j_v = int(np.argmin(np.abs(self.g.yf - cy)))
        assert self.ibm.mask_v[i_v, j_v]
        assert np.isclose(v[i_v, j_v], omega * (self.g.xc[i_v] - cx))


class TestIBMRectangle:
    def test_add_rectangle(self):
        g = CartesianGrid(8, 6, lx=1.0, ly=1.0)
        ibm = ImmersedBoundary(g)
        ibm.add_rectangle(0.2, 0.4, 0.3, 0.7)
        assert ibm.has_solid

    def test_rectangle_interior_solid(self):
        g = CartesianGrid(20, 20, lx=1.0, ly=1.0)
        ibm = ImmersedBoundary(g)
        x0, x1, y0, y1 = 0.3, 0.7, 0.3, 0.7
        ibm.add_rectangle(x0, x1, y0, y1)
        # u-face at centre of rectangle should be solid
        i_c = int(np.argmin(np.abs(g.xf - 0.5)))
        j_c = int(np.argmin(np.abs(g.yc - 0.5)))
        assert ibm.mask_u[i_c, j_c]


class TestIBMIndentedCircle:
    def test_top_indent_clears_top_center(self):
        g = CartesianGrid(80, 80, lx=2.0, ly=2.0)
        ibm = ImmersedBoundary(g)
        cx, cy, r = 1.0, 1.0, 0.5
        ibm.add_circle_with_top_indent(
            cx=cx,
            cy=cy,
            radius=r,
            indent_width=0.3,
            indent_depth=0.2,
        )

        i_center = int(np.argmin(np.abs(g.xf - cx)))
        j_notch = int(np.argmin(np.abs(g.yc - (cy + r - 0.1))))
        j_body = int(np.argmin(np.abs(g.yc - cy)))

        assert not ibm.mask_u[i_center, j_notch]
        assert ibm.mask_u[i_center, j_body]

    def test_invalid_indent_rejected(self):
        g = CartesianGrid(20, 20, lx=2.0, ly=2.0)
        ibm = ImmersedBoundary(g)
        with pytest.raises(ValueError):
            ibm.add_circle_with_top_indent(
                cx=1.0,
                cy=1.0,
                radius=0.3,
                indent_width=0.7,
                indent_depth=0.1,
            )


class TestIBMSweepingJet:
    def test_sweeping_jet_imposes_outlet_velocity(self):
        g = CartesianGrid(80, 80, lx=2.0, ly=2.0)
        ibm = ImmersedBoundary(g)
        ibm.add_sweeping_jet_circle(
            cx=1.0,
            cy=1.0,
            radius=0.5,
            jet_speed=0.4,
            slot_center_angle_deg=90.0,
            slot_width_angle_deg=20.0,
            slot_depth=0.08,
            sweep_amplitude_deg=0.0,
            frequency=0.0,
        )

        u = np.zeros(g.u_shape)
        v = np.zeros(g.v_shape)
        ibm.apply(u, v, time=0.0)

        spec = ibm.sweeping_jets[0]
        assert np.any(spec.mask_u)
        assert np.any(spec.mask_v)
        assert np.all(v[spec.mask_v] >= 0.0)
        assert np.any(v[spec.mask_v] > 0.0)

        body_only_v = ibm.mask_v & ~spec.mask_v
        assert np.allclose(v[body_only_v], 0.0)

    def test_geometry_resolved_jet_cavity_is_fluid(self):
        g = CartesianGrid(100, 100, lx=2.0, ly=2.0)
        ibm = ImmersedBoundary(g)
        ibm.add_geometry_resolved_sweeping_jet_circle(
            cx=1.0,
            cy=1.0,
            radius=0.5,
            jet_speed=0.4,
            cavity_width=0.4,
            cavity_height=0.3,
            slot_width=0.12,
            slot_height=0.08,
            feed_width=0.18,
            feed_height=0.08,
            sweep_amplitude_deg=0.0,
            frequency=0.0,
        )

        i_c = int(np.argmin(np.abs(g.xf - 1.0)))
        j_cavity = int(np.argmin(np.abs(g.yc - 1.28)))
        j_body = int(np.argmin(np.abs(g.yc - 1.0)))
        assert not ibm.mask_u[i_c, j_cavity]
        assert ibm.mask_u[i_c, j_body]
        assert len(ibm.oscillating_jet_patches) == 1
