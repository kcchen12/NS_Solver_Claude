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
