"""Tests for CartesianGrid."""
import numpy as np
import pytest
from src.grid import CartesianGrid
from src.io_utils import load_grid_metadata, save_grid_metadata


class TestCartesianGrid2D:
    def test_cell_count(self):
        g = CartesianGrid(8, 4, lx=2.0, ly=1.0)
        assert g.nx == 8
        assert g.ny == 4
        assert not g.is_3d

    def test_spacing(self):
        g = CartesianGrid(8, 4, lx=2.0, ly=1.0)
        assert np.isclose(g.dx, 0.25)
        assert np.isclose(g.dy, 0.25)

    def test_face_coordinates(self):
        g = CartesianGrid(4, 2, lx=1.0, ly=1.0)
        # x-faces: 0, 0.25, 0.5, 0.75, 1.0
        assert np.allclose(g.xf, [0.0, 0.25, 0.5, 0.75, 1.0])
        # y-faces: 0, 0.5, 1.0
        assert np.allclose(g.yf, [0.0, 0.5, 1.0])

    def test_cell_centre_coordinates(self):
        g = CartesianGrid(4, 2, lx=1.0, ly=1.0)
        assert np.allclose(g.xc, [0.125, 0.375, 0.625, 0.875])
        assert np.allclose(g.yc, [0.25, 0.75])

    def test_p_shape(self):
        g = CartesianGrid(8, 4)
        assert g.p_shape == (8, 4)

    def test_u_shape(self):
        g = CartesianGrid(8, 4)
        assert g.u_shape == (9, 4)

    def test_v_shape(self):
        g = CartesianGrid(8, 4)
        assert g.v_shape == (8, 5)

    def test_zeros_u(self):
        g = CartesianGrid(4, 3)
        u = g.zeros_u()
        assert u.shape == (5, 3)
        assert np.all(u == 0.0)

    def test_zeros_v(self):
        g = CartesianGrid(4, 3)
        v = g.zeros_v()
        assert v.shape == (4, 4)
        assert np.all(v == 0.0)

    def test_zeros_p(self):
        g = CartesianGrid(4, 3)
        p = g.zeros_p()
        assert p.shape == (4, 3)

    def test_metadata_contains_coordinates(self):
        g = CartesianGrid(4, 2, lx=1.0, ly=1.0)
        meta = g.to_metadata()
        assert meta["nx"] == 4
        assert meta["ny"] == 2
        assert np.allclose(meta["xf"], [0.0, 0.25, 0.5, 0.75, 1.0])
        assert np.allclose(meta["yc"], [0.25, 0.75])

    def test_grid_metadata_round_trip(self, tmp_path):
        path = tmp_path / "uniform_grid.npz"
        original = CartesianGrid(4, 2, lx=1.0, ly=1.0)
        save_grid_metadata(str(path), original)
        loaded = load_grid_metadata(str(path))
        assert loaded.nx == original.nx
        assert loaded.ny == original.ny
        assert np.isclose(loaded.lx, original.lx)
        assert np.isclose(loaded.ly, original.ly)
        assert np.allclose(loaded.xf, original.xf)
        assert np.allclose(loaded.yc, original.yc)


class TestCartesianGrid3D:
    def test_3d_flag(self):
        g = CartesianGrid(4, 3, nz=2)
        assert g.is_3d

    def test_3d_shapes(self):
        g = CartesianGrid(4, 3, nz=2)
        assert g.p_shape == (4, 3, 2)
        assert g.u_shape == (5, 3, 2)
        assert g.v_shape == (4, 4, 2)
        assert g.w_shape == (4, 3, 3)

    def test_3d_w_raises_in_2d(self):
        g = CartesianGrid(4, 3)
        with pytest.raises(RuntimeError):
            g.zeros_w()
