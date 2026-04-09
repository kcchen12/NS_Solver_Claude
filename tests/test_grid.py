"""Tests for CartesianGrid."""
import numpy as np
import pytest
from src.grid import (
    CartesianGrid,
    build_nonuniform_grid_metadata,
    stretched_faces_center_band,
    stretched_faces_tanh,
)
from src.io_utils import load_grid_metadata, load_grid_metadata_dict, save_grid_metadata, save_grid_metadata_dict


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


class TestPreparedNonuniformGrid:
    def test_stretched_faces_tanh_uniform_when_beta_nonpositive(self):
        xf = stretched_faces_tanh(4, 1.0, beta=0.0)
        assert np.allclose(xf, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_center_band_returns_uniform_faces_when_beta_nonpositive(self):
        xf, band_start, band_end = stretched_faces_center_band(6, 3.0, beta=0.0)
        assert np.allclose(xf, np.linspace(0.0, 3.0, 7))
        assert np.isclose(band_start, 0.0)
        assert np.isclose(band_end, 3.0)

    def test_nonuniform_metadata_has_expected_shapes(self):
        meta = build_nonuniform_grid_metadata(
            nx=8,
            ny=4,
            lx=2.0,
            ly=1.0,
            beta_x=2.0,
            beta_y=1.5,
        )
        assert meta["grid_type"] == "nonuniform"
        assert meta["xf"].shape == (9,)
        assert meta["yf"].shape == (5,)
        assert meta["dx"].shape == (8,)
        assert meta["dy"].shape == (4,)
        assert meta["nonuniform_mode"] == "center-band"
        assert np.isclose(meta["band_fraction_x"], 1.0 / 3.0)
        assert np.isclose(meta["band_fraction_y"], 1.0 / 3.0)
        assert np.isclose(meta["xf"][0], 0.0)
        assert np.isclose(meta["xf"][-1], 2.0)

    def test_nonuniform_grid_concentrates_cells_in_center_band(self):
        meta = build_nonuniform_grid_metadata(
            nx=90,
            ny=30,
            lx=9.0,
            ly=3.0,
            beta_x=2.0,
            beta_y=2.0,
            band_fraction_x=1.0 / 3.0,
            band_fraction_y=1.0 / 3.0,
        )
        xc = meta["xc"]
        dx = meta["dx"]
        band_mask = (xc >= meta["band_start_x"]) & (xc <= meta["band_end_x"])
        outer_mask = ~band_mask
        assert np.mean(dx[band_mask]) < np.mean(dx[outer_mask])

    def test_nonuniform_grid_outside_spacing_stays_close_to_uniform(self):
        meta = build_nonuniform_grid_metadata(
            nx=90,
            ny=30,
            lx=9.0,
            ly=3.0,
            beta_x=2.0,
            beta_y=2.0,
            band_fraction_x=1.0 / 3.0,
            band_fraction_y=1.0 / 3.0,
        )
        xc = meta["xc"]
        dx = meta["dx"]
        uniform_dx = 9.0 / 90.0
        outer_mask = (xc < meta["band_start_x"]) | (xc > meta["band_end_x"])
        assert np.mean(dx[outer_mask]) == pytest.approx(uniform_dx, rel=0.25)

    def test_nonuniform_metadata_round_trip_dict_loader(self, tmp_path):
        path = tmp_path / "nonuniform_grid.npz"
        original = build_nonuniform_grid_metadata(
            nx=8,
            ny=4,
            lx=2.0,
            ly=1.0,
            beta_x=2.0,
            beta_y=1.0,
        )
        save_grid_metadata_dict(str(path), original)
        loaded = load_grid_metadata_dict(str(path))
        assert loaded["grid_type"] == "nonuniform"
        assert np.allclose(loaded["xf"], original["xf"])
        assert np.allclose(loaded["dy"], original["dy"])

    def test_uniform_loader_rejects_nonuniform_metadata(self, tmp_path):
        path = tmp_path / "nonuniform_grid.npz"
        meta = build_nonuniform_grid_metadata(
            nx=8,
            ny=4,
            lx=2.0,
            ly=1.0,
            beta_x=2.0,
            beta_y=1.0,
        )
        save_grid_metadata_dict(str(path), meta)
        with pytest.raises(ValueError):
            load_grid_metadata(str(path))
