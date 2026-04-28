"""Tests for post-processing helper scripts."""

import numpy as np

from analyze_aerodynamics import _read_config
from view_snapshot_viewer import (
    _compute_snapshot_vorticity,
    pick_slice_and_component,
    plot_vorticity_video,
)


class TestAnalyzeAerodynamicsHelpers:
    def test_read_config_parses_bool_and_float(self, tmp_path):
        config_path = tmp_path / "config.txt"
        config_path.write_text(
            "cylinder = true\n"
            "cylinder_radius = 0.25\n"
            "note = ignored\n"
            "re = 100  # inline comment\n",
            encoding="utf-8",
        )
        config = _read_config(str(config_path))
        assert config["cylinder"] == 1.0
        assert config["cylinder_radius"] == 0.25
        assert config["re"] == 100.0
        assert "note" not in config


class TestSnapshotViewerHelpers:
    def test_pick_slice_and_component_for_component_last(self):
        arr = np.arange(3 * 4 * 2).reshape(3, 4, 2)
        picked = pick_slice_and_component(arr, slice_idx=None, comp_idx=1)
        assert np.array_equal(picked, arr[:, :, 1])

    def test_pick_slice_and_component_for_3d_slice(self):
        arr = np.arange(5 * 3 * 6).reshape(5, 3, 6)
        picked = pick_slice_and_component(arr, slice_idx=2, comp_idx=None)
        assert np.array_equal(picked, arr[2, :, :])

    def test_compute_snapshot_vorticity_for_uniform_flow(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()

        nx, ny = 4, 3
        u = np.ones((nx + 1, ny), dtype=float)
        v = np.zeros((nx, ny + 1), dtype=float)
        np.savez(outdir / "snap_000.0000.npz", u=u, v=v, t=0.0)

        xc, yc, omega = _compute_snapshot_vorticity(str(outdir / "snap_000.0000.npz"))
        assert xc.shape == (nx,)
        assert yc.shape == (ny,)
        assert omega.shape == (nx, ny)
        assert np.allclose(omega, 0.0)

    def test_plot_vorticity_video_writes_gif(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"

        nx, ny = 4, 3
        u0 = np.ones((nx + 1, ny), dtype=float)
        v0 = np.zeros((nx, ny + 1), dtype=float)
        u1 = np.ones((nx + 1, ny), dtype=float)
        v1 = np.zeros((nx, ny + 1), dtype=float)
        v1[:, :] = np.linspace(0.0, 1.0, nx)[:, np.newaxis]

        np.savez(outdir / "snap_000.0000.npz", u=u0, v=v0, t=0.0)
        np.savez(outdir / "snap_000.1000.npz", u=u1, v=v1, t=0.1)

        plot_vorticity_video(
            snapshot_dir=str(outdir),
            save_name="test_vorticity.gif",
            fps=2,
            results_dir=str(results_dir),
        )

        assert (results_dir / "test_vorticity.gif").exists()

    def test_plot_vorticity_video_accepts_frame_stride(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"

        nx, ny = 4, 3
        for idx in range(5):
            u = np.ones((nx + 1, ny), dtype=float)
            v = np.zeros((nx, ny + 1), dtype=float)
            v[:, :] = float(idx) * np.linspace(0.0, 1.0, nx)[:, np.newaxis]
            np.savez(outdir / f"snap_{idx:03d}.0000.npz", u=u, v=v, t=float(idx))

        plot_vorticity_video(
            snapshot_dir=str(outdir),
            save_name="test_vorticity_stride.gif",
            fps=2,
            frame_stride=2,
            results_dir=str(results_dir),
        )

        assert (results_dir / "test_vorticity_stride.gif").exists()
