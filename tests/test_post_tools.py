"""Tests for post-processing helper scripts."""

import numpy as np

from analyze_aerodynamics import _read_config
from view_snapshot_viewer import pick_slice_and_component


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
