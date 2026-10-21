"""Smoke tests for generator diagnostics separated from simulation logic."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kllap.generation.plots import (
    plot_lsstlike_only,
    plot_numerical_floor_diagnostic,
    plot_saved_synthetic_measurements,
    plot_synthetic_and_lsstlike,
)


class GenerationPlotsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        self.filters = ["lsstg"]
        self.colors = {"lsstg": "#33a02c"}
        self.band_to_filter = {"g": "lsstg"}

    def tearDown(self):
        self.temporary.cleanup()

    def assert_nonempty_png(self, path: Path):
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)

    def test_saved_synthetic_plot_uses_the_final_table(self):
        output = self.output / "synthetic.png"
        table = pd.DataFrame(
            {
                "t_days": [0.5, 1.0, 2.0],
                "band": ["g", "g", "g"],
                "mag": [20.0, 20.5, 21.0],
                "photometric_system": ["lsst"] * 3,
            }
        )
        written = plot_saved_synthetic_measurements(
            table,
            self.colors,
            output,
            event_index=0,
            filter_systems=["lsst"],
            filter_by_system={"lsst": {"g": "lsstg"}},
        )
        self.assertTrue(written)
        self.assert_nonempty_png(output)

    def test_numerical_floor_plot_keeps_support_boundary_visible(self):
        output = self.output / "floor.png"
        plot_numerical_floor_diagnostic(
            times=np.array([0.5, 1.0, 2.0, 4.0]),
            raw_magnitudes_by_filter={"lsstg": [20.0, 21.0, 23.0, 25.0]},
            filters=self.filters,
            filter_colors=self.colors,
            apparent_cut_mag=23.0,
            outfile=output,
        )
        self.assert_nonempty_png(output)

    def test_lsstlike_and_overlay_plots_accept_detection_products(self):
        visits = pd.DataFrame(
            {
                "t_days": [0.5, 1.5],
                "band": ["g", "g"],
                "mag_true": [20.0, 21.0],
                "detected": [True, False],
                "mag_obs": [20.1, np.nan],
                "mag_err": [0.05, np.nan],
                "mag_ulim": [np.nan, 24.0],
            }
        )
        lsstlike_output = self.output / "lsstlike.png"
        plot_lsstlike_only(
            visits,
            self.filters,
            self.colors,
            event_index=0,
            outfile=lsstlike_output,
            band_to_filter=self.band_to_filter,
        )
        self.assert_nonempty_png(lsstlike_output)

        overlay_output = self.output / "overlay.png"
        plot_synthetic_and_lsstlike(
            times=np.array([0.5, 1.0, 2.0]),
            mag_dict={"lsstg": [20.0, 20.5, 21.5]},
            observations=visits,
            filters=self.filters,
            filter_colors=self.colors,
            mag_lim=25.0,
            outfile=overlay_output,
            band_to_filter=self.band_to_filter,
        )
        self.assert_nonempty_png(overlay_output)


if __name__ == "__main__":
    unittest.main()
