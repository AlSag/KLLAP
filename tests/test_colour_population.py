from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from types import SimpleNamespace

from plot_kne_colour_isocontours import (
    axis_domain,
    parameter_values,
    parse_parameter_pairs,
    read_bounds,
    load_event_colours,
)


class ColourPopulationTest(unittest.TestCase):
    @staticmethod
    def make_population(root: Path) -> pd.DataFrame:
        (root / "csv").mkdir()
        metadata = pd.DataFrame({
            "event_id": ["0", "1"],
            "ebv_mw": [0.1, 0.1],
            "redshift": [0.1, 0.1],
        })
        metadata.to_csv(root / "summary.csv", index=False)
        for index, g_magnitude in enumerate((20.0, 20.5)):
            pd.DataFrame({
                "t_days": [0.5, 1.0, 0.5, 1.0],
                "band": ["g", "g", "r", "r"],
                "mag": [g_magnitude, g_magnitude, 20.0, 20.0],
            }).to_csv(root / "csv" / f"lightcurve_{index:06d}.csv", index=False)
        return metadata

    def test_parameter_pairs_and_transforms(self):
        self.assertEqual(
            parse_parameter_pairs("Ye_wind:log10_mej_dyn"),
            [("Ye_wind", "log10_mej_dyn")],
        )
        metadata = pd.DataFrame({
            "log10_mej_dyn": [-3.0, -2.0],
            "inclination_EM": [0.0, np.pi / 2.0],
        })
        np.testing.assert_allclose(parameter_values(metadata, "mej_dyn"), [.001, .01])
        np.testing.assert_allclose(parameter_values(metadata, "inclination_deg"), [0, 90])
        self.assertEqual(axis_domain(np.array([np.nan, -3.0, -2.0])), (-3.0, -2.0))

    def test_bounds_use_containing_bin_without_extrapolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounds.csv"
            pd.DataFrame({
                "stage": ["m5_no_mw", "m5_no_mw"],
                "t_min_days": [0.4, 1.0],
                "t_max_days": [1.0, 2.0],
                "color_min": [0.0, 0.5],
                "color_max": [1.0, 1.5],
            }).to_csv(path, index=False)
            bounds = read_bounds(path, np.array([1.0, 2.0, 3.0]), "m5_no_mw")
        np.testing.assert_allclose(bounds.color_min[:2], [0.5, 0.5])
        self.assertEqual(bounds.bounds_status.iloc[2], "missing_or_invalid")

    def test_colour_loading_preserves_mw_and_depth_conventions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.make_population(root)
            args = SimpleNamespace(
                photometric_system="lsst", stage="m5_with_mw",
                time_frame="observer", color_pair="g-r",
            )
            colour, supported, available = load_event_colours(
                args, root, metadata, ("g", "r"), np.array([1.0]),
                {"g": 20.3, "r": 21.0},
            )
        np.testing.assert_allclose(colour[:, 0], [0.0, 0.5])
        self.assertTrue(supported.all())
        np.testing.assert_array_equal(available[:, 0], [True, False])


if __name__ == "__main__":
    unittest.main()
