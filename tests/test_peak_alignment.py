from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from build_color_envelopes import (
    estimate_interior_peak_time,
    interpolate_band_linear,
    interpolate_saved_population,
    time_axis_label,
)


class PeakAlignmentTest(unittest.TestCase):
    def test_later_interior_peak_wins_over_bright_first_model_point(self) -> None:
        estimate = estimate_interior_peak_time(
            np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
            np.array([10.0, 10.5, 11.0, 10.6, 10.2, 10.7]),
            minimum_prominence_mag=0.05,
        )
        self.assertTrue(estimate.valid)
        self.assertTrue(estimate.first_point_is_brightest)
        self.assertAlmostEqual(estimate.discrete_time_days, 1.0)
        self.assertGreater(estimate.time_days, 0.8)
        self.assertLess(estimate.time_days, 1.2)

    def test_boundary_only_peak_is_unresolved(self) -> None:
        estimate = estimate_interior_peak_time(
            np.array([0.2, 0.4, 0.6, 0.8]),
            np.array([10.0, 10.4, 10.8, 11.0]),
            minimum_prominence_mag=0.05,
        )
        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.status, "no_interior_peak")
        self.assertTrue(estimate.first_point_is_brightest)

    def test_small_local_minimum_can_fail_prominence_threshold(self) -> None:
        estimate = estimate_interior_peak_time(
            np.arange(5, dtype=float),
            np.array([11.0, 10.99, 11.0, 10.0, 9.0]),
            minimum_prominence_mag=0.05,
        )
        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.status, "below_minimum_prominence")

    def test_peak_phase_interpolation_keeps_negative_and_zero_phases(self) -> None:
        curve = pd.DataFrame({
            "t_days": [-1.0, 0.0, 1.0],
            "band": ["g", "g", "g"],
            "magnitude": [21.0, 20.0, 22.0],
        })
        result = interpolate_band_linear(
            curve, "g", np.array([-0.5, 0.0, 0.5])
        )
        np.testing.assert_allclose(result, [20.5, 20.0, 21.0])

    def test_peak_axis_names_selected_band(self) -> None:
        self.assertEqual(
            time_axis_label("band_peak", "z"),
            r"$t-t_{\rm peak,z}$ [days]",
        )

    def test_unresolved_event_is_excluded_from_peak_relative_population(self) -> None:
        class Source:
            @staticmethod
            def iter_events(event_ids, bands, systems):
                late_peak = [
                    {"t_days": t, "band": band, "mag": magnitude,
                     "photometric_system": "lsst"}
                    for t, g, r in zip(
                        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                        [10.0, 10.5, 11.0, 10.6, 10.2, 10.7],
                        [10.4, 10.5, 10.6, 10.7, 10.8, 10.9],
                    )
                    for band, magnitude in (("g", g), ("r", r))
                ]
                boundary_only = [
                    {"t_days": t, "band": band, "mag": magnitude,
                     "photometric_system": "lsst"}
                    for t, g, r in zip(
                        [0.2, 0.4, 0.6, 0.8],
                        [10.0, 10.4, 10.8, 11.0],
                        [10.2, 10.3, 10.4, 10.5],
                    )
                    for band, magnitude in (("g", g), ("r", r))
                ]
                yield "0", pd.DataFrame(late_peak)
                yield "1", pd.DataFrame(boundary_only)

        metadata = pd.DataFrame({
            "event_id": ["0", "1"],
            "ebv_mw": [0.0, 0.0],
        })
        population = interpolate_saved_population(
            Source(), metadata, "g", "r", np.array([-0.1, 0.0, 0.1]),
            -0.2, 0.2, "lsst", "band_peak", "g", 0.05,
        )
        self.assertTrue(np.isfinite(population.intrinsic_band1[0]).all())
        self.assertTrue(np.isnan(population.intrinsic_band1[1]).all())
        diagnostics = population.peak_diagnostics.set_index("event_id")
        self.assertTrue(diagnostics.loc["0", "used_in_peak_relative_envelope"])
        self.assertFalse(diagnostics.loc["1", "used_in_peak_relative_envelope"])


if __name__ == "__main__":
    unittest.main()
