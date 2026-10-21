import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analyze_envelope_population_convergence_fast import (
    aggregate_colour_results, colour_errors, compare_profiles,
    envelope_profile, repetition_record, safe_quantile, save_colour_plot,
)


def profile(values, valid=None):
    values = np.asarray(values, float)
    n = len(values)
    return pd.DataFrame({
        "bin_index": np.arange(n), "t_low_days": .4 + np.arange(n) * .4,
        "t_high_days": .8 + np.arange(n) * .4, "t_center_days": .6 + np.arange(n) * .4,
        "valid": np.ones(n, bool) if valid is None else valid,
        "N_train": np.full(n, 100), "low_percentile": np.full(n, 16.),
        "high_percentile": np.full(n, 84.), "color_min": values[:, 0],
        "color_median": values[:, 1], "color_max": values[:, 2]})


class ColourConvergenceTest(unittest.TestCase):
    def test_signed_colours_and_exact_ten_percent_boundary(self):
        _, relative, passes = colour_errors(
            [-.9, 1.1, -1.11], [-1., 1., -1.],
            tolerance_mode="relative", relative_tolerance=.1,
        )
        np.testing.assert_allclose(relative, [.1, .1, .11])
        self.assertEqual(passes.tolist(), [True, True, False])

    def test_zero_reference_and_optional_absolute_floor(self):
        _, relative, passes = colour_errors(
            [0., .001, np.nan], [0., 0., 0.], tolerance_mode="relative"
        )
        self.assertEqual(passes.tolist(), [True, False, False])
        self.assertEqual(relative[0], 0.)
        self.assertTrue(np.isinf(relative[1:]).all())
        _, _, passes = colour_errors(
            [.001, np.nan], [0., 0.], tolerance_mode="absolute",
            absolute_tolerance_mag=.01,
        )
        self.assertEqual(passes.tolist(), [True, False])

    def test_absolute_magnitude_tolerance_is_independent_of_reference_value(self):
        difference, relative, passes = colour_errors(
            [.18, -.02, .27], [.16, 0., .16],
            tolerance_mode="absolute", absolute_tolerance_mag=.10,
        )
        np.testing.assert_allclose(difference, [.02, .02, .11])
        self.assertGreater(relative[0], .10)
        self.assertTrue(np.isinf(relative[1]))
        self.assertEqual(passes.tolist(), [True, True, False])

    def test_one_bad_bound_in_one_bin_fails_even_with_matching_last_bin(self):
        ref = profile([[1., 2., 3.], [2., 3., 4.]])
        sample = profile([[1., 2., 3.], [2., 3., 4.5]])
        comparison = compare_profiles(sample, ref)
        record = repetition_record(comparison, 0, 100, 1)
        self.assertFalse(record["all_reference_bins_pass"])
        self.assertTrue(record["all_bins_pass_color_median"])
        self.assertAlmostEqual(record["max_relative_error_all"], .125)

    def test_missing_reference_bin_is_not_silently_removed(self):
        ref = profile([[1., 2., 3.], [2., 3., 4.]])
        comparison = compare_profiles(ref.iloc[:1], ref)
        self.assertEqual(len(comparison), 2)
        self.assertFalse(comparison.all_three_pass.iloc[1])
        self.assertTrue(np.isinf(comparison.relative_error_color_min.iloc[1]))

    def test_invalid_sample_bin_fails_even_with_finite_matching_colours(self):
        ref = profile([[1., 2., 3.]])
        candidate = profile([[1., 2., 3.]], [False])
        self.assertFalse(compare_profiles(candidate, ref).all_three_pass.iloc[0])

    def test_reference_invalid_bins_do_not_enter_comparison(self):
        ref = profile([[1., 2., 3.], [np.nan, np.nan, np.nan]], [True, False])
        self.assertEqual(len(compare_profiles(ref, ref)), 1)

    def test_all_larger_sizes_must_pass_required_repetition_fraction(self):
        rows = []
        for size, successes in ((100, 10), (200, 8), (300, 9), (400, 10)):
            for rep in range(10):
                row = {"N_requested": size, "all_reference_bins_pass": rep < successes,
                       "N_supported_reference_bins": 2}
                row.update({f"max_relative_error_{name}": .05 for name in
                            ("color_min", "color_median", "color_max", "all")})
                row.update({f"max_absolute_error_{name}_mag": .02 for name in
                            ("color_min", "color_median", "color_max", "all")})
                row["max_metric_error_all"] = .02
                rows.append(row)
        summary, threshold = aggregate_colour_results(pd.DataFrame(rows), .9)
        self.assertEqual(threshold, 300)
        self.assertFalse(summary.all_larger_sizes_also_stable.iloc[0])

    def test_no_reference_bins_is_an_error(self):
        ref = profile([[1., 2., 3.]], [False])
        with self.assertRaises(ValueError):
            compare_profiles(ref, ref)

    def test_infinite_errors_have_infinite_quantiles_and_render(self):
        self.assertTrue(np.isinf(safe_quantile([0., np.inf], .5)))
        ref = profile([[1., 2., 3.]])
        rows = [repetition_record(compare_profiles(ref.iloc[:0], ref), 0, 10, 1),
                repetition_record(compare_profiles(ref, ref), 0, 100, 1)]
        raw = pd.DataFrame(rows)
        summary, threshold = aggregate_colour_results(raw)
        a = SimpleNamespace(tolerance_mode="absolute", relative_tolerance=.1,
                            absolute_tolerance_mag=.1, required_fraction=.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.png"
            save_colour_plot(raw, summary, a, threshold, path)
            self.assertGreater(path.stat().st_size, 1000)

    def test_real_builder_profiles_adaptive_and_reference_percentiles(self):
        n = 120
        sample = pd.DataFrame({"event_id": [str(i) for i in range(n)]})
        colors = np.column_stack((np.linspace(1., 2., n), np.linspace(2., 3., n))).astype(np.float32)
        arrays = {"colour_intrinsic": colors, "colour_extincted": colors}
        a = SimpleNamespace(train_fraction=.7, validation_fraction=.15, builder_seed=12345,
                            stage="no_m5_with_mw", tail_percentiles="16", n_boot=20,
                            bootstrap_threshold_mag=100., minimum_tail_events=1.,
                            minimum_events=5, allow_degenerate_envelope=False, bound_mode="adaptive")
        edges = np.array([.4, .8, 1.2])
        centres = (edges[:-1] + edges[1:]) / 2
        ref = envelope_profile(a, sample, arrays, np.arange(n), {}, ("g", "r"), edges, centres)
        self.assertTrue(ref.valid.all())
        a.bound_mode = "reference-percentiles"
        other = envelope_profile(a, sample, arrays, np.arange(n), {}, ("g", "r"), edges, centres, ref)
        self.assertTrue(compare_profiles(other, ref).all_three_pass.all())

    def test_degenerate_reference_is_excluded_unless_requested(self):
        n = 60
        sample = pd.DataFrame({"event_id": [str(i) for i in range(n)]})
        values = np.linspace(1., 2., n).reshape(-1, 1).astype(np.float32)
        a = SimpleNamespace(train_fraction=.7, validation_fraction=.15, builder_seed=12345,
                            stage="no_m5_with_mw", tail_percentiles="50", n_boot=20,
                            bootstrap_threshold_mag=100., minimum_tail_events=21.,
                            minimum_events=5, allow_degenerate_envelope=False, bound_mode="adaptive")
        arrays = {"colour_intrinsic": values, "colour_extincted": values}
        args = (a, sample, arrays, np.arange(n), {}, ("g", "r"), np.array([.4, .8]), np.array([.6]))
        ref = envelope_profile(*args)
        self.assertFalse(ref.valid.iloc[0])
        self.assertEqual(ref.status.iloc[0], "degenerate_p50_p50_excluded")
        a.allow_degenerate_envelope = True
        self.assertTrue(envelope_profile(*args).valid.iloc[0])


if __name__ == "__main__":
    unittest.main()
