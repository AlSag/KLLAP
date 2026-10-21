from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analyze_envelope_population_convergence_fast import (
    FIXED_FAMILIES,
    FIXED_PERCENTILES,
    aggregate_extent,
    aggregate_shape,
    compare_shape_group,
    extent_values_table,
    last_consecutive_edge,
    multiquantile_profile,
    save_extent_values_table,
    save_shape_values_table,
    shape_values_table,
    shape_repetition_record,
    threshold_table,
)


def make_profile(n_bins=3, invalid=None, value_shift=None):
    invalid = set(invalid or ())
    value_shift = value_shift or {}
    rows = []
    fixed = {"p5": 5., "p16": 16., "p25": 25., "p50": 50.,
             "p75": 75., "p84": 84., "p95": 95.}
    for b in range(n_bins):
        for name, q in fixed.items():
            rows.append({
                "bin_index": b,
                "t_low_days": .4 + .4 * b,
                "t_high_days": .8 + .4 * b,
                "t_center_days": .6 + .4 * b,
                "estimator_id": name,
                "percentile": q,
                "value": q / 100. + b + value_shift.get((b, name), 0.),
                "valid": (b, name) not in invalid,
            })
    return pd.DataFrame(rows)


class MultiQuantileConvergenceTest(unittest.TestCase):
    def test_requested_fixed_percentiles_and_families_are_present(self):
        self.assertEqual(
            FIXED_PERCENTILES,
            (5., 16., 25., 50., 75., 84., 95.),
        )
        self.assertEqual(FIXED_FAMILIES["central_90"],
                         ("p5", "p50", "p95"))
        self.assertEqual(FIXED_FAMILIES["central_68"],
                         ("p16", "p50", "p84"))
        self.assertEqual(FIXED_FAMILIES["central_50"],
                         ("p25", "p50", "p75"))

    def test_extent_is_separate_for_each_percentile_family(self):
        profile = make_profile(invalid={(2, "p5")})
        edge90, bins90 = last_consecutive_edge(
            profile, FIXED_FAMILIES["central_90"]
        )
        edge68, bins68 = last_consecutive_edge(
            profile, FIXED_FAMILIES["central_68"]
        )
        self.assertAlmostEqual(edge90, 1.2)
        self.assertEqual(bins90, 2)
        self.assertAlmostEqual(edge68, 1.6)
        self.assertEqual(bins68, 3)

    def test_shape_requires_every_value_and_reports_missing_support(self):
        reference = make_profile(n_bins=2)
        candidate = make_profile(
            n_bins=2,
            invalid={(1, "p95")},
            value_shift={(0, "p50"): .05},
        )
        comparison = compare_shape_group(
            candidate, reference, "central_90",
            FIXED_FAMILIES["central_90"], shape_t_max=1.2,
            absolute_tolerance_mag=.10,
        )
        record = shape_repetition_record(comparison, 0, 100, 123)
        self.assertEqual(record["N_values_required"], 6)
        self.assertEqual(record["N_values_supported"], 5)
        self.assertFalse(record["all_shape_values_pass"])
        self.assertAlmostEqual(record["max_absolute_error_supported_mag"], .05)

    def test_profile_uses_all_selected_events_without_split(self):
        n_events = 100
        sample = pd.DataFrame({
            "event_id": [str(index) for index in range(n_events)]
        })
        values = np.linspace(-1., 1., n_events, dtype=np.float32)[:, None]
        arrays = {
            "colour_intrinsic": values.copy(),
            "colour_extincted": values.copy(),
        }
        args = SimpleNamespace(
            stage="intrinsic_no_m5_no_mw",
            builder_seed=123,
            n_boot=20,
            bootstrap_threshold_mag=10.,
            minimum_tail_events=1.,
            minimum_events=5,
            # These deliberately extreme values must have no effect.
            train_fraction=.01,
            validation_fraction=.98,
        )
        profile = multiquantile_profile(
            args, sample, arrays, np.arange(n_events), {}, ("g", "r"),
            np.array([.4, .8]), np.array([.6]),
        )
        self.assertEqual(set(profile.N_available), {n_events})
        self.assertEqual(set(profile.estimator_id), {
            "p5", "p16", "p25", "p50", "p75", "p84", "p95",
        })
        self.assertFalse(any(profile.estimator_kind.eq("adaptive")))

    def test_threshold_requires_required_fraction_and_all_larger_sizes(self):
        shape_rows = []
        extent_rows = []
        successes = {100: 10, 200: 8, 300: 9, 400: 10}
        for size, count in successes.items():
            for rep in range(10):
                shape_rows.append({
                    "group_id": "central_68", "N_requested": size,
                    "all_shape_values_pass": rep < count,
                    "N_sample_supported_shape_bins": 3,
                    "N_values_supported": 9,
                    "max_absolute_error_supported_mag": .05,
                    "max_relative_error_supported": .05,
                    "max_metric_error_supported": .05,
                })
                extent_rows.append({
                    "group_id": "central_68", "N_requested": size,
                    "extent_within_tolerance": rep < count,
                    "last_valid_time_days": 4.,
                    "reference_last_valid_time_days": 4.,
                    "absolute_extent_error_days": 0.,
                })
        _, shape_thresholds = aggregate_shape(pd.DataFrame(shape_rows), .9)
        _, extent_thresholds = aggregate_extent(pd.DataFrame(extent_rows), .9)
        self.assertEqual(shape_thresholds["central_68"], 300)
        self.assertEqual(extent_thresholds["central_68"], 300)
        table = threshold_table(shape_thresholds, extent_thresholds, 400)
        row = table.loc[table.group_id.eq("central_68")].iloc[0]
        self.assertEqual(row.N_recommended_both, 300)
        self.assertTrue(row.primary_scientific_declaration)
        self.assertTrue(row.both_plateau_below_reference)

    def test_extent_png_table_contains_main_families_and_plateau_rule(self):
        rows = []
        for group_id in ("central_90", "central_68"):
            for size, stable in ((100, False), (200, True), (300, True)):
                rows.append({
                    "group_id": group_id,
                    "N_requested": size,
                    "last_valid_time_days_median": 4.0 + size / 1000,
                    "last_valid_time_days_p16": 3.8 + size / 1000,
                    "last_valid_time_days_p84": 4.2 + size / 1000,
                    "fraction_extent_within_tolerance": .8 if not stable else 1.,
                    "all_larger_sizes_also_stable": stable,
                })
        summary = pd.DataFrame(rows)
        compact = extent_values_table(summary)
        self.assertEqual(len(compact), 3)
        self.assertIn("p5-p95\nDernier bord valide (j)", compact)
        self.assertEqual(
            compact.iloc[1]["p16-p84\nPlateau atteint"], "Yes"
        )
        self.assertEqual(
            compact.iloc[2]["p16-p84\nPlateau atteint"], "Ref."
        )
        args = SimpleNamespace(tolerance_days=.4, required_fraction=.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.png"
            save_extent_values_table(summary, args, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)

    def test_extent_summary_uses_observed_grid_values(self):
        rows = []
        values = [6.0] * 8 + [6.4] * 2
        for repetition, edge in enumerate(values):
            rows.append({
                "group_id": "central_68", "N_requested": 900,
                "extent_within_tolerance": True,
                "last_valid_time_days": edge,
                "reference_last_valid_time_days": 6.4,
                "absolute_extent_error_days": abs(edge - 6.4),
            })
        summary, _ = aggregate_extent(pd.DataFrame(rows), .9)
        result = summary.iloc[0]
        self.assertEqual(result.last_valid_time_days_median, 6.0)
        self.assertEqual(result.last_valid_time_days_p16, 6.0)
        self.assertEqual(result.last_valid_time_days_p84, 6.4)

    def test_shape_png_table_is_title_free_data_product(self):
        rows = []
        for group_id in ("central_90", "central_68"):
            for size, stable in ((100, False), (200, True), (300, True)):
                rows.append({
                    "group_id": group_id, "N_requested": size,
                    "max_absolute_error_supported_mag_median": .05,
                    "max_absolute_error_supported_mag_p16": .03,
                    "max_absolute_error_supported_mag_p84": .08,
                    "fraction_all_shape_values_pass": 1. if stable else .8,
                    "all_larger_sizes_also_stable": stable,
                })
        summary = pd.DataFrame(rows)
        compact = shape_values_table(summary)
        self.assertIn("p5-p95\nMax |ΔC| (mag)", compact)
        self.assertEqual(
            compact.iloc[1]["p16-p84\nPlateau atteint"], "Yes"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shape_table.png"
            save_shape_values_table(summary, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

