from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plot_colour_envelope_realizations import (
    bootstrap_percentiles_with_draws,
    summarize_nested_prefixes,
)


class ColourEnvelopeRealizationsTest(unittest.TestCase):
    def test_bootstrap_draws_are_reproducible(self):
        values = np.linspace(-1.0, 2.0, 60)
        percentiles = np.array([16.0, 50.0, 84.0])

        first = bootstrap_percentiles_with_draws(
            values,
            percentiles,
            n_bootstrap=40,
            rng=np.random.default_rng(1234),
            batch_size=7,
        )
        second = bootstrap_percentiles_with_draws(
            values,
            percentiles,
            n_bootstrap=40,
            rng=np.random.default_rng(1234),
            batch_size=11,
        )

        for first_array, second_array in zip(first, second):
            np.testing.assert_allclose(first_array, second_array)
        self.assertEqual(first[-1].shape, (40, 3))

    def test_tail_support_can_reject_envelope_but_keep_median(self):
        colours = pd.DataFrame({
            "event_id": np.arange(100),
            "time_days": np.full(100, 0.5),
            "colour_mag": np.linspace(-0.5, 1.5, 100),
        })

        summary, draws = summarize_nested_prefixes(
            colours=colours,
            sample_sizes=[100],
            target_times=np.array([0.5]),
            minimum_valid_events=30,
            minimum_tail_events=20.0,
            bootstrap=False,
            n_bootstrap=0,
            bootstrap_threshold_mag=0.10,
            bootstrap_seed=12345,
            realization=1,
            pair_index=0,
        )

        row = summary.iloc[0]
        self.assertEqual(row.n_valid, 100)
        self.assertFalse(row.envelope_valid)
        self.assertTrue(row.median_valid)
        self.assertEqual(row.failure_scope, "envelope")
        self.assertEqual(row.status, "tail_support_invalid")
        self.assertTrue(draws.empty)

    def test_tail_boundary_is_inclusive_at_125_colours(self):
        colours = pd.DataFrame({
            "event_id": np.arange(125),
            "time_days": np.full(125, 0.5),
            "colour_mag": np.linspace(-0.5, 1.5, 125),
        })

        summary, _ = summarize_nested_prefixes(
            colours=colours,
            sample_sizes=[125],
            target_times=np.array([0.5]),
            minimum_valid_events=30,
            minimum_tail_events=20.0,
            bootstrap=False,
            n_bootstrap=0,
            bootstrap_threshold_mag=0.10,
            bootstrap_seed=12345,
            realization=1,
            pair_index=0,
        )

        row = summary.iloc[0]
        self.assertAlmostEqual(row.expected_tail_events_p16, 20.0)
        self.assertAlmostEqual(row.expected_tail_events_p84, 20.0)
        self.assertTrue(row.envelope_valid)
        self.assertTrue(row.median_valid)
        self.assertTrue(row.valid)

    def test_bootstrap_capture_has_one_row_per_draw_and_percentile(self):
        colours = pd.DataFrame({
            "event_id": np.arange(50),
            "time_days": np.full(50, 2.0),
            "colour_mag": np.linspace(0.0, 1.0, 50),
        })

        _, draws = summarize_nested_prefixes(
            colours=colours,
            sample_sizes=[50],
            target_times=np.array([2.0]),
            minimum_valid_events=1,
            minimum_tail_events=0.0,
            bootstrap=True,
            n_bootstrap=12,
            bootstrap_threshold_mag=10.0,
            bootstrap_seed=321,
            realization=3,
            pair_index=0,
            capture_bootstrap=True,
            capture_sample_sizes={50},
            capture_times=np.array([2.0]),
        )

        self.assertEqual(len(draws), 12 * 3)
        self.assertEqual(set(draws["percentile"]), {"p16", "p50", "p84"})
        self.assertEqual(set(draws["bootstrap_index"]), set(range(1, 13)))


if __name__ == "__main__":
    unittest.main()
