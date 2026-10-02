from __future__ import annotations

import unittest

import pandas as pd

from analyze_envelope_population_convergence import (
    aggregate_results,
    parse_sample_sizes,
)
from build_color_envelopes import select_metadata_subset


class EnvelopePopulationConvergenceTest(unittest.TestCase):
    def test_sample_size_parser_sorts_and_deduplicates(self) -> None:
        self.assertEqual(
            parse_sample_sizes("10_000, 1000,10000, 100000"),
            [1000, 10000, 100000],
        )

    def test_fixed_seed_produces_nested_subsets(self) -> None:
        metadata = pd.DataFrame({"event_id": [str(i) for i in range(20)]})
        small = select_metadata_subset(metadata, 5, 31415)
        large = select_metadata_subset(metadata, 12, 31415)
        repeated = select_metadata_subset(metadata, 5, 31415)
        self.assertTrue(set(small["event_id"]).issubset(set(large["event_id"])))
        self.assertEqual(list(small["event_id"]), list(repeated["event_id"]))

    def test_threshold_requires_all_larger_sizes_to_pass(self) -> None:
        rows = []
        times = {
            100: [8.0, 8.4, 8.0],
            1000: [9.6, 9.6, 10.0],
            10000: [10.0, 10.0, 10.4],
            100000: [10.0, 10.0, 10.0],
        }
        for sample_size, values in times.items():
            for repetition, value in enumerate(values):
                rows.append(
                    {
                        "N_requested": sample_size,
                        "N_input_events": sample_size,
                        "repetition": repetition,
                        "last_valid_time_days": value,
                        "last_low_percentile": 2.0,
                        "N_train_last_bin": sample_size * 0.7,
                    }
                )
        raw, summary, reference, threshold = aggregate_results(
            pd.DataFrame(rows),
            tolerance=0.4,
            required_fraction=0.9,
        )
        self.assertAlmostEqual(reference, 10.0)
        self.assertEqual(threshold, 1000)
        self.assertFalse(
            bool(summary.loc[summary["N_requested"].eq(100), "point_passes_stability"].iloc[0])
        )
        self.assertTrue(raw["reference_time_days"].eq(10.0).all())


if __name__ == "__main__":
    unittest.main()
