from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from kne_envelope_core import (
    RUBIN_MW_R,
    event_extinction_mag,
    percentile_diagnostics,
    select_metadata_subset,
)


class EnvelopeCoreTest(unittest.TestCase):
    def test_stored_extinction_overrides_coefficient_fallback(self) -> None:
        stored = pd.Series({"ebv_mw": 0.1, "A_lsst_g_mw": 0.72})
        fallback = pd.Series({"ebv_mw": 0.1})
        self.assertAlmostEqual(event_extinction_mag(stored, "lsst", "g"), 0.72)
        self.assertAlmostEqual(
            event_extinction_mag(fallback, "lsst", "g"),
            0.1 * RUBIN_MW_R["g"],
        )

    def test_fixed_seed_subsets_are_nested_and_reproducible(self) -> None:
        metadata = pd.DataFrame({"event_id": [str(index) for index in range(100)]})
        small = select_metadata_subset(metadata, 10, 2026)
        large = select_metadata_subset(metadata, 40, 2026)
        repeated = select_metadata_subset(metadata, 10, 2026)
        self.assertTrue(set(small.event_id).issubset(set(large.event_id)))
        self.assertEqual(small.event_id.tolist(), repeated.event_id.tolist())

    def test_tail_support_selects_widest_valid_symmetric_pair(self) -> None:
        values = np.linspace(-1.0, 1.0, 1000)
        diagnostics, selected = percentile_diagnostics(
            values,
            [16.0, 25.0, 50.0],
            n_boot=100,
            bootstrap_threshold=10.0,
            minimum_tail_events=200.0,
            rng=np.random.default_rng(7),
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["low_percentile"], 25.0)
        self.assertEqual(selected["high_percentile"], 75.0)
        p16 = diagnostics.set_index("percentile").loc[16.0]
        self.assertFalse(bool(p16["tail_support_valid"]))
        self.assertTrue(bool(p16["bootstrap_width_valid"]))

    def test_tail_boundary_is_inclusive(self) -> None:
        values = np.linspace(-1.0, 1.0, 125)
        diagnostics, selected = percentile_diagnostics(
            values,
            [16.0, 50.0],
            n_boot=50,
            bootstrap_threshold=10.0,
            minimum_tail_events=20.0,
            rng=np.random.default_rng(11),
        )
        p16 = diagnostics.set_index("percentile").loc[16.0]
        self.assertEqual(float(p16["expected_nearest_tail_events"]), 20.0)
        self.assertTrue(bool(p16["tail_support_valid"]))
        self.assertEqual(selected["low_percentile"], 16.0)


if __name__ == "__main__":
    unittest.main()
