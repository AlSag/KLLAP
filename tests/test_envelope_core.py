from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from kne_envelope_core import (
    percentile_diagnostics,
    select_metadata_subset,
)


class EnvelopeCoreTest(unittest.TestCase):
    def test_fixed_seed_subsets_are_nested_and_reproducible(self) -> None:
        metadata = pd.DataFrame({"event_id": [str(index) for index in range(100)]})
        small = select_metadata_subset(metadata, 10, 2026)
        large = select_metadata_subset(metadata, 40, 2026)
        repeated = select_metadata_subset(metadata, 10, 2026)
        self.assertTrue(set(small.event_id).issubset(set(large.event_id)))
        self.assertEqual(small.event_id.tolist(), repeated.event_id.tolist())

    def test_requested_interval_includes_sparse_bins(self):
        diagnostics, pair = percentile_diagnostics(np.array([1., 3., np.nan]), [1.])
        self.assertEqual(pair["low_percentile"], 1.)
        self.assertAlmostEqual(pair["color_min"], 1.02)
        self.assertAlmostEqual(pair["color_max"], 2.98)
        _, single = percentile_diagnostics(np.array([4.]), [1.])
        self.assertEqual(single["color_min"], single["color_max"])
        self.assertIsNone(percentile_diagnostics(np.array([np.nan]), [1.])[1])
