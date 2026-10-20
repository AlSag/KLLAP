from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kne_m5 import (
    M5_SCENARIO_LABELS,
    M5_SCENARIO_PERCENTILES,
    M5_SCENARIOS,
    depths_from_table,
    load_m5_percentile,
)


class SharedM5Test(unittest.TestCase):
    def test_named_scenarios_have_one_canonical_order(self):
        self.assertEqual(
            M5_SCENARIOS,
            ("worst_p25", "median_p50", "best_p75"),
        )
        self.assertEqual(
            tuple(M5_SCENARIO_PERCENTILES),
            tuple(M5_SCENARIO_LABELS),
        )
        self.assertEqual(M5_SCENARIO_PERCENTILES["best_p75"], 75.0)

    def test_unscoped_lookup_prefers_common_global_both_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m5_depth_quantiles_by_band.csv"
            pd.DataFrame({
                "threshold_scope": [
                    "common_global", "common_global",
                    "latitude_conditioned", "latitude_conditioned",
                ],
                "threshold_latitude_group": ["both", "both", "high", "high"],
                "band": ["g", "r", "g", "r"],
                "quantile": [0.75, 0.75, 0.75, 0.75],
                "m5": [25.1, 24.7, 25.8, 25.4],
            }).to_csv(path, index=False)

            result = depths_from_table(path, 75.0, ("g", "r"))

            self.assertEqual(result, {"g": 25.1, "r": 24.7})

    def test_explicit_latitude_scope_selects_high_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m5_depth_quantiles_by_band.csv"
            pd.DataFrame({
                "threshold_scope": [
                    "common_global", "common_global",
                    "latitude_conditioned", "latitude_conditioned",
                ],
                "threshold_latitude_group": ["both", "both", "high", "high"],
                "band": ["g", "r", "g", "r"],
                "quantile": [75.0, 75.0, 75.0, 75.0],
                "m5": [25.1, 24.7, 25.8, 25.4],
            }).to_csv(path, index=False)

            result = depths_from_table(
                path,
                75.0,
                ("g", "r"),
                threshold_scope="latitude_conditioned",
            )

            self.assertEqual(result, {"g": 25.8, "r": 25.4})

    def test_legacy_median_table_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opsim_m5_medians_by_band.csv"
            pd.DataFrame({
                "band": ["g", "r"],
                "m5_median": [24.9, 24.4],
            }).to_csv(path, index=False)

            result = depths_from_table(path, 50.0, ("g", "r"))

            self.assertEqual(result, {"g": 24.9, "r": 24.4})

    def test_raw_opsim_fallback_is_disabled_for_latitude_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            opsim = root / "opsim"
            run.mkdir()
            opsim.mkdir()
            pd.DataFrame({
                "band": ["g", "g", "r", "r"],
                "fiveSigmaDepth": [24.0, 26.0, 23.0, 25.0],
            }).to_csv(opsim / "opsim_visits_field000.csv", index=False)

            with self.assertRaises(FileNotFoundError):
                load_m5_percentile(
                    run,
                    opsim,
                    50.0,
                    ("g", "r"),
                    threshold_scope="latitude_conditioned",
                )

            depths, source = load_m5_percentile(
                run,
                opsim,
                50.0,
                ("g", "r"),
                threshold_scope="common_global",
            )
            self.assertEqual(depths, {"g": 25.0, "r": 24.0})
            self.assertIn("raw OpSim visits", source)


if __name__ == "__main__":
    unittest.main()
