import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_color_envelopes import load_ztf_depth_thresholds


class ZTFDepthPipelineTest(unittest.TestCase):
    def test_manual_fixed_depths(self):
        thresholds, provenance = load_ztf_depth_thresholds(
            directory=None,
            quantile_table=None,
            scenario="median_p50",
            bands=("g", "r"),
            n_samples=2000,
            seed=123,
            depth_source="manual",
            manual_thresholds={"g": 20.8, "r": 20.6, "i": 20.3},
        )

        self.assertEqual(thresholds, {"g": 20.8, "r": 20.6})
        self.assertEqual(set(provenance["scenario"]), {"manual_fixed"})
        self.assertEqual(
            set(provenance["source_kind"]),
            {"user_supplied_fixed_depth"},
        )

    def test_manual_depth_requires_each_colour_band(self):
        with self.assertRaisesRegex(ValueError, "--ztf-m5-r"):
            load_ztf_depth_thresholds(
                directory=None,
                quantile_table=None,
                scenario="median_p50",
                bands=("g", "r"),
                n_samples=2000,
                seed=123,
                depth_source="manual",
                manual_thresholds={"g": 20.8},
            )

    def test_precomputed_quantile_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ztf_depth_quantiles.csv"
            pd.DataFrame(
                [
                    {"band": "g", "scenario": "median_p50", "m5": 20.7},
                    {"band": "r", "scenario": "median_p50", "m5": 20.5},
                    {"band": "i", "scenario": "median_p50", "m5": 20.3},
                ]
            ).to_csv(path, index=False)

            thresholds, provenance = load_ztf_depth_thresholds(
                directory=None,
                quantile_table=path,
                scenario="median_p50",
                bands=("g", "r"),
                n_samples=2000,
                seed=123,
            )

            self.assertEqual(thresholds, {"g": 20.7, "r": 20.5})
            self.assertEqual(set(provenance["band"]), {"g", "r"})
            self.assertTrue(provenance["source_sha256"].str.len().eq(64).all())

    def test_ztf_rejects_unsupported_band(self):
        with self.assertRaisesRegex(ValueError, "only g, r and i"):
            load_ztf_depth_thresholds(
                directory=None,
                quantile_table=None,
                scenario="median_p50",
                bands=("g", "z"),
                n_samples=2000,
                seed=123,
            )


if __name__ == "__main__":
    unittest.main()
