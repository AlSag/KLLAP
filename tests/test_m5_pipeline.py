from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kne_pipeline_common import (
    compute_m5_depth_quantiles,
    discover_m5_quantile_table,
    write_m5_depth_quantiles,
)


class M5PipelineTest(unittest.TestCase):
    def make_export(self, root: Path) -> None:
        # Galactic-centre and north-Galactic-pole sight lines provide one low
        # and one high |b| field without relying on Astropy in the helper.
        pd.DataFrame(
            {
                "field_index": [0, 1],
                "ra0_deg": [266.4051, 192.8595],
                "dec0_deg": [-28.9362, 27.1283],
            }
        ).to_csv(root / "opsim_fields_index.csv", index=False)
        pd.DataFrame(
            {"band": ["g", "g", "r", "r"], "fiveSigmaDepth": [23, 25, 22, 24]}
        ).to_csv(root / "opsim_visits_field000.csv", index=False)
        pd.DataFrame(
            {"band": ["g", "g", "r", "r"], "fiveSigmaDepth": [24, 26, 23, 25]}
        ).to_csv(root / "opsim_visits_field001.csv", index=False)

    def test_quantile_schema_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_export(root)
            table = compute_m5_depth_quantiles(root, latitude_threshold_deg=20)
            required = {
                "threshold_scope",
                "threshold_latitude_group",
                "band",
                "scenario",
                "m5",
            }
            self.assertTrue(required.issubset(table.columns))
            row = table.loc[
                table["threshold_scope"].eq("common_global")
                & table["band"].eq("g")
                & table["scenario"].eq("median_p50")
            ]
            self.assertEqual(len(row), 1)
            self.assertAlmostEqual(float(row.iloc[0]["m5"]), 24.5)
            groups = set(
                table.loc[
                    table["threshold_scope"].eq("latitude_conditioned"),
                    "threshold_latitude_group",
                ]
            )
            self.assertEqual(groups, {"low", "high"})

    def test_discovery_from_recorded_opsim_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            export = base / "opsim"
            run = base / "runs" / "example"
            export.mkdir(parents=True)
            run.mkdir(parents=True)
            self.make_export(export)
            expected = write_m5_depth_quantiles(export)
            (run / "run_info.txt").write_text(
                "[opsim_context]\n"
                f"opsim_dir_resolved: {export.resolve()}\n",
                encoding="utf-8",
            )
            self.assertEqual(discover_m5_quantile_table(run), expected)


if __name__ == "__main__":
    unittest.main()
