"""BOOM contracts: timestamp uniqueness, selections, errors and range audits."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from generate_kne_lightcurves import BOOM_M5_DEFAULT, synthetic_to_boom_csv
from snia_parquet_to_simu_input import export_population

ROOT = Path(__file__).resolve().parents[1]


class BoomExportsTest(unittest.TestCase):
    def test_complete_export_keeps_faint_points_and_m5_mode_drops_them(self):
        table = pd.DataFrame({
            "time_mjd": [60000., 60000., 60000.2, 60000.2],
            "t_days": [0., 0., .2, .2],
            "band": ["g", "r", "g", "r"],
            "mag": [24.8, 24.4, 25.8, 25.4],
        })
        original = table.copy(deep=True)
        complete = synthetic_to_boom_csv(table, 7, BOOM_M5_DEFAULT, export_mode="complete")
        detected = synthetic_to_boom_csv(table, 7, BOOM_M5_DEFAULT, export_mode="m5-detected")
        pd.testing.assert_frame_equal(table, original)
        self.assertEqual(len(complete), 4)
        self.assertEqual(len(detected), 2)
        self.assertTrue(np.isfinite(complete[["mag", "mag_err"]]).all().all())
        np.testing.assert_allclose(complete.mag_err[:2], np.sqrt(.005**2 + .04))
        self.assertGreater(complete.mag_err.iloc[2], complete.mag_err.iloc[0])
        self.assertEqual(complete.candidate_id.tolist(), [f"0007_{i}" for i in range(1, 5)])
        self.assertEqual(complete.objectid.dropna().tolist(), [7])

    def test_timestamp_shift_is_shortened_before_next_epoch(self):
        table = pd.DataFrame({
            "time_mjd": [60000.] * 6 + [60000.00001],
            "t_days": [0.] * 6 + [.00001],
            "band": list("ugrizy") + ["g"], "mag": [20.] * 7,
        })
        result = synthetic_to_boom_csv(table, 0, BOOM_M5_DEFAULT, export_mode="complete")
        self.assertTrue(result.time_mjd.is_unique)
        self.assertTrue(result.time_mjd.is_monotonic_increasing)
        self.assertLess(result.time_mjd.iloc[-2], 60000.00001)
        np.testing.assert_allclose(result.time_mjd - 60000., result.t_days, atol=1e-11)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet backend not installed")
    def test_snia_preserves_supplied_errors_and_encodes_faint_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "snia.parquet"
            pd.DataFrame({
                "sn_id": [0, 0, 4], "mjd": [60000., 60000., 60001.],
                "band": ["g", "r", "i"], "mag": [24.8, 26., 22.],
                "magerr": [.123, .987, .456], "ra": [10.] * 3, "dec": [-30.] * 3,
            }).to_parquet(source, index=False)
            report = export_population(source, root / "boom", 30., False)
            self.assertEqual(report["detections"], 2)
            self.assertEqual(report["upper_limits"], 1)
            first = pd.read_csv(root / "boom/lightcurve_0000.csv")
            self.assertAlmostEqual(first.mag_err.iloc[0], .123)
            self.assertTrue(np.isnan(first.mag.iloc[1]))
            self.assertEqual(first.mag_err.iloc[1], 24.4)
            self.assertEqual(first.candidate_id.tolist(), ["0000_1", "0000_2"])
            self.assertTrue(first.time_mjd.is_unique)
            last = pd.read_csv(root / "boom/lightcurve_0004.csv")
            self.assertAlmostEqual(last.mag_err.iloc[0], .456)

    def test_audit_uses_the_inclusive_filename_range_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (1, 10, 11, 12):
                pd.DataFrame({"objectid": [index], "time_mjd": [60000.],
                              "band": ["g"], "mag": [20.], "mag_err": [.1]}).to_csv(
                    root / f"lightcurve_{index:04d}.csv", index=False)
            pd.DataFrame({"objectid": [11], "n_passed_filter": [3]}).to_csv(
                root / "stats.csv", index=False)
            completed = subprocess.run([
                sys.executable, str(ROOT / "compute_color_filter_efficiency.py"),
                "--input-dir", str(root), "--filter-stats", str(root / "stats.csv"),
                "--start-index", "10", "--end-index", "11",
                "--summary-output", str(root / "summary.json"),
                "--audit-output", str(root / "audit.csv"),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["expected_events"], 2)
            self.assertEqual(summary["passed_filter"], 1)
            self.assertEqual(summary["conditional_filter_efficiency_percent"], 50.)
            self.assertEqual(pd.read_csv(root / "audit.csv").file_index.tolist(), [10, 11])
