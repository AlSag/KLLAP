from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import build_color_envelopes as builder


class EnvelopeBuilderGoldenTest(unittest.TestCase):
    """Protect the scientific output while the command is reorganised."""

    def _make_run(self, root: Path) -> None:
        csv_dir = root / "csv"
        csv_dir.mkdir()
        summary_rows = []
        for event_index in range(12):
            summary_rows.append({
                "event_id": event_index,
                "model": "Bu2026_MLP",
                "ebv_mw": 0.0,
                "inclination_EM": 0.2 + 0.01 * event_index,
                "luminosity_distance": 40.0 + event_index,
                "redshift": 0.01 + 0.0001 * event_index,
                "log10_mej_dyn": -3.0 + 0.02 * event_index,
                "log10_mej_wind": -2.5 + 0.01 * event_index,
                "v_ej_dyn": 0.1,
                "Ye_dyn": 0.2,
                "v_ej_wind": 0.05,
                "Ye_wind": 0.3,
                "phi": 30.0,
            })
            colour_offset = -0.3 + 0.06 * event_index
            lightcurve_rows = []
            for time_days in (0.6, 1.0, 1.4):
                lightcurve_rows.extend((
                    {
                        "t_days": time_days,
                        "band": "g",
                        "mag": 20.0 + colour_offset + 0.1 * (time_days - 0.6),
                        "photometric_system": "lsst",
                    },
                    {
                        "t_days": time_days,
                        "band": "r",
                        "mag": 20.0,
                        "photometric_system": "lsst",
                    },
                ))
            pd.DataFrame(lightcurve_rows).to_csv(
                csv_dir / f"lightcurve_{event_index:04d}.csv", index=False
            )
        pd.DataFrame(summary_rows).to_csv(root / "summary.csv", index=False)
        pd.DataFrame([
            {
                "threshold_scope": "common_global",
                "threshold_latitude_group": "both",
                "band": band,
                "scenario": "median_p50",
                "m5": 99.0,
            }
            for band in ("g", "r")
        ]).to_csv(root / "m5.csv", index=False)

    def test_reference_tables_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self._make_run(run_dir)
            output = run_dir / "output"
            args = builder.build_parser().parse_args([
                "--run-dir", str(run_dir),
                "--photometric-system", "lsst",
                "--color-pair", "g-r",
                "--t-min", "0.4",
                "--t-max", "1.6",
                "--bin-width", "0.4",
                "--n-boot", "30",
                "--bootstrap-threshold-mag", "10",
                "--minimum-events", "2",
                "--minimum-tail-events", "0",
                "--train-fraction", "0.7",
                "--validation-fraction", "0.15",
                "--seed", "12345",
                "--colored-plot-sizes", "2",
                "--outlier-table-plot-size", "2",
                "--max-outlier-table-events", "2",
                "--m5-quantile-table", str(run_dir / "m5.csv"),
                "--output-dir", str(output),
            ])
            self.assertEqual(builder.main(args), output)

            summary = pd.read_csv(output / "stage_summary.csv")
            self.assertEqual(summary["stage"].tolist(), list(builder.STAGES))
            self.assertEqual(summary["N_valid_bins_anywhere"].tolist(), [2] * 4)
            self.assertEqual(
                summary["N_initial_consecutive_valid_bins"].tolist(), [2] * 4
            )
            np.testing.assert_allclose(
                summary["last_initial_consecutive_time_days"], [1.2] * 4
            )
            self.assertEqual(summary["N_native_population_points"].tolist(), [36] * 4)

            stage_dir = output / "intrinsic_no_m5_no_mw"
            rules = pd.read_csv(stage_dir / "dynamic_color_cut_rules.csv")
            self.assertEqual(rules["N_train_events"].tolist(), [8, 8])
            np.testing.assert_allclose(
                rules["color_median"], [0.0900011063, 0.1300010681], atol=2e-6
            )
            np.testing.assert_allclose(
                rules["color_min"], [-0.2398317726, -0.1998313904], atol=2e-6
            )
            np.testing.assert_allclose(
                rules["color_max"], [0.3599147038, 0.3999150848], atol=2e-6
            )

            availability = pd.read_csv(
                stage_dir / "envelope_availability_diagnostics.csv"
            )
            self.assertEqual(availability["N_train_model_supported"].tolist(), [8, 8, 0])
            self.assertEqual(
                availability["candidate_envelope_valid"].tolist(), [True, True, False]
            )

            split = pd.read_csv(output / "event_split.csv")
            self.assertEqual(
                split["split"].tolist(),
                ["validation", "train", "test", "train", "train", "test",
                 "train", "train", "train", "test", "train", "train"],
            )


if __name__ == "__main__":
    unittest.main()
