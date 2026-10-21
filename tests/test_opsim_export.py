from __future__ import annotations

import unittest
import inspect

import numpy as np
import pandas as pd

import get_opsim_baseline as opsim


class FakeDustMap:
    def run(self, slice_points):
        size = len(np.atleast_1d(slice_points["ra"]))
        return {"ebv": np.linspace(0.1, 0.2, size)}


class OpSimExportTest(unittest.TestCase):
    def test_import_exposes_main_without_reading_runtime_configuration(self) -> None:
        self.assertTrue(callable(opsim.main))
        self.assertFalse(hasattr(opsim, "RUNTIME"))

    def test_public_configuration_has_no_command_line_arguments(self) -> None:
        self.assertEqual(
            len(inspect.signature(opsim.runtime_configuration).parameters), 0
        )

    def test_selection_builders_keep_queries_parameterized(self) -> None:
        target = opsim.build_selection("TARGET", target_name="DD:COSMOS")
        self.assertEqual(target["where_sql"], "target_name = ?")
        self.assertEqual(target["params"], ("DD:COSMOS",))
        self.assertEqual(target["label"], "dd_cosmos")

        ddf = opsim.build_selection("ONE_DDF", ddf_name="COSMOS")
        self.assertIn("UPPER(?)", ddf["where_sql"])
        self.assertEqual(ddf["params"], ("COSMOS",))

    def test_stratum_allocation_reports_capacity_deficit_for_global_topup(self) -> None:
        strata = pd.DataFrame(
            {
                "target_name": ["wide", "deep", "small"],
                "n_centers": [10, 4, 1],
                "n_obs": [100, 80, 5],
            }
        )
        allocated = opsim.allocate_n_per_stratum(
            strata,
            n_total=8,
            weight_col="n_centers",
            min_per_target=1,
        )
        # Capacity clipping is local to this helper. The sampling function
        # deliberately fills this one-field deficit from the remaining global
        # pool after cross-stratum deduplication.
        self.assertEqual(int(allocated["n_alloc"].sum()), 7)
        self.assertLessEqual(int(allocated["n_alloc"].sum()), 8)
        self.assertTrue((allocated["n_alloc"] <= allocated["n_centers"]).all())
        self.assertTrue((allocated["n_alloc"] >= 1).all())

    def test_extinction_columns_use_each_band_coefficient(self) -> None:
        centers = pd.DataFrame(
            {"fieldRA": [10.0, 20.0], "fieldDec": [-5.0, 5.0]}
        )
        coefficients = {band: float(index + 1) for index, band in enumerate("ugrizy")}
        result = opsim.add_mw_extinction_to_centers(
            centers,
            FakeDustMap(),
            coefficients,
        )
        np.testing.assert_allclose(result["ebv_mw"], [0.1, 0.2])
        for band in "ugrizy":
            np.testing.assert_allclose(
                result[f"A_{band}_mw"],
                result["ebv_mw"] * coefficients[band],
            )


if __name__ == "__main__":
    unittest.main()
