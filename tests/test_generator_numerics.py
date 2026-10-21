"""Deterministic numerical contracts for the light-curve generator."""
import ast
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from kllap.common import interpolate_magnitude_log_time
from kllap.generation.core import (
    apply_synthetic_m5_median_cut,
    derive_sky_seed,
    ensure_mw_extinction_columns,
    keep_only_mw_extincted_synthetic_magnitude,
    trim_lsstlike_after_last_detection,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "kllap/generation/command.py"


def generator_functions(*names, **namespace):
    """Load selected generator functions without importing heavy backends."""
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    definitions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = set(names).difference(definitions)
    if missing:
        raise AssertionError(f"Missing generator definitions: {sorted(missing)}")
    module = ast.Module(body=[definitions[name] for name in names], type_ignores=[])
    scope = {"np": np, "pd": pd}
    scope.update(namespace)
    exec(compile(module, str(GENERATOR), "exec"), scope)
    return scope


class MidpointRng:
    """Small deterministic stand-in for Generator.uniform()."""

    def uniform(self, low, high):
        return 0.5 * (float(low) + float(high))


class GeneratorNumericsTest(unittest.TestCase):
    def test_lazy_rubin_initialization_updates_multisystem_coefficients(self):
        coefficients = {band: float(index) for index, band in enumerate("ugrizy", 1)}

        class PhotometricParameters:
            def __init__(self, bandpass):
                self.bandpass = bandpass

        scope = generator_functions(
            "initialize_rubin_photometry",
            BP=None,
            PHOT=None,
            DUST_R_X=None,
            DUST_R_X_BY_SYSTEM={"lsst": None},
            rubin_bandpasses=lambda: {band: f"bp-{band}" for band in "ugrizy"},
            PhotometricParameters=PhotometricParameters,
            get_rubin_dust_coefficients=lambda: coefficients,
        )
        with redirect_stdout(io.StringIO()):
            scope["initialize_rubin_photometry"]()

        self.assertEqual(scope["DUST_R_X"], coefficients)
        self.assertIs(scope["DUST_R_X_BY_SYSTEM"]["lsst"], coefficients)
        self.assertEqual(scope["PHOT"]["g"].bandpass, "g")

    def test_log_time_interpolation_has_no_extrapolation(self):
        query = np.array([0.5, 1.0, np.sqrt(10.0), 10.0, 100.0, 101.0])
        result = interpolate_magnitude_log_time(
            [1.0, 10.0, 100.0], [20.0, 22.0, 24.0], query
        )
        np.testing.assert_allclose(
            result[1:5], [20.0, 21.0, 22.0, 24.0], atol=1e-12
        )
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[-1]))

    def test_log_time_interpolation_ignores_invalid_model_magnitude(self):
        result = interpolate_magnitude_log_time(
            [1.0, 10.0, 100.0], [20.0, np.nan, 24.0], [10.0]
        )
        np.testing.assert_allclose(result, [22.0], atol=1e-12)

    def test_mw_extinction_is_computed_from_each_band_coefficient(self):
        coefficients = {
            "u": 4.0,
            "g": 3.0,
            "r": 2.0,
            "i": 1.5,
            "z": 1.2,
            "y": 1.0,
        }
        visits = pd.DataFrame({"band": ["g", "r"], "ebv_mw": [0.1, 0.2]})
        result = ensure_mw_extinction_columns(visits, coefficients)

        np.testing.assert_allclose(result["R_mw"], [3.0, 2.0])
        np.testing.assert_allclose(result["A_mw"], [0.3, 0.4])
        np.testing.assert_allclose(result["A_g_mw"], [0.3, 0.6])
        self.assertNotIn("A_mw", visits.columns)

    def test_m5_cut_is_inclusive_and_rebuilds_identifiers(self):
        measurements = pd.DataFrame(
            {
                "band": ["g", "g", "r", "i"],
                "mag": [24.0, 24.01, 22.4, 20.0],
                "objectid": ["old"] * 4,
                "candidate_id": ["old_1", "old_2", "old_3", "old_4"],
            }
        )
        result = apply_synthetic_m5_median_cut(
            measurements, "event42", {"g": 24.0, "r": 22.3}
        )

        self.assertEqual(result["band"].tolist(), ["g"])
        self.assertEqual(result["mag"].tolist(), [24.0])
        self.assertEqual(result["objectid"].tolist(), ["event42"])
        self.assertEqual(result["candidate_id"].tolist(), ["event42_1"])

    def test_saved_synthetic_magnitude_is_the_extincted_value(self):
        measurements = pd.DataFrame(
            {"mag": [20.0, 21.0], "mag_mw_extincted": [20.3, 21.4]}
        )
        result = keep_only_mw_extincted_synthetic_magnitude(measurements)

        np.testing.assert_allclose(result["mag"], [20.3, 21.4])
        self.assertNotIn("mag_mw_extincted", result.columns)
        np.testing.assert_allclose(measurements["mag"], [20.0, 21.0])

    def test_last_detection_window_never_removes_a_detection(self):
        observations = pd.DataFrame(
            {
                "observationStartMJD": [109.0, 101.0, 108.0, 105.0, 102.0, 110.0],
                "t_days": [9.0, 1.0, 8.0, 5.0, 2.0, 10.0],
                "mag_true": [20.0, 20.0, 20.0, 20.0, 20.0, np.nan],
                "detected": [False, False, False, True, True, False],
            }
        )
        result, last_detection, window_end, has_detection = (
            trim_lsstlike_after_last_detection(
            observations, days_after_last_detection=3.0
            )
        )

        self.assertTrue(has_detection)
        self.assertEqual(last_detection, 5.0)
        self.assertEqual(window_end, 8.0)
        self.assertEqual(result["t_days"].tolist(), [1.0, 2.0, 5.0, 8.0])
        self.assertEqual(int(result["detected"].sum()), 2)

    def test_inclination_sampling_matches_the_selected_geometry(self):
        rng = MidpointRng()
        uniform_angle = generator_functions(
            "sample_inclination_for_event",
            INCLINATION_SAMPLING="uniform_angle",
            INCLINATION_MIN_RAD=0.0,
            INCLINATION_MAX_RAD=np.pi / 2.0,
        )["sample_inclination_for_event"]
        isotropic = generator_functions(
            "sample_inclination_for_event",
            INCLINATION_SAMPLING="uniform_cos",
            INCLINATION_MIN_RAD=0.0,
            INCLINATION_MAX_RAD=np.pi / 2.0,
        )["sample_inclination_for_event"]

        self.assertAlmostEqual(uniform_angle(rng), np.pi / 4.0)
        self.assertAlmostEqual(isotropic(rng), np.pi / 3.0)

    def test_sky_seeds_are_reproducible_and_latitude_specific(self):
        values = {
            derive_sky_seed(12345, "all", 0.0),
            derive_sky_seed(12345, "low", 10.0),
            derive_sky_seed(12345, "high", 10.0),
            derive_sky_seed(12345, "bin", 10.0, 20.0),
        }
        self.assertEqual(len(values), 4)
        self.assertEqual(
            derive_sky_seed(12345, "bin", 10.0, 20.0),
            derive_sky_seed(12345, "bin", 10.0, 20.0),
        )
        with self.assertRaises(ValueError):
            derive_sky_seed(12345, "bin", 10.0)


if __name__ == "__main__":
    unittest.main()
