from __future__ import annotations

import unittest

import numpy as np

from estimate_kne_lsst_distance_reach import (
    crossing_distance,
    decreasing_isotonic,
    wilson_interval,
)


class DistanceReachTest(unittest.TestCase):
    def test_decreasing_isotonic_pools_upward_fluctuation(self) -> None:
        values = np.array([0.9, 0.7, 0.75, 0.2])
        fitted = decreasing_isotonic(values, np.ones(4))
        np.testing.assert_allclose(fitted, [0.9, 0.725, 0.725, 0.2])
        self.assertTrue(np.all(np.diff(fitted) <= 0))

    def test_crossing_distance_uses_linear_interpolation(self) -> None:
        distance, status = crossing_distance(
            np.array([100.0, 200.0, 300.0]),
            np.array([0.8, 0.4, 0.0]),
            0.5,
        )
        self.assertEqual(status, "interpolated")
        self.assertAlmostEqual(distance, 175.0)

    def test_unbracketed_low_completeness_is_reported(self) -> None:
        distance, status = crossing_distance(
            np.array([100.0, 200.0]),
            np.array([0.4, 0.2]),
            0.5,
        )
        self.assertTrue(np.isnan(distance))
        self.assertEqual(status, "below_target_at_minimum_distance")

    def test_wilson_interval_contains_fraction(self) -> None:
        low, high = wilson_interval(50, 100, 0.95)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


if __name__ == "__main__":
    unittest.main()
