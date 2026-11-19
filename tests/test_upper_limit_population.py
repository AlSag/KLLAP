"""Empirical colours and UL classifications must not impose sample-size gates."""
from __future__ import annotations

import unittest

import numpy as np

from build_color_envelopes import (
    build_stage_rules, censored_colour_arrays, default_configuration,
)


class UpperLimitPopulationTest(unittest.TestCase):
    def test_detection_limit_and_unknown_categories_are_distinct(self):
        first = np.array([[24., 25., 23., 26., np.nan, 24.8]])
        second = np.array([[23., 23., 25., 27., 23., 24.4]])
        status, lower, upper = censored_colour_arrays(first, second, 24.8, 24.4)
        self.assertEqual(status.tolist(), [[1, 2, 3, 4, 0, 1]])
        self.assertAlmostEqual(lower[0, 1], 1.8)
        self.assertTrue(np.isposinf(upper[0, 1]))
        self.assertAlmostEqual(upper[0, 2], -1.4)
        self.assertTrue(np.isneginf(lower[0, 3]))
        self.assertTrue(np.isposinf(upper[0, 3]))
        self.assertTrue(np.isnan(lower[0, 4]))

    def test_singletons_and_bins_after_an_empty_bin_remain_in_full_rules(self):
        values = np.array([[1., np.nan, 3.]])
        args = default_configuration()
        full, consecutive, _, availability = build_stage_rules(
            values, np.isfinite(values), np.array([True]), "no_m5",
            np.array([0., 1., 2., 3.]), np.array([.5, 1.5, 2.5]), [1.], args,
        )
        self.assertEqual(full.bin_index.tolist(), [0, 2])
        self.assertEqual(full.color_min.tolist(), [1., 3.])
        self.assertEqual(full.N_population_events.tolist(), [1, 1])
        self.assertEqual(consecutive.bin_index.tolist(), [0])
        self.assertEqual([row["N_population_stage_available"] for row in availability], [1, 0, 1])
