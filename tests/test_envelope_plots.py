from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kllap.envelope.plots import (
    comparison_ylim,
    draw_envelope,
    draw_population,
    format_axes,
)


class EnvelopePlotsTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_population_plot_keeps_tracks_and_isolated_points(self) -> None:
        fig, ax = plt.subplots()
        draw_population(
            ax,
            [np.array([0.5, 1.0]), np.array([2.0])],
            [np.array([0.1, 0.2]), np.array([0.3])],
        )
        # One LineCollection for the trajectory and one PathCollection for the
        # isolated supported point.
        self.assertEqual(len(ax.collections), 2)

    def test_envelope_draws_three_boundaries_per_bin(self) -> None:
        rules = pd.DataFrame(
            {
                "t_min_days": [0.4, 0.8],
                "t_max_days": [0.8, 1.2],
                "color_min": [-0.5, -0.4],
                "color_median": [0.0, 0.1],
                "color_max": [0.5, 0.6],
            }
        )
        fig, ax = plt.subplots()
        draw_envelope(ax, rules)
        self.assertEqual(len(ax.lines), 6)
        self.assertEqual(len(ax.collections), 2)

    def test_axes_and_limits_are_presentation_only(self) -> None:
        fig, ax = plt.subplots()
        format_axes(ax, "g-r", 0.4, 4.0)
        self.assertEqual(ax.get_xlabel(), r"$t-t_{\rm merger}$ [days]")
        self.assertEqual(ax.get_ylabel(), "g-r [mag]")
        np.testing.assert_allclose(ax.get_xlim(), [0.4, 4.0])

        limits = comparison_ylim(
            [np.array([-1.0, 0.0, 2.0])],
            pd.DataFrame({"color_min": [-0.5], "color_max": [1.5]}),
        )
        self.assertIsNotNone(limits)
        self.assertLess(limits[0], -0.9)
        self.assertGreater(limits[1], 1.9)


if __name__ == "__main__":
    unittest.main()
