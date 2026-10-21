from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from plot_kne_colour_isocontours import (
    display_axis_domain,
    draw_panel,
    fixed_levels,
    parameter_colour_grid,
    plotting_axis,
    population_model,
)


class ColourIsocontoursTest(unittest.TestCase):
    def test_fixed_levels_include_requested_endpoints_and_exact_zero(self):
        levels = fixed_levels("-0.6:0.6:0.3")

        np.testing.assert_allclose(levels, [-0.6, -0.3, 0.0, 0.3, 0.6])
        self.assertEqual(levels[2], 0.0)

    def test_grid_uses_cell_median_and_masks_insufficient_support(self):
        x = np.array([0.2, 0.3, 0.7, 1.2, 1.3])
        y = np.array([0.2, 0.3, 0.7, 0.2, 0.3])
        colour = np.array([1.0, 3.0, 100.0, 4.0, 8.0])
        available = np.ones(5, dtype=bool)
        edges = np.array([0.0, 1.0, 2.0])

        grid = parameter_colour_grid(
            x,
            y,
            colour,
            available,
            edges,
            edges,
            minimum_cell_events=2,
            smooth_sigma=0.0,
        )

        self.assertEqual(grid["count"][0, 0], 3)
        self.assertEqual(grid["p50"][0, 0], 3.0)
        self.assertTrue(grid["supported"][0, 0])
        self.assertEqual(grid["count"][1, 0], 2)
        self.assertEqual(grid["p50"][1, 0], 6.0)
        self.assertFalse(grid["supported"][0, 1])
        self.assertTrue(np.isnan(grid["smoothed_p50"][0, 1]))

    def test_smoothing_never_fills_an_unsupported_cell(self):
        x = np.array([0.2, 0.3, 1.2, 1.3])
        y = np.array([0.2, 0.3, 0.2, 0.3])
        colour = np.array([1.0, 1.0, 3.0, 3.0])
        available = np.ones(4, dtype=bool)
        edges = np.array([0.0, 1.0, 2.0])

        grid = parameter_colour_grid(
            x,
            y,
            colour,
            available,
            edges,
            edges,
            minimum_cell_events=2,
            smooth_sigma=2.0,
        )

        np.testing.assert_allclose(
            grid["p50"][:, 0],
            [1.0, 3.0],
        )
        self.assertTrue(np.isfinite(grid["smoothed_p50"][:, 0]).all())
        self.assertTrue(np.isnan(grid["smoothed_p50"][:, 1]).all())

    def test_log_mass_axis_changes_only_plot_coordinates(self):
        log_edges = np.array([-4.0, -3.0, -2.0])
        coordinates, label, scale = plotting_axis("log10_mej_dyn", log_edges)

        np.testing.assert_allclose(coordinates, [1e-4, 1e-3, 1e-2])
        self.assertEqual(scale, "log")
        self.assertIn("M_", label)

    def test_bu2026_display_domain_can_use_full_model_validity(self):
        metadata = pd.DataFrame({"model": ["Bu2026_MLP", "Bu2026_MLP"]})
        model = population_model(metadata)
        domain = display_axis_domain(
            np.array([-2.0, -0.9]), "log10_mej_wind", model, "model"
        )

        np.testing.assert_allclose(domain, [-4.0, -0.5502131530142267])

    def test_count_heatmap_keeps_nonzero_cells_below_contour_support(self):
        grid = {
            "count": np.array([[0, 2], [1, 4]]),
            "p50": np.array([[np.nan, 1.0], [2.0, 3.0]]),
            "supported": np.array([[False, False], [False, True]]),
            "smoothed_p50": np.array([[np.nan, np.nan], [np.nan, 3.0]]),
            "N_available": 7,
        }
        fig, ax = plt.subplots()
        mesh, _ = draw_panel(
            ax,
            grid,
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([]),
            LogNorm(vmin=1, vmax=4),
            plt.get_cmap("viridis"),
            0.4,
            0.8,
        )
        displayed = mesh.get_array()
        self.assertTrue(np.ma.getmaskarray(displayed)[0, 0])
        self.assertFalse(np.ma.getmaskarray(displayed)[0, 1])
        self.assertEqual(displayed[0, 1], 1)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
