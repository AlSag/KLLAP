from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

import build_color_envelopes as envelope_command
import get_opsim_baseline as opsim_command
import kne_envelope_product_plots as envelope_plots
import get_opsim_baseline as opsim_plots


ROOT = Path(__file__).resolve().parents[1]


class PlotModuleBoundaryTest(unittest.TestCase):
    def test_commands_reexport_moved_plot_functions(self):
        self.assertIs(
            envelope_command.save_stage_plots,
            envelope_plots.save_stage_plots,
        )
        self.assertIs(
            envelope_command.save_dual_system_comparison_plots,
            envelope_plots.save_dual_system_comparison_plots,
        )
        self.assertIs(
            opsim_command.save_m5_violin_plot,
            opsim_plots.save_m5_violin_plot,
        )

    def test_self_contained_opsim_command_owns_its_plots(self):
        tree = ast.parse((ROOT / "get_opsim_baseline.py").read_text())
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("save_m5_violin_plot", functions)
        self.assertIn("show_revisit_histograms", functions)

    def test_empty_revisit_table_is_a_valid_no_op(self):
        table = pd.DataFrame(columns=["band", "delta_t_days"])
        self.assertIsNone(
            opsim_plots.show_revisit_histograms(
                table, 0, linear=True, logarithmic=True, bins=10
            )
        )

    def test_stage_directory_convention_is_preserved(self):
        self.assertEqual(
            envelope_plots.stage_output_name("m5_no_mw", "best_p75"),
            "m5_no_mw_best_p75",
        )
        self.assertEqual(
            envelope_plots.stage_output_name(
                "intrinsic_no_m5_no_mw", "best_p75"
            ),
            "intrinsic_no_m5_no_mw",
        )

    def test_event_display_colours_are_order_stable(self):
        indices = np.array([8, 2, 5])
        first = envelope_plots.event_colour_map(indices)
        second = envelope_plots.event_colour_map(indices)
        self.assertEqual(first, second)
        self.assertEqual(list(first), [8, 2, 5])
        self.assertEqual(len(set(first.values())), 3)


if __name__ == "__main__":
    unittest.main()
