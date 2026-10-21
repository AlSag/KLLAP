from __future__ import annotations

import ast
import unittest
from pathlib import Path

from kne_envelope_core import STAGES
from kne_m5 import M5_SCENARIOS


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_COMMANDS = (
    "get_opsim_baseline.py",
    "generate_kne_lightcurves.py",
    "build_color_envelopes.py",
    "analyze_envelope_population_convergence_fast.py",
    "plot_colour_envelope_realizations.py",
    "plot_kne_colour_isocontours.py",
    "run_latitude_experiment.py",
    "analyze_kne_latitude.py",
)

RETIRED_COMMANDS = (
    "analyze_envelope_population_convergence.py",
    "replot_envelope_population_convergence.py",
    "plot_convergence_support_diagnostics.py",
    "plot_kne_colour_parameter_space.py",
    "estimate_kne_lsst_distance_reach.py",
    "analyze_kne_latitude_losses.py",
    "compare_kne_latitude_bins.py",
    "plot_kne_low_high_plateau.py",
)


class RepositoryArchitectureTest(unittest.TestCase):
    def test_canonical_stage_and_m5_scenario_order(self):
        self.assertEqual(
            STAGES,
            (
                "intrinsic_no_m5_no_mw",
                "no_m5_with_mw",
                "m5_no_mw",
                "m5_with_mw",
            ),
        )
        self.assertEqual(
            M5_SCENARIOS,
            ("worst_p25", "median_p50", "best_p75"),
        )

    def test_public_commands_exist_and_have_a_module_guard(self):
        for relative in PUBLIC_COMMANDS:
            with self.subTest(command=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file())
                tree = ast.parse(path.read_text())
                guarded_main = any(
                    isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"
                    and any(
                        isinstance(comparator, ast.Constant)
                        and comparator.value == "__main__"
                        for comparator in node.test.comparators
                    )
                    for node in tree.body
                )
                self.assertTrue(guarded_main)

    def test_retired_redundant_commands_do_not_return(self):
        for relative in RETIRED_COMMANDS:
            with self.subTest(command=relative):
                self.assertFalse((ROOT / relative).exists())


if __name__ == "__main__":
    unittest.main()
