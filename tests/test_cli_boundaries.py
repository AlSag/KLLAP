from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliBoundaryTest(unittest.TestCase):
    def test_generation_and_opsim_interfaces_live_in_public_scripts(self):
        for filename, function_name in (
            ("generate_kne_lightcurves.py", "configure_runtime"),
            ("get_opsim_baseline.py", "runtime_configuration"),
            ("build_color_envelopes.py", "interactive"),
            ("analyze_envelope_population_convergence_fast.py", "interactive_configuration"),
            ("plot_colour_envelope_realizations.py", "configure_runtime"),
            ("plot_kne_colour_isocontours.py", "interactive"),
            ("analyze_kne_latitude.py", "population_interactive"),
            ("run_latitude_experiment.py", "interactive_configuration"),
        ):
            tree = ast.parse((ROOT / filename).read_text())
            functions = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            self.assertIn(function_name, functions)
        self.assertFalse((ROOT / "kne_generation_cli.py").exists())
        self.assertFalse((ROOT / "kne_opsim_cli.py").exists())
        self.assertFalse((ROOT / "kne_envelope_cli.py").exists())
        self.assertFalse((ROOT / "kne_colour_population.py").exists())
        self.assertFalse((ROOT / "kne_latitude_population.py").exists())
        self.assertFalse((ROOT / "kne_latitude_comparison.py").exists())
        self.assertFalse((ROOT / "kne_latitude_plateau.py").exists())

    def test_programmatic_defaults_require_no_terminal_flags(self):
        import generate_kne_lightcurves
        import build_color_envelopes
        import plot_colour_envelope_realizations
        import plot_kne_colour_isocontours
        import analyze_kne_latitude
        import run_latitude_experiment

        generation = generate_kne_lightcurves.default_runtime_configuration()
        envelope = build_color_envelopes.default_configuration()
        realizations = plot_colour_envelope_realizations.default_configuration()
        isocontours = plot_kne_colour_isocontours.default_configuration()
        latitude = analyze_kne_latitude.population_default_configuration()
        experiment = run_latitude_experiment.default_configuration()
        self.assertEqual(generation.model, "Bu2026_MLP")
        self.assertEqual(envelope.color_pair, "g-r")
        self.assertEqual(realizations.times, "0.5,1.0,1.5,2.0")
        self.assertEqual(isocontours.stage, "compare_no_mw")
        self.assertEqual(latitude.m5_scenario, "median_p50")
        self.assertEqual(experiment.bin_edges, [0, 5, 10, 15, 20, 25, 30, 40, 50, 90])

    def test_public_workflows_have_no_argument_parser_surface(self):
        for filename in (
            "build_color_envelopes.py",
            "analyze_envelope_population_convergence_fast.py",
            "plot_colour_envelope_realizations.py",
            "plot_kne_colour_isocontours.py",
            "analyze_kne_latitude.py",
            "run_latitude_experiment.py",
        ):
            tree = ast.parse((ROOT / filename).read_text())
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"add_argument", "parse_args"}
            ]
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
