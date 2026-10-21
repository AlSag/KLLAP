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
        ):
            tree = ast.parse((ROOT / filename).read_text())
            functions = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            self.assertIn(function_name, functions)
        self.assertFalse((ROOT / "kne_generation_cli.py").exists())
        self.assertFalse((ROOT / "kne_opsim_cli.py").exists())
        self.assertFalse((ROOT / "kne_envelope_cli.py").exists())

    def test_programmatic_defaults_require_no_terminal_flags(self):
        import generate_kne_lightcurves
        import build_color_envelopes

        generation = generate_kne_lightcurves.default_runtime_configuration()
        envelope = build_color_envelopes.default_configuration()
        self.assertEqual(generation.model, "Bu2026_MLP")
        self.assertEqual(envelope.color_pair, "g-r")

    def test_public_workflows_have_no_argument_parser_surface(self):
        for filename in (
            "build_color_envelopes.py",
            "analyze_envelope_population_convergence_fast.py",
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
