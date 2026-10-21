from __future__ import annotations

import ast
import unittest
from pathlib import Path

import kne_envelope_cli


ROOT = Path(__file__).resolve().parents[1]


class CliBoundaryTest(unittest.TestCase):
    def test_generation_and_opsim_interfaces_live_in_public_scripts(self):
        for filename, function_name in (
            ("generate_kne_lightcurves.py", "configure_runtime"),
            ("get_opsim_baseline.py", "runtime_configuration"),
        ):
            tree = ast.parse((ROOT / filename).read_text())
            functions = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            self.assertIn(function_name, functions)
        self.assertFalse((ROOT / "kne_generation_cli.py").exists())
        self.assertFalse((ROOT / "kne_opsim_cli.py").exists())

    def test_programmatic_defaults_require_no_terminal_flags(self):
        import generate_kne_lightcurves

        generation = generate_kne_lightcurves.default_runtime_configuration()
        envelope = kne_envelope_cli.build_parser().parse_args([])
        self.assertEqual(generation.model, "Bu2026_MLP")
        self.assertEqual(envelope.color_pair, "g-r")

    def test_envelope_scientific_functions_do_not_prompt(self):
        tree = ast.parse((ROOT / "build_color_envelopes.py").read_text())
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("input", called_names)
        self.assertNotIn("prompt_choice", called_names)
        self.assertNotIn("prompt_text", called_names)


if __name__ == "__main__":
    unittest.main()
