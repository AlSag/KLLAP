from __future__ import annotations

import ast
import unittest
from pathlib import Path

import kne_envelope_cli
import kne_generation_cli
import kne_opsim_cli


ROOT = Path(__file__).resolve().parents[1]


class CliBoundaryTest(unittest.TestCase):
    def test_cli_modules_do_not_import_heavy_scientific_stacks(self):
        forbidden = {"astropy", "fiesta", "jax", "matplotlib", "rubin_sim"}
        for filename in (
            "kne_generation_cli.py",
            "kne_opsim_cli.py",
            "kne_envelope_cli.py",
        ):
            with self.subTest(module=filename):
                tree = ast.parse((ROOT / filename).read_text())
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(alias.name.split(".", 1)[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.add(node.module.split(".", 1)[0])
                self.assertTrue(imported.isdisjoint(forbidden))

    def test_public_commands_reexport_their_interface(self):
        # Import only the lightweight commands here. The generator's optional
        # scientific dependencies are intentionally not required by this test.
        import build_color_envelopes
        import get_opsim_baseline

        self.assertIs(build_color_envelopes.build_parser, kne_envelope_cli.build_parser)
        self.assertIs(build_color_envelopes.interactive, kne_envelope_cli.interactive)
        self.assertIs(
            get_opsim_baseline.runtime_configuration,
            kne_opsim_cli.runtime_configuration,
        )

    def test_parser_contracts_remain_available_from_cli_modules(self):
        generation = kne_generation_cli.build_runtime_parser().parse_args([])
        envelope = kne_envelope_cli.build_parser().parse_args([])
        opsim = kne_opsim_cli.runtime_configuration(["--seed", "123"])
        opsim_random_seed = kne_opsim_cli.runtime_configuration([])
        self.assertEqual(generation.model, "Bu2026_MLP")
        self.assertEqual(envelope.color_pair, "g-r")
        self.assertEqual(opsim.seed, 123)
        self.assertIsInstance(opsim_random_seed.seed, int)

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
