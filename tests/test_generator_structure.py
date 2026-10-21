"""Structural regression tests for the light-curve generator entry point."""
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "generate_kne_lightcurves.py"


class GeneratorStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(GENERATOR.read_text())

    def test_runtime_configuration_is_inside_main(self):
        top_level_assignments = {
            target.id
            for node in self.tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("RUNTIME", top_level_assignments)

        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        runtime_assignments = {
            target.id
            for node in ast.walk(main)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertIn("RUNTIME", runtime_assignments)

    def test_main_is_called_only_by_the_module_guard(self):
        guards = [
            node
            for node in self.tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ]
        self.assertEqual(len(guards), 1)
        calls = [
            node
            for node in ast.walk(guards[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "main"
        ]
        self.assertEqual(len(calls), 1)

    def test_rubin_initialization_is_lazy(self):
        forbidden = {
            "rubin_bandpasses",
            "get_rubin_dust_coefficients",
            "configure_runtime",
        }
        top_level_calls = {
            node.func.id
            for statement in self.tree.body
            if not isinstance(statement, (ast.FunctionDef, ast.ClassDef))
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue(forbidden.isdisjoint(top_level_calls))


if __name__ == "__main__":
    unittest.main()
