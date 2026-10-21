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
        called = {
            node.func.id
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("configure_runtime", called)
        self.assertIn("run_generation", called)

    def test_top_level_function_names_are_unique(self):
        names = [
            node.name
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
        ]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        self.assertEqual(duplicates, [])

    def test_public_file_contains_science_plots_and_terminal_interface(self):
        functions = {
            node.name for node in self.tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("ensure_mw_extinction_columns", functions)
        self.assertIn("plot_synthetic_only", functions)
        self.assertIn("configure_runtime", functions)
        self.assertIn("run_generation", functions)

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

    def test_module_helpers_do_not_read_run_generation_locals_as_globals(self):
        forbidden_by_function = {
            "plot_saved_synthetic_measurements": {
                "FILTER_SYSTEMS", "FILTER_BY_SYSTEM",
            },
            "apply_lsst_realism_to_fiesta": {
                "APPLY_MW_EXTINCTION_TO_LSST_REALISM",
            },
            "sample_distance_redshift_for_event": {
                "DISTANCE_SAMPLING", "FIXED_REDSHIFT",
                "FIXED_LUMINOSITY_DISTANCE_MPC",
            },
            "sample_inclination_for_event": {
                "INCLINATION_SAMPLING", "INCLINATION_MIN_RAD",
                "INCLINATION_MAX_RAD",
            },
            "add_multisystem_extinction_summary": {"FILTER_SYSTEMS"},
        }
        functions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for name, forbidden in forbidden_by_function.items():
            loaded = {
                node.id
                for node in ast.walk(functions[name])
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            self.assertTrue(forbidden.isdisjoint(loaded), name)


if __name__ == "__main__":
    unittest.main()
