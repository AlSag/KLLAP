from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import build_color_envelopes as builder
import kne_population_io as population_io


ROOT = Path(__file__).resolve().parents[1]


class PopulationIoBoundaryTest(unittest.TestCase):
    def test_builder_keeps_compatibility_exports(self):
        self.assertIs(
            builder.SyntheticPopulationSource,
            population_io.SyntheticPopulationSource,
        )
        self.assertIs(builder.prepare_metadata, population_io.prepare_metadata)

    def test_storage_module_has_no_analysis_or_plot_dependency(self):
        tree = ast.parse((ROOT / "kne_population_io.py").read_text())
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertTrue(
            imported_roots.isdisjoint({"matplotlib", "scipy", "fiesta", "jax"})
        )

    def test_legacy_csv_source_adds_the_historical_lsst_system(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "csv").mkdir()
            pd.DataFrame({
                "t_days": [1.0, 1.0],
                "band": ["g", "r"],
                "mag": [20.0, 20.2],
            }).to_csv(root / "csv" / "lightcurve_0000.csv", index=False)
            source = population_io.SyntheticPopulationSource(root)
            self.assertEqual(source.kind, "legacy_csv_files")
            event_id, table = next(source.iter_events(["0"], ("g", "r")))
            self.assertEqual(event_id, "0")
            self.assertEqual(table["photometric_system"].tolist(), ["lsst", "lsst"])


if __name__ == "__main__":
    unittest.main()
