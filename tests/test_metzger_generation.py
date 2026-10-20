"""Backend and generator-contract checks; no OpSim or weight downloads needed."""
import argparse
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import kne_model_backends as backend

ROOT = Path(__file__).resolve().parents[1]


def generator_functions(*names, **namespace):
    """Load selected definitions without executing the simulation at import."""
    tree = ast.parse((ROOT / "generate_kne_lightcurves.py").read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(definitions) != len(names):
        raise AssertionError("Missing generator definition")
    ns = dict(vars(backend), argparse=argparse, np=np, pd=pd, Path=Path, json=json, sys=sys)
    ns.update(namespace)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ROOT / "generate_kne_lightcurves.py"), "exec"), ns)
    return ns


def model_args(model="MetzgerModel", **updates):
    ns = generator_functions("build_runtime_parser")
    args = ns["build_runtime_parser"]().parse_args(["--model", model])
    for key, value in updates.items():
        setattr(args, key, value)
    return args


class MetzgerGenerationTest(unittest.TestCase):
    def test_four_models_and_distance_pairing_are_available(self):
        ns = generator_functions("build_runtime_parser")
        for name in backend.MODEL_NAMES:
            args = ns["build_runtime_parser"]().parse_args([
                "--model", name, "--catalog-distance-policy", "current-fixed"])
            self.assertEqual(args.model, name)
            self.assertEqual(args.catalog_distance_policy, "current-fixed")

    def test_default_bounds_and_configuration_are_explicit(self):
        args = model_args()
        backend.validate_model_options(args)
        self.assertEqual(set(backend.metzger_prior_bounds(args)), set(backend.METZGER_PARAMETERS))
        self.assertEqual(backend.model_configuration(args)["source_time_points"], 1000)
        self.assertFalse(backend.model_configuration(args)["inclination_used_by_model"])

    def test_metzger_rejects_incompatible_posterior_or_weights(self):
        for args in (model_args(parameter_source="posterior"),
                     model_args(fiesta_surrogate_dir=Path("weights"))):
            with self.assertRaises(ValueError):
                backend.validate_model_options(args)

    def test_full_model_velocity_must_be_below_vmax(self):
        args = model_args("MetzgerFullModel", metzger_vmax=0.2)
        with self.assertRaisesRegex(ValueError, "v_min < vmax"):
            backend.validate_model_options(args)

    def test_invalid_integration_grid_and_beta_are_rejected(self):
        for change in ({"analytical_n_times": 1}, {"analytical_t_min_days": 0},
                       {"analytical_t_max_days": float("nan")}, {"metzger_beta_range": (-1, 4)}):
            with self.assertRaises(ValueError):
                backend.validate_model_options(model_args(**change))

    def test_time_dilation_is_applied_only_to_analytical_outputs(self):
        source = np.array([0.1, 1.0, 10.0])
        for name in backend.ANALYTICAL_MODELS:
            np.testing.assert_allclose(backend.saved_observer_times(source, {"redshift": 0.2}, name), source * 1.2)
        for name in ("Bu2026_MLP", "Bu2019_MLP"):
            np.testing.assert_array_equal(backend.saved_observer_times(source, {"redshift": 0.2}, name), source)

    def test_model_configuration_records_precursor_and_vmax(self):
        args = model_args("MetzgerFullModel", metzger_neutron_precursor=False, metzger_vmax=0.8)
        conf = backend.model_configuration(args)
        self.assertFalse(conf["neutron_precursor"])
        self.assertEqual(conf["vmax_fraction_c"], 0.8)

    def test_catalogue_keeps_intrinsics_when_distance_is_overridden(self):
        args = model_args("MetzgerFullModel")
        conf = backend.model_configuration(args)
        cols = [*backend.METZGER_PARAMETERS, "inclination_EM", "redshift", "luminosity_distance"]
        ns = generator_functions("load_intrinsic_catalog", N_SAMPLES=1, MODEL_NAME=args.model,
                                 PARAMETER_SOURCE="broad", DISTANCE_SAMPLING="fixed", seed=12345,
                                 MODEL_CONFIGURATION=conf, INTRINSIC_PARAMETER_COLUMNS=cols,
                                 CATALOG_DISTANCE_POLICY="current-fixed", FIXED_REDSHIFT=0.124,
                                 FIXED_LUMINOSITY_DISTANCE_MPC=600.0, RUNTIME=args)
        row = dict(event_id="000000", event_index=0, model=args.model, intrinsic_seed=12345,
                   parameter_source="broad", posterior_inclination="", distance_sampling="fixed",
                   inclination_sampling="uniform_cos", posterior_index=np.nan,
                   model_configuration_json=json.dumps(conf), log10_mej=-2, log10_vej=-1,
                   beta=3, log10_kappa_r=1, inclination_EM=0.5, redshift=0.009, luminosity_distance=40.7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalogue.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            params, _ = ns["load_intrinsic_catalog"](path)[0]
            self.assertEqual(params["luminosity_distance"], 600.0)
            self.assertEqual(params["log10_mej"], -2.0)
            self.assertEqual(params["inclination_EM"], 0.5)
            row["model_configuration_json"] = json.dumps({**conf, "neutron_precursor": False})
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "model options or priors differ"):
                ns["load_intrinsic_catalog"](path)


if __name__ == "__main__":
    unittest.main()
