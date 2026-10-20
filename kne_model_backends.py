"""FIESTA model selection for population generation (no import-time downloads).

The semi-analytical implementation is imported from FIESTA, not copied here.
Audited against fiestaEM commit 6ba093cc13cb768a089b07768e550a6a46b737ca.
See docs/metzger_generation.md for units, simulation priors and provenance.
"""

import argparse
import hashlib
import importlib
import inspect
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

MODEL_NAMES = ("Bu2026_MLP", "Bu2019_MLP", "MetzgerModel", "MetzgerFullModel")
ANALYTICAL_MODELS = frozenset(("MetzgerModel", "MetzgerFullModel"))
METZGER_PARAMETERS = ("log10_mej", "log10_vej", "beta", "log10_kappa_r")
AUDITED_FIESTA_COMMIT = "6ba093cc13cb768a089b07768e550a6a46b737ca"


def add_metzger_arguments(parser):
    """Add explicit simulation priors and the source-frame integration grid."""
    for option, default, help_text in (
        ("log10-mej-range", (-3.0, -1.0), "log10(M_ej / solar mass)"),
        ("log10-vej-range", (math.log10(0.05), math.log10(0.3)), "log10(v_min / c)"),
        ("beta-range", (2.0, 4.0), "Positive velocity-profile exponent beta"),
        ("log10-kappa-range", (math.log10(0.5), math.log10(30.0)), "log10(kappa_r / (cm^2/g))"),
    ):
        parser.add_argument(
            "--metzger-" + option, type=float, nargs=2, default=default,
            metavar=("MIN", "MAX"), help=help_text + "; uniform in this variable",
        )
    parser.add_argument("--analytical-t-min-days", type=float, default=0.1,
                        help="First source-frame integration time (days)")
    parser.add_argument("--analytical-t-max-days", type=float, default=30.0,
                        help="Last source-frame integration time (days)")
    parser.add_argument("--analytical-n-times", type=int, default=1000,
                        help="Number of logarithmic integration times; independent of saved cadence")
    parser.add_argument("--metzger-vmax", type=float, default=0.7,
                        help="MetzgerFullModel maximum shell velocity in units of c")
    parser.add_argument("--metzger-neutron-precursor", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable the neutron precursor in MetzgerFullModel")


def metzger_prior_bounds(args):
    """Return chosen simulation bounds, not surrogate training limits."""
    return dict(zip(METZGER_PARAMETERS, (
        tuple(args.metzger_log10_mej_range), tuple(args.metzger_log10_vej_range),
        tuple(args.metzger_beta_range), tuple(args.metzger_log10_kappa_range),
    )))


def validate_model_options(args):
    if args.model not in MODEL_NAMES:
        raise ValueError(f"Unsupported FIESTA model: {args.model}")
    if args.model != "Bu2026_MLP" and args.parameter_source != "broad":
        raise ValueError(
            f"{args.model} requires --parameter-source broad. The supported "
            "Bu2026/AT2017gfo posterior has a different parameterization."
        )
    if args.model not in ANALYTICAL_MODELS:
        return
    if args.fiesta_surrogate_dir is not None:
        raise ValueError("A Metzger model needs no --fiesta-surrogate-dir or surrogate weights")
    for name, (lower, upper) in metzger_prior_bounds(args).items():
        if not all(math.isfinite(v) for v in (lower, upper)) or lower >= upper:
            raise ValueError(f"{name} requires finite MIN < MAX")
    if args.metzger_beta_range[0] <= 0:
        raise ValueError("Metzger beta must be positive")
    if args.metzger_log10_vej_range[1] >= 0:
        raise ValueError("Metzger velocities must be below c (log10_vej < 0)")
    if not (math.isfinite(args.analytical_t_min_days)
            and math.isfinite(args.analytical_t_max_days)
            and 0 < args.analytical_t_min_days < args.analytical_t_max_days):
        raise ValueError("Analytical integration requires 0 < t_min < t_max (finite, source days)")
    if args.analytical_n_times < 2:
        raise ValueError("--analytical-n-times must be at least 2")
    if args.model == "MetzgerFullModel":
        if not (0 < args.metzger_vmax < 1):
            raise ValueError("--metzger-vmax must be between 0 and 1")
        if args.metzger_log10_vej_range[1] >= math.log10(args.metzger_vmax):
            raise ValueError("MetzgerFullModel requires every v_min < vmax")


def model_configuration(args):
    """Configuration also stored in each new Metzger intrinsic catalogue."""
    if args.model not in ANALYTICAL_MODELS:
        return {}
    result = {
        "source_time_min_days": args.analytical_t_min_days,
        "source_time_max_days": args.analytical_t_max_days,
        "source_time_points": args.analytical_n_times,
        "source_time_spacing": "logarithmic",
        "inclination_used_by_model": False,
        "population_prior_bounds": {k: list(v) for k, v in metzger_prior_bounds(args).items()},
    }
    if args.model == "MetzgerFullModel":
        result.update(vmax_fraction_c=args.metzger_vmax,
                      neutron_precursor=args.metzger_neutron_precursor)
    return result


def create_fiesta_model(args, filters):
    """Instantiate only the selected backend; return model and metadata."""
    validate_model_options(args)
    try:
        installed_version = version("fiestaEM")
    except PackageNotFoundError:
        installed_version = "unavailable (source checkout?)"
    metadata = {"fiesta_version": installed_version}
    if args.model in ANALYTICAL_MODELS:
        # Explicit energy evolution benefits from double precision; retain the
        # legacy surrogate precision when a surrogate is selected instead.
        import jax
        jax.config.update("jax_enable_x64", True)
        try:
            module = importlib.import_module("fiesta.models.analytical_models")
            model_class = getattr(module, args.model)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"The installed FIESTA does not expose {args.model}. See "
                "docs/metzger_generation.md for an isolated, pinned environment."
            ) from exc
        import jax.numpy as jnp
        times = jnp.geomspace(args.analytical_t_min_days, args.analytical_t_max_days,
                             args.analytical_n_times)
        kwargs = {"filters": list(filters), "times": times}
        if args.model == "MetzgerFullModel":
            kwargs.update(neutron_precursor=args.metzger_neutron_precursor, vmax=args.metzger_vmax)
        model = model_class(**kwargs)
        if tuple(model.parameter_names) != METZGER_PARAMETERS:
            raise RuntimeError("FIESTA Metzger parameterization differs from the audited implementation")
        metadata.update(backend="FIESTA semi-analytical", api="AnalyticalModel",
                        implementation_file=inspect.getfile(model_class),
                        implementation_sha256=hashlib.sha256(
                            Path(inspect.getfile(model_class)).read_bytes()).hexdigest(),
                        audited_fiesta_commit=AUDITED_FIESTA_COMMIT,
                        configuration=model_configuration(args),
                        jax_enable_x64=True,
                        prediction_time_frame="source; generator multiplies by (1+z)")
    else:
        try:
            model_class = getattr(importlib.import_module("fiesta.inference.lightcurve_model"), "BullaFlux")
            api = "legacy BullaFlux"
        except (ImportError, AttributeError):
            model_class = getattr(importlib.import_module("fiesta.models"), "FluxSurrogate")
            api = "FluxSurrogate"
        directory = str(args.fiesta_surrogate_dir.expanduser().resolve()) if args.fiesta_surrogate_dir else None
        model = model_class(name=args.model, filters=list(filters), directory=directory)
        metadata.update(backend="FIESTA surrogate", api=api,
                        prediction_time_frame="observer (FIESTA redshift conversion)")
    return model, metadata


def saved_observer_times(times, params, model_name):
    """Analytical FIESTA returns source times; flux surrogates return observer times."""
    times = np.asarray(times, dtype=float)
    if model_name in ANALYTICAL_MODELS:
        z = float(params["redshift"])
        if not math.isfinite(z) or z < 0:
            raise ValueError("redshift must be finite and non-negative")
        return times * (1.0 + z)
    return times
