#!/usr/bin/env python3
"""Terminal interface and runtime validation for KNe generation.

This module contains the argument parser, interactive menu, and validation of
option combinations. It does not initialize Rubin, instantiate FIESTA, draw a
population, evaluate light curves, or write a run.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np

from kne_generation_core import derive_sky_seed
from kne_model_backends import (
    ANALYTICAL_MODELS,
    MODEL_NAMES,
    add_metzger_arguments,
    validate_model_options,
)
from kne_pipeline_common import prompt_choice, prompt_text


def build_runtime_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without parsing or prompting.

    Running the script with no arguments starts the interactive menu.  Passing
    one or more arguments runs non-interactively unless ``--interactive`` is
    also supplied.  ``--help`` therefore provides a complete reproducible
    alternative to the terminal menu.
    """
    parser = argparse.ArgumentParser(
        description="Generate broad-uniform or posterior-based KNe light curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-name",
        default="broad_run",
        help="Readable label used in the output-directory name",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_NAMES),
        default="Bu2026_MLP",
        help=(
            "FIESTA flux surrogate or spherical semi-analytical Metzger model. "
            "Metzger models require the analytical_models module, not weights."
        ),
    )
    parser.add_argument(
        "--fiesta-surrogate-dir",
        type=Path,
        help=(
            "Optional directory containing <model>.pkl and "
            "<model>_metadata.pkl. When omitted, FIESTA uses its built-in "
            "directory or downloads the selected surrogate."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Parent directory in which the new run directory is created",
    )
    parser.add_argument(
        "--filter-system",
        choices=["lsst", "ps1", "ztf", "lsst+ps1", "lsst+ztf"],
        default="lsst+ps1",
        help=(
            "Generate one survey system, or evaluate LSST together with PS1 "
            "or ZTF for the same physical event, distance, merger epoch and "
            "OpSim field. Standalone PS1/ZTF modes are synthetic-only; ZTF "
            "depth cuts are applied later by build_color_envelopes.py."
        ),
    )
    parser.add_argument(
        "--parameter-source",
        choices=["broad", "posterior"],
        default="broad",
        help="Use broad uniform priors or complete rows from a FIESTA posterior",
    )
    parser.add_argument(
        "--posterior-file",
        type=Path,
        help="posterior.npz or a tar archive containing posterior.npz",
    )
    parser.add_argument(
        "--posterior-inclination",
        choices=["posterior", "isotropic"],
        default="posterior",
        help="Keep the joint posterior inclination or replace it by an isotropic draw",
    )
    parser.add_argument(
        "--opsim-dir",
        type=Path,
        default=Path("opsim_exports_all_survey"),
        help=(
            "Directory containing opsim_visits_field*.csv and the m5 summary "
            "written by get_opsim_baseline.py"
        ),
    )
    parser.add_argument(
        "--galactic-latitude-mode",
        choices=["all", "high", "low", "bin"],
        default="all",
        help=(
            "Restrict OpSim field selection: all fields, |b| greater than or "
            "equal to the threshold, |b| below the threshold, or a disjoint "
            "interval min <= |b| < max"
        ),
    )
    parser.add_argument(
        "--galactic-latitude-threshold-deg",
        type=float,
        default=20.0,
        help="Absolute Galactic-latitude boundary used by high/low modes",
    )
    parser.add_argument(
        "--galactic-latitude-min-deg",
        type=float,
        default=0.0,
        help="Inclusive lower |b| edge used by latitude-bin mode",
    )
    parser.add_argument(
        "--galactic-latitude-max-deg",
        type=float,
        default=5.0,
        help="Exclusive upper |b| edge used by latitude-bin mode (90 is inclusive)",
    )
    parser.add_argument(
        "--output-mode",
        choices=[
            "all-synthetic-lsstlike-detected",
            "detected-pairs",
            "lsstlike-detected",
            "synthetic-mjd",
        ],
        default="detected-pairs",
        help=(
            "all-synthetic-lsstlike-detected: all synthetic + LSST-like for detected events; "
            "detected-pairs: synthetic + LSST-like for detected events; "
            "lsstlike-detected: LSST-like only for detected events; "
            "synthetic-mjd: synthetic only with OpSim context and absolute MJD"
        ),
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=1,
        help="Minimum detected=True points required to save an LSST-like file",
    )
    parser.add_argument(
        "--synthetic-cadence-days",
        type=float,
        default=0.0,
        help=(
            "Cadence of saved synthetic CSVs in observer-frame days; "
            "0 preserves the native FIESTA grid"
        ),
    )
    parser.add_argument(
        "--synthetic-depth-cut",
        choices=["none", "m5-median"],
        default="none",
        help=(
            "Optional band-dependent cut on saved synthetic points using the "
            "precomputed median fiveSigmaDepth from get_opsim_baseline.py"
        ),
    )
    parser.add_argument(
        "--storage-format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help=(
            "Persist eligible per-event products as legacy CSV files, as "
            "sharded Parquet datasets, or in both forms"
        ),
    )
    parser.add_argument(
        "--parquet-events-per-shard",
        type=int,
        default=500,
        help="Number of events buffered in each Parquet shard",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=["zstd", "snappy", "gzip", "none"],
        default="zstd",
        help="Compression codec used for Parquet shards",
    )
    parser.add_argument(
        "--csv-sample-size",
        type=int,
        default=0,
        help=(
            "Number of injected event IDs selected reproducibly for legacy "
            "per-event CSV output; 0 allows every eligible event"
        ),
    )
    parser.add_argument(
        "--opsim-cache-size",
        type=int,
        default=32,
        help="Maximum number of fully prepared OpSim field tables kept in memory",
    )
    parser.add_argument(
        "--surrogate-batch-size",
        type=int,
        default=64,
        help=(
            "Requested FIESTA prediction batch size. The first batch is "
            "checked against scalar predictions; unsupported or inconsistent "
            "batch APIs fall back automatically to scalar evaluation"
        ),
    )
    parser.add_argument(
        "--plot-mode",
        choices=["none", "saved"],
        default="none",
        help=(
            "Write PNG diagnostics for the light-curve products saved by the "
            "selected output mode"
        ),
    )
    parser.add_argument(
        "--max-plot-events",
        type=int,
        default=20,
        help="Maximum saved events plotted per product; 0 plots every saved event",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=80000,
        help="Number of independent kilonova events to inject",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=1,
        help=(
            "Number of independent maximum-size populations generated with "
            "the same scientific configuration. Each realization is written "
            "to its own run directory."
        ),
    )
    parser.add_argument(
        "--realization-seeds",
        help=(
            "Optional comma-separated intrinsic seeds, one per realization. "
            "When omitted, reproducible child seeds are derived from --seed."
        ),
    )
    parser.add_argument(
        "--pairing-mode",
        choices=["none", "create", "reuse"],
        default="none",
        help=(
            "none: draw an ordinary independent population; create: draw and "
            "save a reusable intrinsic-population catalogue; reuse: load an "
            "existing catalogue so event_id identifies the same physical KNe "
            "in several sky/latitude runs"
        ),
    )
    parser.add_argument(
        "--intrinsic-catalog",
        type=Path,
        help=(
            "CSV catalogue created or reused by paired runs. Required when "
            "--pairing-mode is create or reuse"
        ),
    )
    parser.add_argument(
        "--catalog-distance-policy", choices=["stored", "current-fixed"], default="stored",
        help="Reuse stored catalogue distances, or replace only distance/redshift by this run's fixed distance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Intrinsic-population seed. Use the same value for every member "
            "of a paired family; random when omitted"
        ),
    )
    parser.add_argument(
        "--sky-seed",
        type=int,
        help=(
            "Independent seed for OpSim field, merger epoch and photometric "
            "noise. Use different values for low/high runs. Defaults to --seed"
        ),
    )
    parser.add_argument(
        "--distance-mode",
        choices=["comoving-volume", "fixed"],
        default="comoving-volume",
        help="Draw distance uniformly in comoving volume or use one fixed luminosity distance",
    )
    parser.add_argument(
        "--fixed-distance-mpc",
        type=float,
        default=10.0,
        help="Fixed luminosity distance used when --distance-mode fixed",
    )
    parser.add_argument(
        "--distance-min-mpc",
        type=float,
        default=100.0,
        help="Minimum luminosity distance for comoving-volume draws",
    )
    parser.add_argument(
        "--distance-max-mpc",
        type=float,
        default=500.0,
        help="Maximum luminosity distance for comoving-volume draws",
    )
    parser.add_argument(
        "--numerical-floor-absolute-mag",
        type=float,
        default=0.0,
        help="Approximate absolute-magnitude floor of Bu2026",
    )
    parser.add_argument(
        "--numerical-floor-margin-mag",
        type=float,
        default=1.0,
        help="Safety margin applied before the numerical floor",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show the terminal menu even when other arguments are supplied",
    )
    parser.add_argument("--runtime-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--single-realization-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    add_metzger_arguments(parser)
    return parser


def configure_runtime() -> argparse.Namespace:
    """Parse options, optionally prompt the user, and validate combinations."""
    parser = build_runtime_parser()
    parsed = parser.parse_args()
    if parsed.runtime_config is not None:
        payload = json.loads(parsed.runtime_config.read_text())
        path_fields = {
            action.dest for action in parser._actions if action.type is Path
        }
        for field in path_fields:
            if payload.get(field) is not None:
                payload[field] = Path(payload[field])
        args = argparse.Namespace(**payload)
        args.runtime_config = parsed.runtime_config
        args.single_realization_worker = True
        interactive = False
    else:
        args = parsed
        interactive = bool(args.interactive or len(sys.argv) == 1)
    # This VM-oriented generator stores only MW-extincted synthetic magnitudes.
    args.synthetic_schema = "compact-extincted"
    if interactive:
        args.run_name = prompt_text("Readable run name", args.run_name)
        args.model = prompt_choice(
            "Kilonova light-curve model",
            [
                ("Bu2026_MLP", "FIESTA Bu2026_MLP (LSST, PS1 and ZTF)"),
                ("Bu2019_MLP", "FIESTA Bu2019_MLP (LSST, PS1 and ZTF)"),
                ("MetzgerModel", "FIESTA MetzgerModel (semi-analytical, NMMA-style)"),
                ("MetzgerFullModel", "FIESTA MetzgerFullModel (semi-analytical, Redback-style)"),
            ],
            args.model,
        )
        if args.model == "Bu2019_MLP":
            surrogate_path = prompt_text(
                "Local FIESTA surrogate directory (empty = automatic download)",
                str(args.fiesta_surrogate_dir or ""),
            ).strip()
            args.fiesta_surrogate_dir = (
                Path(surrogate_path) if surrogate_path else None
            )
        if args.model in ANALYTICAL_MODELS:
            print("Metzger population priors: uniform in the listed variables.")
            for attribute, label in (
                ("metzger_log10_mej_range", "log10(M_ej / Msun)"),
                ("metzger_log10_vej_range", "log10(v_min / c)"),
                ("metzger_beta_range", "Velocity-profile exponent beta"),
                ("metzger_log10_kappa_range", "log10(kappa_r / (cm^2/g))"),
            ):
                value = getattr(args, attribute)
                entered = prompt_text(label + " range (MIN MAX)", f"{value[0]} {value[1]}")
                limits = [float(v) for v in entered.replace(",", " ").split()]
                if len(limits) != 2:
                    raise ValueError(label + " requires MIN MAX")
                setattr(args, attribute, limits)
            args.analytical_t_min_days = float(prompt_text(
                "First source-frame integration time [days]", str(args.analytical_t_min_days)))
            args.analytical_t_max_days = float(prompt_text(
                "Last source-frame integration time [days]", str(args.analytical_t_max_days)))
            args.analytical_n_times = int(prompt_text(
                "Logarithmic integration time points (not the saved cadence)", str(args.analytical_n_times)))
            if args.model == "MetzgerFullModel":
                args.metzger_vmax = float(prompt_text(
                    "Maximum shell velocity [fraction of c]", str(args.metzger_vmax)))
                precursor = prompt_choice("Neutron precursor", [("yes", "Include"), ("no", "Disable")],
                                          "yes" if args.metzger_neutron_precursor else "no")
                args.metzger_neutron_precursor = precursor == "yes"
        args.parameter_source = prompt_choice(
            "Physical-parameter population",
            ([("broad", "Broad uniform parameter ranges"), ("posterior", "FIESTA posterior (.npz or AT2017gfo.tar)")]
             if args.model == "Bu2026_MLP" else [("broad", "Broad uniform parameter ranges")]),
            args.parameter_source,
        )
        if args.parameter_source == "posterior":
            args.posterior_file = Path(prompt_text("Posterior file", str(args.posterior_file or "AT2017gfo.tar")))
            args.posterior_inclination = prompt_choice(
                "Inclination treatment",
                [("posterior", "Keep inclination from each posterior row"), ("isotropic", "AT2017gfo-like ejecta, new isotropic inclination")],
                args.posterior_inclination,
            )
        args.filter_system = prompt_choice(
            "Photometric filter system",
            [
                ("lsst+ps1", "LSST ugrizy + PS1 grizy for the same events"),
                ("lsst+ztf", "LSST ugrizy + ZTF gri for the same events"),
                ("lsst", "LSST ugrizy only"),
                ("ps1", "PS1 grizy only (synthetic products; no Rubin m5)"),
                ("ztf", "ZTF gri only (synthetic products; ZTF depth applied later)"),
            ],
            args.filter_system,
        )
        if args.filter_system in {"ps1", "ztf"}:
            # Rubin OpSim visits and m5 values cannot be applied to a PS1-only
            # model evaluation. We retain the OpSim sky position, merger MJD
            # and E(B-V) metadata, but save only synthetic PS1 products.
            args.output_mode = "synthetic-mjd"
            print(
                f"{args.filter_system.upper()}-only mode: scientific product "
                "selection is fixed to 'Synthetic OpSim RA/Dec/MJD only'. "
                "Rubin LSST-like products require LSST model magnitudes."
            )
        else:
            args.output_mode = prompt_choice(
                "Scientific event-product selection",
                [
                    (
                        "all-synthetic-lsstlike-detected",
                        "All synthetic + LSST-like only for detected events",
                    ),
                    (
                        "detected-pairs",
                        "Synthetic OpSim MJD + LSST-like, detected events only",
                    ),
                    (
                        "lsstlike-detected",
                        "LSST-like only, detected events only",
                    ),
                    (
                        "synthetic-mjd",
                        "Synthetic OpSim RA/Dec/MJD only",
                    ),
                ],
                args.output_mode,
            )
        if args.output_mode in {
            "all-synthetic-lsstlike-detected",
            "detected-pairs",
            "synthetic-mjd",
        }:
            args.synthetic_cadence_days = float(
                prompt_text(
                    "Saved synthetic cadence [days] (0 = native FIESTA grid)",
                    str(args.synthetic_cadence_days),
                )
            )
            if args.filter_system in {"ps1", "ztf"}:
                args.synthetic_depth_cut = "none"
                print(
                    "Synthetic depth cut: none (survey depth is applied by "
                    "build_color_envelopes.py, not during generation)."
                )
            else:
                args.synthetic_depth_cut = prompt_choice(
                    "Synthetic depth cut",
                    [
                        ("none", "Keep every finite synthetic model point"),
                        (
                            "m5-median",
                            "Keep points brighter than the median OpSim m5 of their band",
                        ),
                    ],
                    args.synthetic_depth_cut,
                )
        else:
            args.synthetic_cadence_days = 0.0
            args.synthetic_depth_cut = "none"
        args.storage_format = prompt_choice(
            "Scientific table storage",
            [
                ("csv", "One legacy CSV per eligible event"),
                ("parquet", "Sharded Parquet datasets only"),
                ("both", "Sharded Parquet + a legacy CSV event sample"),
            ],
            args.storage_format,
        )
        if args.storage_format in {"parquet", "both"}:
            args.parquet_events_per_shard = int(prompt_text(
                "Events per Parquet shard",
                str(args.parquet_events_per_shard),
            ))
            args.parquet_compression = prompt_choice(
                "Parquet compression",
                [
                    ("zstd", "Zstandard: compact and fast [recommended]"),
                    ("snappy", "Snappy: fastest, less compact"),
                    ("gzip", "Gzip: compact, slower"),
                    ("none", "No compression"),
                ],
                args.parquet_compression,
            )
        if args.storage_format in {"csv", "both"}:
            args.csv_sample_size = int(prompt_text(
                "Random injected events eligible for individual CSVs (0 = all)",
                str(args.csv_sample_size),
            ))
        args.opsim_cache_size = int(prompt_text(
            "OpSim field tables kept in memory",
            str(args.opsim_cache_size),
        ))
        args.surrogate_batch_size = int(prompt_text(
            "Requested surrogate batch size (automatic validated fallback)",
            str(args.surrogate_batch_size),
        ))
        args.plot_mode = prompt_choice(
            "Diagnostic light-curve plots",
            [
                ("none", "Do not create PNG plots"),
                (
                    "saved",
                    "Plot each saved synthetic and/or LSST-like product",
                ),
            ],
            args.plot_mode,
        )
        if args.plot_mode == "saved":
            args.max_plot_events = int(
                prompt_text(
                    "Maximum saved events plotted per product (0 = all)",
                    str(args.max_plot_events),
                )
            )
        args.distance_mode = prompt_choice(
            "Luminosity-distance treatment",
            [
                ("comoving-volume", "Direct draw uniform in comoving volume"),
                ("fixed", "Fixed luminosity distance"),
            ],
            args.distance_mode,
        )
        if args.distance_mode == "fixed":
            args.fixed_distance_mpc = float(
                prompt_text("Fixed luminosity distance [Mpc]", str(args.fixed_distance_mpc))
            )
        if args.model == "Bu2026_MLP":
            args.numerical_floor_absolute_mag = float(
                prompt_text(
                    "Bu2026 numerical floor in absolute magnitude",
                    str(args.numerical_floor_absolute_mag),
                )
            )
            args.numerical_floor_margin_mag = float(
                prompt_text(
                    "Safety margin before numerical floor [mag]",
                    str(args.numerical_floor_margin_mag),
                )
            )
        args.opsim_dir = Path(prompt_text("OpSim export directory", str(args.opsim_dir)))
        args.galactic_latitude_mode = prompt_choice(
            "OpSim-field Galactic-latitude selection",
            [
                ("all", "All available Galactic latitudes"),
                ("high", "High latitude: |b| >= threshold"),
                ("low", "Low latitude: |b| < threshold"),
                ("bin", "Disjoint interval: min <= |b| < max"),
            ],
            args.galactic_latitude_mode,
        )
        if args.galactic_latitude_mode in {"low", "high"}:
            args.galactic_latitude_threshold_deg = float(
                prompt_text(
                    "Absolute Galactic-latitude threshold [deg]",
                    str(args.galactic_latitude_threshold_deg),
                )
            )
        elif args.galactic_latitude_mode == "bin":
            args.galactic_latitude_min_deg = float(
                prompt_text(
                    "Inclusive lower |b| edge [deg]",
                    str(args.galactic_latitude_min_deg),
                )
            )
            args.galactic_latitude_max_deg = float(
                prompt_text(
                    "Exclusive upper |b| edge [deg]",
                    str(args.galactic_latitude_max_deg),
                )
            )
        if args.output_mode in {
            "all-synthetic-lsstlike-detected",
            "detected-pairs",
            "lsstlike-detected",
        }:
            args.min_detections = int(
                prompt_text("Minimum number of LSST-like detections", str(args.min_detections))
            )
        args.n_samples = int(prompt_text("Number of simulated events", str(args.n_samples)))
        args.pairing_mode = prompt_choice(
            "Intrinsic-population pairing",
            [
                (
                    "none",
                    "Independent population (legacy behaviour)",
                ),
                (
                    "create",
                    "Create a reusable intrinsic KNe catalogue",
                ),
                (
                    "reuse",
                    "Reuse an intrinsic KNe catalogue for a paired sky run",
                ),
            ],
            args.pairing_mode,
        )
        if args.pairing_mode != "none":
            args.intrinsic_catalog = Path(
                prompt_text(
                    "Shared intrinsic catalogue",
                    str(args.intrinsic_catalog or "paired_intrinsic_population.csv"),
                )
            )
        if args.pairing_mode == "reuse" and args.distance_mode == "fixed":
            args.catalog_distance_policy = prompt_choice(
                "Distance used when reusing the intrinsic catalogue",
                [("current-fixed", "Use the fixed distance entered for this run"),
                 ("stored", "Keep the distance and redshift stored in the catalogue")],
                args.catalog_distance_policy,
            )
        args.n_realizations = int(
            prompt_text(
                "Number of independent population realizations",
                str(args.n_realizations),
            )
        )
        seed_text = prompt_text(
            (
                "Master intrinsic-population seed (integer)"
                if args.n_realizations > 1
                else "Intrinsic-population seed (integer)"
                if args.pairing_mode != "none"
                else "Random seed (integer)"
            ),
            str(args.seed if args.seed is not None else secrets.randbits(32)),
        )
        args.seed = int(seed_text)
        if args.pairing_mode != "none":
            default_sky_seed = derive_sky_seed(
                args.seed,
                args.galactic_latitude_mode,
                args.galactic_latitude_threshold_deg,
                args.galactic_latitude_max_deg,
            )
            args.sky_seed = int(
                prompt_text(
                    "Sky/noise seed (same = paired sky; different = new sky)",
                    str(
                        args.sky_seed
                        if args.sky_seed is not None
                        else default_sky_seed
                    ),
                )
            )
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.n_realizations <= 0:
        raise ValueError("--n-realizations must be positive")
    if args.n_realizations > 1 and args.pairing_mode != "none":
        raise ValueError(
            "Multiple independent realizations require --pairing-mode none. "
            "Paired catalogues serve a different low/high-sky comparison."
        )
    validate_model_options(args)
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.sky_seed is not None and args.sky_seed < 0:
        raise ValueError("--sky-seed must be non-negative")
    if args.min_detections <= 0:
        raise ValueError("--min-detections must be positive")
    if args.fixed_distance_mpc <= 0:
        raise ValueError("--fixed-distance-mpc must be positive")
    if args.distance_min_mpc <= 0:
        raise ValueError("--distance-min-mpc must be positive")
    if args.distance_max_mpc <= args.distance_min_mpc:
        raise ValueError("--distance-max-mpc must exceed --distance-min-mpc")
    if not np.isfinite(args.numerical_floor_absolute_mag):
        raise ValueError("--numerical-floor-absolute-mag must be finite")
    if not np.isfinite(args.numerical_floor_margin_mag):
        raise ValueError("--numerical-floor-margin-mag must be finite")
    if args.numerical_floor_margin_mag < 0:
        raise ValueError("--numerical-floor-margin-mag must be >= 0")
    if args.synthetic_cadence_days < 0:
        raise ValueError("--synthetic-cadence-days must be >= 0")
    if args.max_plot_events < 0:
        raise ValueError("--max-plot-events must be >= 0")
    if args.parquet_events_per_shard < 1:
        raise ValueError("--parquet-events-per-shard must be >= 1")
    if args.csv_sample_size < 0:
        raise ValueError("--csv-sample-size must be >= 0")
    if args.opsim_cache_size < 0:
        raise ValueError("--opsim-cache-size must be >= 0")
    if args.surrogate_batch_size < 1:
        raise ValueError("--surrogate-batch-size must be >= 1")
    if args.pairing_mode == "none" and args.intrinsic_catalog is not None:
        raise ValueError(
            "--intrinsic-catalog requires --pairing-mode create or reuse"
        )
    if args.pairing_mode in {"create", "reuse"} and args.intrinsic_catalog is None:
        raise ValueError(
            "--intrinsic-catalog is required with --pairing-mode create or reuse"
        )
    if args.pairing_mode == "create" and args.intrinsic_catalog.expanduser().exists():
        raise FileExistsError(
            f"Intrinsic catalogue already exists: {args.intrinsic_catalog}. "
            "Select pairing mode 'reuse' or choose another path."
        )
    if args.pairing_mode == "reuse" and not args.intrinsic_catalog.expanduser().is_file():
        raise FileNotFoundError(
            f"Intrinsic catalogue not found: {args.intrinsic_catalog}"
        )
    if args.catalog_distance_policy == "current-fixed":
        if args.pairing_mode != "reuse" or args.distance_mode != "fixed":
            raise ValueError("--catalog-distance-policy current-fixed requires --pairing-mode reuse and --distance-mode fixed")
    if args.filter_system in {"ps1", "ztf"} and args.output_mode != "synthetic-mjd":
        raise ValueError(
            f"--filter-system {args.filter_system} supports only --output-mode "
            "synthetic-mjd. LSST-like products require LSST model magnitudes; "
            "use a combined lsst+... system when Rubin products are needed."
        )
    if args.storage_format in {"parquet", "both"}:
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Parquet output requires pyarrow. Install it with "
                "`python -m pip install pyarrow` or select --storage-format csv."
            ) from exc
    if not 0.0 <= args.galactic_latitude_threshold_deg <= 90.0:
        raise ValueError("--galactic-latitude-threshold-deg must lie in [0, 90]")
    if not 0.0 <= args.galactic_latitude_min_deg < args.galactic_latitude_max_deg <= 90.0:
        raise ValueError(
            "Latitude-bin edges must satisfy 0 <= min < max <= 90 deg"
        )
    if args.synthetic_depth_cut != "none":
        if args.filter_system != "lsst":
            raise ValueError(
                "The generator-level m5 cut is LSST-only. For a paired "
                "LSST+PS1 or LSST+ZTF run select 'none'; "
                "build_color_envelopes.py applies the appropriate survey "
                "depth consistently after generation."
            )
        if args.output_mode == "lsstlike-detected":
            raise ValueError(
                "--synthetic-depth-cut applies only to modes that save a "
                "synthetic CSV: all-synthetic-lsstlike-detected, "
                "detected-pairs or synthetic-mjd"
            )
    if (
        args.parameter_source == "posterior"
        and args.pairing_mode != "reuse"
        and args.posterior_file is None
    ):
        raise ValueError("--posterior-file is required when --parameter-source posterior")
    return args


