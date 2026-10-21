#!/usr/bin/env python3
"""Terminal interface for colour-envelope construction.

This module owns argument definitions and interactive prompts only. It does
not read populations, calculate colours or percentiles, apply depth cuts, or
create scientific products.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kllap.colour.common import prompt_choice, prompt_text
from kllap.envelope.core import DEFAULT_TAIL_GRID
from kllap.m5 import M5_SCENARIO_LABELS

ZTF_DEPTH_FILENAMES = {
    "g": ("lims_public_g.joblib", "lims_g.joblib"),
    "r": ("lims_public_r.joblib", "lims_r.joblib"),
    "i": ("lims_i.joblib", "lims_public_i.joblib"),
}
AT2017GFO_DEFAULT_MERGER_MJD = 57982.52851852


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    parser.add_argument("--color-pair", default="g-r")
    parser.add_argument(
        "--photometric-system",
        choices=["lsst", "ps1", "ztf", "both", "lsst+ztf"],
        default="both",
        help=(
            "Envelope system. 'both' preserves the historical LSST+PS1 mode; "
            "'lsst+ztf' builds both envelopes from the same event IDs and "
            "writes direct comparison plots. ZTF depth can be entered directly, "
            "read from a quantile table, or loaded from legacy KDE files."
        ),
    )
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--t-max", type=float, default=26.0)
    parser.add_argument("--bin-width", type=float, default=0.4)
    parser.add_argument("--tail-percentiles", default=DEFAULT_TAIL_GRID)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-threshold-mag", type=float, default=0.10)
    parser.add_argument("--minimum-events", type=int, default=30)
    parser.add_argument("--minimum-tail-events", type=float, default=20.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--event-subsample-seed",
        type=int,
        help=(
            "When --max-events is positive, randomly select that many events "
            "without replacement. Reusing one seed with increasing sample "
            "sizes produces nested subsets. If omitted, preserve the legacy "
            "first-event-ID selection."
        ),
    )
    parser.add_argument(
        "--colored-plot-sizes",
        default="10,100",
        help="Nested random KNe sample sizes used for coloured trajectory plots.",
    )
    parser.add_argument("--population-plot-seed", type=int, default=20260921)
    parser.add_argument(
        "--outlier-table-plot-size",
        type=int,
        default=10,
        help=(
            "Coloured population sample whose permanent exits are retained "
            "in the CSV and PNG tables."
        ),
    )
    parser.add_argument(
        "--minimum-final-outside-bins",
        type=int,
        default=3,
        help=(
            "Minimum number of consecutive comparable bins in the final "
            "outside-envelope run required to call a permanent exit."
        ),
    )
    parser.add_argument(
        "--max-outlier-table-events",
        type=int,
        default=20,
        help="Maximum number of permanent exits displayed in each PNG table.",
    )
    parser.add_argument("--m5-quantile-table", type=Path)
    parser.add_argument(
        "--ztf-depth-source",
        choices=["manual", "table", "joblib"],
        help=(
            "Source of the ZTF limiting magnitudes. 'manual' uses --ztf-m5-g/r/i; "
            "'table' reads --ztf-depth-quantile-table; 'joblib' keeps the legacy "
            "NMMA KDE input. If omitted in non-interactive mode, the source is "
            "inferred from the supplied arguments."
        ),
    )
    parser.add_argument(
        "--ztf-m5-g",
        type=float,
        help="User-supplied fixed ZTF g-band 5-sigma limiting magnitude (AB mag).",
    )
    parser.add_argument(
        "--ztf-m5-r",
        type=float,
        help="User-supplied fixed ZTF r-band 5-sigma limiting magnitude (AB mag).",
    )
    parser.add_argument(
        "--ztf-m5-i",
        type=float,
        help="User-supplied fixed ZTF i-band 5-sigma limiting magnitude (AB mag).",
    )
    parser.add_argument(
        "--ztf-depth-dir",
        type=Path,
        help=(
            "Directory containing trusted lims_public_g.joblib, "
            "lims_public_r.joblib and lims_i.joblib ZTF depth KDEs. Joblib "
            "files are pickle-based: never load files from an untrusted source."
        ),
    )
    parser.add_argument(
        "--ztf-depth-quantile-table",
        type=Path,
        help=(
            "Optional CSV with columns band, scenario, m5. When supplied it "
            "is used instead of loading the ZTF joblib files."
        ),
    )
    parser.add_argument(
        "--ztf-depth-samples",
        type=int,
        default=200000,
        help="Monte-Carlo draws per ZTF-band KDE used to estimate p25/p50/p75.",
    )
    parser.add_argument(
        "--ztf-depth-seed",
        type=int,
        default=20261001,
        help="Random seed used only for ZTF depth-distribution sampling.",
    )
    parser.add_argument(
        "--threshold-scope",
        choices=["common_global", "latitude_conditioned"],
        default="common_global",
    )
    parser.add_argument(
        "--m5-scenario",
        choices=list(M5_SCENARIO_LABELS),
        default="median_p50",
    )
    parser.add_argument(
        "--population-latitude-selection",
        choices=["all", "high", "low"],
        default="all",
        help=(
            "Latitude selection applied to the injected KNe population: all "
            "events, |b| >= threshold, or |b| < threshold. This is independent "
            "of --threshold-scope, which controls the m5 quantile table."
        ),
    )
    parser.add_argument(
        "--latitude-threshold-deg",
        type=float,
        default=20.0,
        help="Absolute Galactic-latitude threshold for high/low population selection.",
    )
    parser.add_argument(
        "--assume-latitude-selection-already-applied",
        action="store_true",
        help=(
            "Allow high/low selection when summary.csv has no latitude column, "
            "only if the input run was already generated with that selection."
        ),
    )
    parser.add_argument(
        "--assume-high-latitude-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--at2017gfo-file",
        type=Path,
        help=(
            "Whitespace-delimited AT2017gfo photometry: UTC date, filter, "
            "magnitude, magnitude uncertainty. Omit to skip the overlay."
        ),
    )
    parser.add_argument(
        "--at2017gfo-merger-mjd",
        type=float,
        default=AT2017GFO_DEFAULT_MERGER_MJD,
        help="Reference merger epoch used to compute observer-frame phases.",
    )
    parser.add_argument(
        "--at2017gfo-match-window-days",
        type=float,
        default=0.4,
        help="Maximum separation used to pair the two observed bands.",
    )
    parser.add_argument(
        "--at2017gfo-ebv",
        type=float,
        default=0.105,
        help="Milky-Way E(B-V) toward AT2017gfo.",
    )
    parser.add_argument(
        "--at2017gfo-error-sigma",
        type=float,
        default=1.0,
        help=(
            "Number of photometric standard deviations used when deciding "
            "whether an observed colour is compatible with the envelope."
        ),
    )
    parser.add_argument(
        "--at2017gfo-photometry-mode",
        choices=["dereddened", "mw_extincted"],
        default="dereddened",
        help=(
            "Whether the supplied magnitudes have already been corrected for "
            "Milky-Way extinction."
        ),
    )
    parser.add_argument(
        "--at2017gfo-bestfit-file",
        type=Path,
        help=(
            "Optional FIESTA bestfit_params.pkl containing light_curves/times "
            "and per-filter model magnitudes. Its model colour is overlaid "
            "without photometric-error pass/fail classification."
        ),
    )
    parser.add_argument(
        "--at2017gfo-bestfit-photometry-mode",
        choices=["dereddened", "mw_extincted"],
        default="dereddened",
        help=(
            "Whether the model magnitudes saved in --at2017gfo-bestfit-file "
            "have already been corrected for Milky-Way extinction."
        ),
    )
    parser.add_argument(
        "--snia-parquet",
        type=Path,
        help=(
            "Optional long-format LSST SN Ia parquet. When supplied, SN Ia "
            "colour populations and KNe--SN Ia overlap products are written."
        ),
    )
    parser.add_argument(
        "--snia-magnitude-column",
        choices=["mag_perfect", "mag"],
        default="mag_perfect",
        help=(
            "SN Ia magnitude used for colours. mag_perfect is recommended "
            "for comparison with the noise-free KNe population."
        ),
    )
    parser.add_argument(
        "--snia-input-mw-mode",
        choices=["dereddened", "mw_extincted"],
        default="dereddened",
        help=(
            "Whether the input SN Ia magnitude already contains foreground "
            "Milky-Way extinction. The supplied mock is documented without "
            "MW columns and is therefore treated as dereddened by default."
        ),
    )
    parser.add_argument(
        "--snia-time-reference",
        choices=["peak", "first_observation"],
        default="peak",
        help=(
            "SN Ia time origin: SALT2 t0 (peak) or the first available MJD "
            "of each SN in either requested colour band."
        ),
    )
    parser.add_argument(
        "--snia-position-seed",
        type=int,
        default=20260929,
        help="Seed for deterministic assignment of KNe sky positions to SNe Ia.",
    )
    parser.add_argument(
        "--snia-max-events",
        type=int,
        default=0,
        help="Maximum number of SNe Ia; 0 keeps every object.",
    )
    parser.add_argument(
        "--write-adapted-snia-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write the selected-band SN Ia measurements in the pipeline's "
            "long format with projected sky coordinates and both MW conventions."
        ),
    )
    parser.add_argument("--interactive", action="store_true")
    return parser


def interactive(args: argparse.Namespace) -> argparse.Namespace:
    args.run_dir = Path(prompt_text("Run directory", str(args.run_dir)))
    args.color_pair = prompt_text("Colour pair", args.color_pair)
    args.photometric_system = prompt_choice(
        "Photometric system for the colour envelope",
        [
            ("both", "LSST and PS1 from the same synthetic events"),
            ("lsst+ztf", "LSST and ZTF from the same synthetic events"),
            ("lsst", "LSST only"),
            ("ps1", "PS1 colour with Rubin/LSST m5 selection"),
            ("ztf", "ZTF colour with a supplied ZTF depth selection"),
        ],
        args.photometric_system,
    )
    args.population_latitude_selection = prompt_choice(
        "Galactic-latitude selection for the KNe population",
        [
            ("all", "Use all generated Galactic latitudes (no latitude cut)"),
            ("high", "Keep events with |b| >= a configurable threshold"),
            ("low", "Keep events with |b| < a configurable threshold"),
        ],
        args.population_latitude_selection,
    )
    if args.population_latitude_selection != "all":
        args.latitude_threshold_deg = float(prompt_text(
            "Absolute Galactic-latitude threshold |b| [deg]",
            str(args.latitude_threshold_deg),
        ))
    args.t_min = float(prompt_text("First temporal-bin lower edge [days]", str(args.t_min)))
    args.t_max = float(prompt_text("Last temporal-bin upper edge [days]", str(args.t_max)))
    args.bin_width = float(prompt_text("Fixed bin width [days]", str(args.bin_width)))
    args.n_boot = int(prompt_text("Bootstrap repetitions", str(args.n_boot)))
    args.minimum_tail_events = float(prompt_text(
        "Minimum expected events in each percentile tail",
        str(args.minimum_tail_events),
    ))
    args.bootstrap_threshold_mag = float(prompt_text(
        "Maximum 68% bootstrap half-width per bound [mag]",
        str(args.bootstrap_threshold_mag),
    ))
    args.colored_plot_sizes = prompt_text(
        "Random coloured KNe sample sizes",
        args.colored_plot_sizes,
    )
    args.population_plot_seed = int(prompt_text(
        "Population-plot random seed",
        str(args.population_plot_seed),
    ))
    args.minimum_final_outside_bins = int(prompt_text(
        "Minimum consecutive final outside bins",
        str(args.minimum_final_outside_bins),
    ))
    args.max_outlier_table_events = int(prompt_text(
        "Maximum permanent exits in each PNG table",
        str(args.max_outlier_table_events),
    ))
    if args.photometric_system in {"ztf", "lsst+ztf"}:
        inferred_source = args.ztf_depth_source
        if inferred_source is None:
            if any(
                value is not None
                for value in (args.ztf_m5_g, args.ztf_m5_r, args.ztf_m5_i)
            ):
                inferred_source = "manual"
            elif args.ztf_depth_quantile_table is not None:
                inferred_source = "table"
            elif args.ztf_depth_dir is not None:
                inferred_source = "joblib"
            else:
                inferred_source = "manual"
        args.ztf_depth_source = prompt_choice(
            "ZTF limiting-depth input",
            [
                ("manual", "Enter fixed 5-sigma limiting magnitudes"),
                ("table", "Read p25/p50/p75 values from a CSV table"),
                ("joblib", "Read the legacy NMMA joblib KDE files"),
            ],
            inferred_source,
        )
        requested_ztf_bands = tuple(
            dict.fromkeys(
                piece.strip().lower()
                for piece in args.color_pair.split("-")
                if piece.strip()
            )
        )
        if args.ztf_depth_source == "manual":
            for band in requested_ztf_bands:
                if band not in ZTF_DEPTH_FILENAMES:
                    continue
                attribute = f"ztf_m5_{band}"
                current = getattr(args, attribute)
                default = "" if current is None else str(current)
                value = prompt_text(
                    f"Fixed ZTF {band}-band 5-sigma limiting magnitude [AB mag]",
                    default,
                )
                if not value.strip():
                    raise ValueError(f"A fixed ZTF {band}-band depth is required")
                setattr(args, attribute, float(value))
        elif args.ztf_depth_source == "table":
            args.ztf_depth_quantile_table = Path(prompt_text(
                "ZTF depth-quantile CSV table",
                str(args.ztf_depth_quantile_table or "ztf_depth_quantiles.csv"),
            ))
        else:
            args.ztf_depth_dir = Path(prompt_text(
                "Directory containing the three ZTF lims joblib files",
                str(args.ztf_depth_dir or "ztf_depth_distributions"),
            ))

    if not (
        args.photometric_system == "ztf"
        and args.ztf_depth_source == "manual"
    ):
        args.m5_scenario = prompt_choice(
            "m5 threshold used for the depth cut",
            list(M5_SCENARIO_LABELS.items()),
            args.m5_scenario,
        )
    else:
        print(
            "ZTF manual-depth mode: the entered fixed limits are used directly; "
            "p25/p50/p75 is not applied."
        )
    if args.photometric_system != "ztf":
        args.threshold_scope = prompt_choice(
            "m5 latitude scope",
            [
                ("common_global", "Same all-sky thresholds for controlled comparisons"),
                ("latitude_conditioned", "Thresholds measured at |b| >= 20 deg"),
            ],
            args.threshold_scope,
        )
    at2017gfo_comparison = prompt_choice(
        "AT2017gfo comparison",
        [
            ("enable", "Overlay AT2017gfo and report the points outside the envelope"),
            ("skip", "Do not compare this run with AT2017gfo"),
        ],
        "enable" if args.at2017gfo_file is not None else "skip",
    )
    if at2017gfo_comparison == "enable":
        at_path = prompt_text(
            "AT2017gfo photometry file",
            (
                str(args.at2017gfo_file)
                if args.at2017gfo_file is not None
                else "AT2017gfo_corrected.dat"
            ),
        ).strip()
        args.at2017gfo_file = Path(at_path)
        args.at2017gfo_merger_mjd = float(prompt_text(
            "AT2017gfo merger epoch [MJD]",
            str(args.at2017gfo_merger_mjd),
        ))
        args.at2017gfo_match_window_days = float(prompt_text(
            "Maximum inter-band matching separation [days]",
            str(args.at2017gfo_match_window_days),
        ))
        args.at2017gfo_ebv = float(prompt_text(
            "AT2017gfo Milky-Way E(B-V)",
            str(args.at2017gfo_ebv),
        ))
        args.at2017gfo_error_sigma = float(prompt_text(
            "Photometric-error compatibility level [sigma]",
            str(args.at2017gfo_error_sigma),
        ))
        args.at2017gfo_photometry_mode = prompt_choice(
            "AT2017gfo input photometry",
            [
                ("dereddened", "Already corrected for Milky-Way extinction"),
                ("mw_extincted", "Still contains Milky-Way extinction"),
            ],
            args.at2017gfo_photometry_mode,
        )
        bestfit_path = prompt_text(
            "AT2017gfo best-fit pickle (empty = do not overlay)",
            (
                str(args.at2017gfo_bestfit_file)
                if args.at2017gfo_bestfit_file is not None else ""
            ),
        ).strip()
        args.at2017gfo_bestfit_file = Path(bestfit_path) if bestfit_path else None
        if args.at2017gfo_bestfit_file is not None:
            args.at2017gfo_bestfit_photometry_mode = prompt_choice(
                "Foreground MW extinction in the AT2017gfo best-fit magnitudes",
                [
                    ("dereddened", "Absent/already corrected"),
                    ("mw_extincted", "Already present"),
                ],
                args.at2017gfo_bestfit_photometry_mode,
            )
    else:
        args.at2017gfo_file = None
        args.at2017gfo_bestfit_file = None

    snia_comparison = prompt_choice(
        "SN Ia population comparison",
        [
            ("enable", "Build SN Ia colour plots and KNe--SN Ia overlap products"),
            ("skip", "Do not load an SN Ia population"),
        ],
        "enable" if args.snia_parquet is not None else "skip",
    )
    if snia_comparison == "enable":
        snia_path = prompt_text(
            "SN Ia parquet file",
            str(args.snia_parquet) if args.snia_parquet is not None
            else "snia_mock_6band.parquet",
        ).strip()
        args.snia_parquet = Path(snia_path)
        args.snia_magnitude_column = prompt_choice(
            "SN Ia magnitude used for the colour",
            [
                ("mag_perfect", "Noise-free model magnitude"),
                ("mag", "Noisy observed magnitude"),
            ],
            args.snia_magnitude_column,
        )
        args.snia_input_mw_mode = prompt_choice(
            "Foreground MW extinction in the SN Ia input magnitude",
            [
                ("dereddened", "Absent/already corrected; add A for the MW-present output"),
                ("mw_extincted", "Already present; subtract A for the corrected output"),
            ],
            args.snia_input_mw_mode,
        )
        args.snia_time_reference = prompt_choice(
            "SN Ia time reference",
            [
                ("first_observation", "First available time of each SN Ia"),
                ("peak", "SALT2 t0 (time of maximum light)"),
            ],
            args.snia_time_reference,
        )
        args.snia_position_seed = int(prompt_text(
            "SN Ia to KNe sky-position assignment seed",
            str(args.snia_position_seed),
        ))
        args.snia_max_events = int(prompt_text(
            "Maximum number of SNe Ia (0 = all)",
            str(args.snia_max_events),
        ))
    else:
        args.snia_parquet = None
    return args
