#!/usr/bin/env python3
"""Build four colour envelopes from saved synthetic light curves.

The reader prefers sharded ``parquet/synthetic_part_*.parquet`` populations
and falls back to legacy one-file-per-event CSV products. New paired shards
may contain LSST, PS1 and ZTF magnitudes for the same event IDs. The script never
evaluates the surrogate, so both storage layouts preserve the generator's
numerical and photometric selections exactly.

Stages
------
1. intrinsic_no_m5_no_mw:
   numerically supported magnitudes saved by ``generate_kne_lightcurves.py``,
   with MW extinction removed and without an m5 cut.
2. no_m5_with_mw:
   the saved MW-extincted magnitudes, without an m5 cut.
3. m5_no_mw:
   the same intrinsic magnitudes, retained only when both bands are brighter
   than their selected m5 thresholds.
4. m5_with_mw:
   MW extinction is added to both bands before applying the same m5 cuts.

For a PS1 envelope, colour values and extinction coefficients are PS1. Rubin
OpSim m5 is nevertheless evaluated on the corresponding saved LSST bands,
because OpSim supplies Rubin rather than PS1 visit depths. Thus the code never
compares a PS1 magnitude directly with an LSST m5 threshold.

For a ZTF envelope, colour values, MW coefficients and depth selection are all
ZTF. The depth threshold is the requested p25/p50/p75 quantile of the empirical
per-image limiting-magnitude KDE in each band. This controlled global-depth
comparison does not simulate ZTF cadence, weather correlations or visit noise.

The script never reloads or evaluates Bu2026_MLP.  Consequently, the numerical
support cut chosen during generation is preserved exactly.  Saved synthetic
``mag`` values include MW extinction; intrinsic magnitudes are reconstructed
with ``m_intrinsic = m_saved - R_band E(B-V)``.

The analysed population can use every generated latitude, only
``|b| >= b_cut`` events, or only ``|b| < b_cut`` events. ``b_cut`` is freely
configurable. This population selection is independent of the latitude scope
used to obtain the OpSim m5 quantiles.

Envelope bounds are chosen from a grid extending up to p49-p51 and, if
necessary, the degenerate p50-p50 case.  The widest statistically valid pair
is retained in each temporal bin.  The median is always stored and drawn.

Population panels use every finite saved synthetic point, including points
rejected by m5.  The no-MW stages show the complete intrinsic population;
the with-MW stages show the complete MW-extincted population.  Each event is
drawn as an individual trajectory joining its saved samples:
there is no population binning and no evaluation at envelope-bin centres.
Envelope estimation still uses fixed bin centres for a reproducible cut
definition.

An optional Rubin/LSST SN Ia parquet population can be projected onto a
deterministic sample of the KNe sky positions.  The corresponding KNe
``E(B-V)`` and Rubin extinction coefficients are then used to produce four
strictly separated SN Ia stages: with/without foreground MW extinction,
each with/without the selected Rubin m5 cut.  The code writes SN Ia population
plots, SN Ia envelopes, KNe--SN Ia overlays, and per-bin overlap diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dynamic_color_cut_common import (
    assign_event_splits,
    bootstrap_percentiles_batched,
    canonical_event_id,
    discover_synthetic_files,
    event_id_from_path,
    interpolate_band,
    load_summary,
    parse_floats,
    prompt_choice,
    prompt_text,
    run_root,
)
from kne_pipeline_common import discover_m5_quantile_table


RUBIN_MW_R = {
    "u": 4.757217815396922,
    "g": 3.6605664439892616,
    "r": 2.70136780871597,
    "i": 2.0536599130965882,
    "z": 1.5900964472616756,
    "y": 1.3077049588254708,
}

PS1_MW_R = {
    "g": 3.172,
    "r": 2.271,
    "i": 1.682,
    "z": 1.322,
    "y": 1.087,
}

ZTF_MW_R = {
    "g": 3.303,
    "r": 2.285,
    "i": 1.698,
}

MW_R_BY_SYSTEM = {
    "lsst": RUBIN_MW_R,
    "ps1": PS1_MW_R,
    "ztf": ZTF_MW_R,
}

ZTF_DEPTH_FILENAMES = {
    "g": ("lims_public_g.joblib", "lims_g.joblib"),
    "r": ("lims_public_r.joblib", "lims_r.joblib"),
    "i": ("lims_i.joblib", "lims_public_i.joblib"),
}

M5_SCENARIO_PERCENTILES = {
    "worst_p25": 25.0,
    "median_p50": 50.0,
    "best_p75": 75.0,
}

LATITUDE_COLUMNS = (
    "galactic_b_deg",
    "galactic_latitude_deg",
    "b_gal_deg",
    "b_deg",
)

RA_COLUMNS = ("fieldRA", "field_ra_deg", "ra_deg", "ra")
DEC_COLUMNS = ("fieldDec", "field_dec_deg", "dec_deg", "dec")

SNIA_CONVENTIONS = {
    "mw_corrected": "intrinsic_no_m5_no_mw",
    "mw_present": "no_m5_with_mw",
    "m5_no_mw": "m5_no_mw",
    "m5_with_mw": "m5_with_mw",
}

SNIA_BOOTSTRAP_SEED_OFFSETS = {
    "mw_corrected": 8001,
    "mw_present": 8002,
    "m5_no_mw": 8003,
    "m5_with_mw": 8004,
}

# q is the lower percentile of the symmetric p_q--p_(100-q) envelope.
DEFAULT_TAIL_GRID = (
    "0.02,0.05,0.1,0.25,0.5,0.53,1,2.5,5,10,16,20,25,30,35,"
    "40,42,44,45,46,47,48,49,50"
)

M5_SCENARIOS = {
    "worst_p25": "worst p25",
    "median_p50": "median p50",
    "best_p75": "best p75",
}

STAGES = (
    "intrinsic_no_m5_no_mw",
    "no_m5_with_mw",
    "m5_no_mw",
    "m5_with_mw",
)

MW_EXTINCTED_STAGES = frozenset({"no_m5_with_mw", "m5_with_mw"})
M5_STAGES = frozenset({"m5_no_mw", "m5_with_mw"})

# Preserve the historical bootstrap streams of the original three stages.
# The new no-m5/MW stage receives its own previously unused offset.
STAGE_BOOTSTRAP_SEED_OFFSETS = {
    "intrinsic_no_m5_no_mw": 1,
    "m5_no_mw": 2,
    "m5_with_mw": 3,
    "no_m5_with_mw": 4,
}

AT2017GFO_DEFAULT_MERGER_MJD = 57982.52851852
AT2017GFO_FILTER_ALIASES = {
    "u": ("sdssu", "sdss::u", "u"),
    "g": ("ps1::g", "ps1g", "g"),
    "r": ("ps1::r", "ps1r", "r"),
    "i": ("ps1::i", "ps1i", "i"),
    "z": ("ps1::z", "ps1z", "z"),
    "y": ("ps1::y", "ps1y", "y"),
}


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
        choices=list(M5_SCENARIOS),
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
            [(key, label) for key, label in M5_SCENARIOS.items()],
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
    else:
        args.at2017gfo_file = None

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


def discover_m5_table(run_dir: Path, explicit: Path | None) -> Path:
    return discover_m5_quantile_table(run_root(run_dir), explicit=explicit)


def population_latitude_label(args: argparse.Namespace) -> str:
    """Human-readable KNe population latitude selection."""
    mode = args.population_latitude_selection
    if mode == "all":
        return "all generated Galactic latitudes (no latitude cut)"
    operator = ">=" if mode == "high" else "<"
    return f"|b| {operator} {args.latitude_threshold_deg:g} deg"


def population_latitude_tag(args: argparse.Namespace) -> str:
    """Filesystem-safe tag preventing different latitude cuts overwriting."""
    mode = args.population_latitude_selection
    if mode == "all":
        return "all_latitudes"
    threshold = f"{args.latitude_threshold_deg:g}".replace(".", "p")
    relation = "ge" if mode == "high" else "lt"
    return f"absb_{relation}_{threshold}deg"


class SyntheticPopulationSource:
    """Read saved synthetic curves from Parquet shards or legacy CSV files.

    Parquet is preferred when ``parquet/synthetic_part_*.parquet`` exists.
    Each shard is loaded only once and grouped by its repeated ``event_id``;
    this avoids one filesystem operation per KNe while bounding memory by the
    generator's shard size. The legacy one-CSV-per-event layout remains fully
    supported for existing runs and VM-oriented samples.
    """

    def __init__(self, run_dir: Path):
        self.root = run_root(run_dir)
        self.parquet_paths = sorted(
            (self.root / "parquet").glob("synthetic_part_*.parquet")
        )
        self.csv_paths = discover_synthetic_files(run_dir)
        self.csv_by_event = {
            event_id_from_path(path): path for path in self.csv_paths
        }
        if self.parquet_paths:
            self.kind = "parquet_shards"
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Reading the synthetic Parquet population requires pyarrow. "
                    "Install it with `python -m pip install pyarrow`."
                ) from exc
        elif self.csv_paths:
            self.kind = "legacy_csv_files"
        else:
            self.kind = "missing"

    def available_event_ids(self) -> set[str]:
        if self.kind == "legacy_csv_files":
            return set(self.csv_by_event)
        if self.kind == "parquet_shards":
            summary = load_summary(self.root)
            if not summary.empty and "synthetic_parquet_saved" in summary:
                saved = (
                    summary["synthetic_parquet_saved"]
                    .astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
                )
                return set(summary.loc[saved, "event_id"].map(canonical_event_id))
            available: set[str] = set()
            for path in self.parquet_paths:
                identifiers = pd.read_parquet(path, columns=["event_id"])
                available.update(
                    identifiers["event_id"].map(canonical_event_id).unique()
                )
            return available
        return set()

    def iter_events(
        self,
        requested_event_ids: list[str],
        bands: tuple[str, str],
        photometric_systems: tuple[str, ...] = ("lsst",),
    ):
        requested = {canonical_event_id(value) for value in requested_event_ids}
        if self.kind == "legacy_csv_files":
            for event_id in requested_event_ids:
                canonical = canonical_event_id(event_id)
                lightcurve = pd.read_csv(self.csv_by_event[canonical])
                if "photometric_system" not in lightcurve:
                    lightcurve["photometric_system"] = "lsst"
                yield canonical, lightcurve
            return

        for path in self.parquet_paths:
            import pyarrow.parquet as pq

            schema_names = set(pq.read_schema(path).names)
            has_system = "photometric_system" in schema_names
            columns = ["event_id", "t_days", "band", "mag"]
            filters = [("band", "in", list(bands))]
            if has_system:
                columns.append("photometric_system")
                filters.append(
                    ("photometric_system", "in", list(photometric_systems))
                )
            elif any(system != "lsst" for system in photometric_systems):
                raise ValueError(
                    f"{path} has no photometric_system column. It is a legacy "
                    "LSST-only shard and cannot build the requested non-LSST envelope."
                )
            shard = pd.read_parquet(
                path,
                columns=columns,
                filters=filters,
            )
            if shard.empty:
                continue
            if not has_system:
                shard["photometric_system"] = "lsst"
            shard["event_id"] = shard["event_id"].map(canonical_event_id)
            shard = shard.loc[shard["event_id"].isin(requested)]
            for event_id, lightcurve in shard.groupby("event_id", sort=False):
                yield canonical_event_id(event_id), lightcurve.drop(
                    columns=["event_id"]
                ).reset_index(drop=True)


def load_m5(
    path: Path,
    scope: str,
    scenario: str,
    bands: tuple[str, str],
) -> dict[str, float]:
    table = pd.read_csv(path)
    required = {
        "threshold_scope", "threshold_latitude_group", "band", "scenario", "m5"
    }
    if not required.issubset(table.columns):
        raise ValueError(f"{path} is missing {sorted(required - set(table.columns))}")
    group = "both" if scope == "common_global" else "high"
    use = table.loc[
        (table["threshold_scope"] == scope)
        & (table["threshold_latitude_group"] == group)
        & table["band"].astype(str).str.lower().isin(bands)
        & (table["scenario"] == scenario)
    ]
    lookup = {
        str(row.band).lower(): float(row.m5)
        for row in use.itertuples()
    }
    missing = set(bands) - set(lookup)
    if missing:
        raise ValueError(
            f"Missing {scenario} m5 thresholds for {sorted(missing)} in {path}"
        )
    return lookup


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_ztf_depth_file(directory: Path, band: str) -> Path:
    """Resolve the NMMA ZTF depth-KDE filename for one band."""
    for filename in ZTF_DEPTH_FILENAMES[band]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    expected = ", ".join(ZTF_DEPTH_FILENAMES[band])
    raise FileNotFoundError(
        f"Missing ZTF {band}-band depth distribution below {directory}. "
        f"Expected one of: {expected}"
    )


def load_trusted_ztf_kde(path: Path):
    """Load one trusted historical sklearn KDE used by the NMMA ZTF simulator.

    The distributed objects were serialized with an older scikit-learn.  The
    two compatibility aliases below preserve their public ``sample`` behaviour
    with recent scikit-learn releases.  Joblib uses pickle internally, hence
    the caller must only provide files obtained from a trusted source.
    """
    try:
        import joblib
        import sklearn.metrics._dist_metrics as distance_metrics
    except ImportError as exc:
        raise ImportError(
            "Reading ZTF lims joblib files requires joblib and scikit-learn."
        ) from exc

    if (
        not hasattr(distance_metrics, "EuclideanDistance")
        and hasattr(distance_metrics, "EuclideanDistance64")
    ):
        distance_metrics.EuclideanDistance = distance_metrics.EuclideanDistance64
    kde = joblib.load(path)
    if not hasattr(kde, "sample"):
        raise TypeError(f"{path} does not contain a sampleable depth KDE")
    if not hasattr(kde, "bandwidth_") and hasattr(kde, "bandwidth"):
        kde.bandwidth_ = kde.bandwidth
    return kde


def load_ztf_depth_thresholds(
    directory: Path | None,
    quantile_table: Path | None,
    scenario: str,
    bands: tuple[str, str],
    n_samples: int,
    seed: int,
    depth_source: str | None = None,
    manual_thresholds: dict[str, float | None] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return fixed ZTF limiting-magnitude thresholds and their provenance.

    Direct user values take no percentile interpretation.  A quantile table
    supplies an already calibrated p25/p50/p75 scenario.  Loading historical
    joblib KDEs remains available as an explicit legacy compatibility mode.
    """
    if any(band not in ZTF_DEPTH_FILENAMES for band in bands):
        raise ValueError("ZTF envelopes support only g, r and i bands")

    manual_thresholds = manual_thresholds or {}
    if depth_source is None:
        if any(value is not None for value in manual_thresholds.values()):
            depth_source = "manual"
        elif quantile_table is not None:
            depth_source = "table"
        elif directory is not None:
            depth_source = "joblib"
        else:
            raise ValueError(
                "ZTF depth selection requires fixed --ztf-m5-* values, "
                "--ztf-depth-quantile-table, or the legacy --ztf-depth-dir"
            )

    if depth_source == "manual":
        lookup: dict[str, float] = {}
        rows: list[dict[str, object]] = []
        for band in dict.fromkeys(bands):
            value = manual_thresholds.get(band)
            if value is None or not np.isfinite(value):
                raise ValueError(
                    f"Manual ZTF depth mode requires --ztf-m5-{band}"
                )
            value = float(value)
            if not 0.0 < value < 40.0:
                raise ValueError(
                    f"Invalid fixed ZTF {band}-band limiting magnitude: {value}"
                )
            lookup[band] = value
            rows.append({
                "band": band,
                "scenario": "manual_fixed",
                "percentile": np.nan,
                "m5": value,
                "source_kind": "user_supplied_fixed_depth",
                "source_file": np.nan,
                "source_sha256": np.nan,
                "sampling_seed": np.nan,
                "sampling_draws": np.nan,
            })
        return lookup, pd.DataFrame(rows)

    if depth_source == "table":
        if quantile_table is None:
            raise ValueError(
                "ZTF table mode requires --ztf-depth-quantile-table"
            )
        source = quantile_table.expanduser().resolve()
        table = pd.read_csv(source)
        required = {"band", "scenario", "m5"}
        if not required.issubset(table.columns):
            raise ValueError(
                f"{source} is missing {sorted(required - set(table.columns))}"
            )
        use = table.loc[
            table["band"].astype(str).str.lower().isin(bands)
            & (table["scenario"].astype(str) == scenario)
        ].copy()
        lookup = {
            str(row.band).lower(): float(row.m5)
            for row in use.itertuples()
        }
        missing = set(bands) - set(lookup)
        if missing:
            raise ValueError(
                f"Missing {scenario} ZTF thresholds for {sorted(missing)} in {source}"
            )
        use["source_file"] = str(source)
        use["source_sha256"] = file_sha256(source)
        use["source_kind"] = "user_supplied_quantile_table"
        use["sampling_seed"] = np.nan
        use["sampling_draws"] = np.nan
        return lookup, use

    if depth_source != "joblib":
        raise ValueError(f"Unsupported ZTF depth source: {depth_source}")
    if directory is None:
        raise ValueError(
            "Legacy ZTF joblib mode requires --ztf-depth-dir"
        )
    if n_samples < 1000:
        raise ValueError("--ztf-depth-samples must be at least 1000")
    directory = directory.expanduser().resolve()
    percentile = M5_SCENARIO_PERCENTILES[scenario]
    rows: list[dict[str, object]] = []
    lookup: dict[str, float] = {}
    for band_index, band in enumerate(dict.fromkeys(bands)):
        path = find_ztf_depth_file(directory, band)
        kde = load_trusted_ztf_kde(path)
        band_seed = int(seed + 1009 * (band_index + 1))
        draws = np.asarray(
            kde.sample(n_samples=int(n_samples), random_state=band_seed),
            dtype=float,
        ).reshape(-1)
        draws = draws[np.isfinite(draws)]
        if draws.size < 1000:
            raise ValueError(f"Too few finite ZTF depth draws from {path}")
        quantiles = {
            key: float(np.percentile(draws, percentile_value))
            for key, percentile_value in M5_SCENARIO_PERCENTILES.items()
        }
        lookup[band] = quantiles[scenario]
        for scenario_name, value in quantiles.items():
            rows.append({
                "band": band,
                "scenario": scenario_name,
                "percentile": M5_SCENARIO_PERCENTILES[scenario_name],
                "m5": value,
                "source_kind": "legacy_nmma_joblib_kde",
                "source_file": str(path),
                "source_sha256": file_sha256(path),
                "sampling_seed": band_seed,
                "sampling_draws": int(draws.size),
            })
    return lookup, pd.DataFrame(rows)


def event_extinction_mag(
    metadata_row: pd.Series,
    photometric_system: str,
    band: str,
) -> float:
    """Return the exact A_band stored by the paired generator when available.

    New LSST+PS1 runs store ``A_<system>_<band>_mw`` once per event.  Older
    LSST runs are supported through ``R_band * E(B-V)``.  PS1 fallback uses
    the same Schlafly & Finkbeiner coefficients as the generator.
    """
    column = f"A_{photometric_system}_{band}_mw"
    if column in metadata_row.index:
        value = pd.to_numeric(pd.Series([metadata_row[column]]), errors="coerce").iloc[0]
        if np.isfinite(value):
            return float(value)
    ebv = float(metadata_row["ebv_mw"])
    return float(MW_R_BY_SYSTEM[photometric_system][band] * ebv)


def select_metadata_subset(
    metadata: pd.DataFrame,
    maximum_events: int,
    subsample_seed: int | None,
) -> pd.DataFrame:
    """Select a reproducible event subset, optionally nested across sample sizes.

    With one fixed ``subsample_seed``, each requested size is a prefix of the
    same random permutation.  Consequently, the N=1,000 sample is contained in
    the N=10,000 sample.  A missing seed preserves the historical first-event
    selection used by ``--max-events``.
    """
    if maximum_events <= 0 or maximum_events >= len(metadata):
        return metadata.copy().reset_index(drop=True)
    maximum = min(int(maximum_events), len(metadata))
    if subsample_seed is None:
        return metadata.iloc[:maximum].copy().reset_index(drop=True)
    subset_rng = np.random.default_rng(subsample_seed)
    selected = subset_rng.permutation(len(metadata))[:maximum]
    subset = metadata.iloc[selected].copy()
    return subset.sort_values(
        "event_id", key=lambda s: pd.to_numeric(s, errors="coerce")
    ).reset_index(drop=True)


def prepare_metadata(
    args: argparse.Namespace,
    available_event_ids: set[str],
) -> pd.DataFrame:
    summary = load_summary(args.run_dir)
    if summary.empty:
        raise FileNotFoundError(f"Missing usable {run_root(args.run_dir) / 'summary.csv'}")
    physical_columns = (
        "log10_mej_dyn", "v_ej_dyn", "Ye_dyn", "log10_mej_wind",
        "v_ej_wind", "Ye_wind", "inclination_EM",
        "luminosity_distance", "redshift",
    )
    required_columns = ("ebv_mw",) + physical_columns
    missing = [c for c in required_columns if c not in summary]
    if missing:
        raise ValueError(f"summary.csv is missing {missing}")
    metadata = summary.drop_duplicates("event_id").copy()

    latitude_column = next((c for c in LATITUDE_COLUMNS if c in metadata), None)
    latitude_mode = args.population_latitude_selection
    before_latitude = len(metadata)
    if latitude_mode == "all":
        print(
            "No Galactic-latitude cut applied to the KNe population: "
            f"{len(metadata):,}/{before_latitude:,} events retained"
        )
    elif latitude_column is not None:
        latitude = pd.to_numeric(metadata[latitude_column], errors="coerce")
        finite = latitude.notna()
        if latitude_mode == "high":
            latitude_keep = finite & (
                latitude.abs() >= args.latitude_threshold_deg
            )
        else:
            latitude_keep = finite & (
                latitude.abs() < args.latitude_threshold_deg
            )
        metadata = metadata.loc[latitude_keep].copy()
        print(
            f"Applied {population_latitude_label(args)} using {latitude_column}: "
            f"{len(metadata):,}/{before_latitude:,} events retained"
        )
    else:
        selection_assumed = bool(
            args.assume_latitude_selection_already_applied
            or (args.assume_high_latitude_run and latitude_mode == "high")
        )
        if not selection_assumed and sys.stdin.isatty():
            confirmation = prompt_choice(
                "summary.csv has no Galactic-latitude column",
                [
                    (
                        "yes",
                        "The input run already contains only events satisfying "
                        + population_latitude_label(args),
                    ),
                    ("no", "Stop: the requested latitude selection cannot be verified"),
                ],
                "no",
            )
            selection_assumed = confirmation == "yes"
        if not selection_assumed:
            raise ValueError(
                "The requested population selection "
                f"({population_latitude_label(args)}) cannot be applied because "
                "summary.csv has no Galactic-latitude column. Supply such a "
                "column, choose --population-latitude-selection all, or use "
                "--assume-latitude-selection-already-applied only for an input "
                "run already generated with this exact selection."
            )
        print(
            "No Galactic-latitude column: accepting the explicit confirmation "
            "that the input run already satisfies "
            f"{population_latitude_label(args)}."
        )

    for column in required_columns:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    metadata = metadata.dropna(subset=list(required_columns))
    metadata["event_id"] = metadata["event_id"].map(canonical_event_id)
    before_files = len(metadata)
    metadata = metadata.loc[metadata["event_id"].isin(available_event_ids)].copy()
    print(
        f"Matched saved synthetic events: {len(metadata):,}/{before_files:,} "
        "selected summary events"
    )
    metadata = metadata.sort_values(
        "event_id", key=lambda s: pd.to_numeric(s, errors="coerce")
    ).reset_index(drop=True)
    metadata = select_metadata_subset(
        metadata,
        args.max_events,
        args.event_subsample_seed,
    )
    if metadata.empty:
        raise ValueError(
            "No events satisfying the requested Galactic-latitude selection "
            "have a saved synthetic light curve"
        )
    return metadata


def percentile_diagnostics(
    values: np.ndarray,
    lower_grid: list[float],
    n_boot: int,
    bootstrap_threshold: float,
    minimum_tail_events: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    lower = sorted(set(float(q) for q in lower_grid if 0.0 < q <= 50.0))
    percentiles = sorted(set(lower + [100.0 - q for q in lower] + [50.0]))
    estimate, boot16, boot84, half = bootstrap_percentiles_batched(
        values, percentiles, n_boot, rng
    )
    n_values = len(values)
    rows = []
    for q, value, low, high, width in zip(percentiles, estimate, boot16, boot84, half):
        expected_tail = n_values * min(q, 100.0 - q) / 100.0
        width_valid = bool(np.isfinite(width) and width <= bootstrap_threshold)
        tail_valid = bool(expected_tail >= minimum_tail_events)
        rows.append({
            "percentile": q,
            "value": value,
            "bootstrap_p16": low,
            "bootstrap_p84": high,
            "bootstrap_half_width_68": width,
            "expected_nearest_tail_events": expected_tail,
            "bootstrap_width_valid": width_valid,
            "tail_support_valid": tail_valid,
            "percentile_valid": bool(width_valid and tail_valid),
        })
    diagnostics = pd.DataFrame(rows)
    by_q = diagnostics.set_index("percentile")
    median_valid = bool(by_q.loc[50.0, "percentile_valid"])
    valid_pairs = [
        q for q in lower
        if median_valid
        and bool(by_q.loc[q, "percentile_valid"])
        and bool(by_q.loc[100.0 - q, "percentile_valid"])
    ]
    if not valid_pairs:
        return diagnostics, None

    # The smallest q is the widest statistically valid envelope.  If broad
    # pairs fail, the search continues through p45-p55, p49-p51 and p50-p50.
    q_low = min(valid_pairs)
    q_high = 100.0 - q_low
    selected = {
        "low_percentile": q_low,
        "high_percentile": q_high,
        "color_min": float(by_q.loc[q_low, "value"]),
        "color_median": float(by_q.loc[50.0, "value"]),
        "color_max": float(by_q.loc[q_high, "value"]),
        "low_bootstrap_half_width_68": float(by_q.loc[q_low, "bootstrap_half_width_68"]),
        "median_bootstrap_half_width_68": float(by_q.loc[50.0, "bootstrap_half_width_68"]),
        "high_bootstrap_half_width_68": float(by_q.loc[q_high, "bootstrap_half_width_68"]),
        "expected_lower_tail_events": float(n_values * q_low / 100.0),
        "expected_upper_tail_events": float(n_values * q_low / 100.0),
        "nominal_coverage_percent": float(q_high - q_low),
    }
    return diagnostics, selected


def draw_population(
    ax: plt.Axes,
    time_tracks: list[np.ndarray],
    color_tracks: list[np.ndarray],
    trajectory_colors: list[str] | None = None,
) -> None:
    """Draw every event trajectory without time/colour binning."""
    segments = []
    segment_colors = []
    isolated_times = []
    isolated_colors = []
    isolated_plot_colors = []
    for track_index, (times, colors) in enumerate(zip(time_tracks, color_tracks)):
        times = np.asarray(times, dtype=float)
        colors = np.asarray(colors, dtype=float)
        plot_color = (
            trajectory_colors[track_index]
            if trajectory_colors is not None else "black"
        )
        finite = np.isfinite(times) & np.isfinite(colors)
        points = np.column_stack((times[finite], colors[finite]))
        if len(points) >= 2:
            segments.append(points)
            segment_colors.append(plot_color)
        elif len(points) == 1:
            isolated_times.append(points[0, 0])
            isolated_colors.append(points[0, 1])
            isolated_plot_colors.append(plot_color)
    if segments:
        collection = LineCollection(
            segments,
            colors=segment_colors,
            linewidths=0.16 if trajectory_colors is None else 0.75,
            alpha=0.018 if trajectory_colors is None else 0.78,
            rasterized=True,
        )
        ax.add_collection(collection)
    if isolated_times:
        ax.scatter(
            isolated_times,
            isolated_colors,
            s=0.18 if trajectory_colors is None else 3.0,
            alpha=0.025 if trajectory_colors is None else 0.78,
            color=isolated_plot_colors,
            edgecolors="none",
            rasterized=True,
        )
    if not segments and not isolated_times:
        return


def draw_envelope(ax: plt.Axes, rules: pd.DataFrame) -> None:
    for row in rules.itertuples():
        x = [row.t_min_days, row.t_max_days]
        ax.fill_between(
            x,
            [row.color_min, row.color_min],
            [row.color_max, row.color_max],
            color="#4c78a8",
            alpha=0.24,
        )
        ax.plot(x, [row.color_min] * 2, color="#2f5f98", lw=1.0)
        ax.plot(x, [row.color_max] * 2, color="#2f5f98", lw=1.0)
        ax.plot(x, [row.color_median] * 2, color="#d55e00", lw=1.25)


def format_axes(ax: plt.Axes, pair: str, t_min: float, t_max: float) -> None:
    ax.set_xlabel(r"$t-t_{\rm merger}$ [days]")
    ax.set_ylabel(f"{pair} [mag]")
    ax.set_xlim(t_min, t_max)
    ax.grid(alpha=0.18)


def format_snia_axes(
    ax: plt.Axes,
    pair: str,
    t_min: float,
    t_max: float,
    comparison: bool,
    time_reference: str,
) -> None:
    snia_reference = (
        r"$t-t_{\rm first\ observation}$"
        if time_reference == "first_observation"
        else r"$t-t_{\rm max}$"
    )
    if comparison:
        ax.set_xlabel(
            r"phase [days] (KNe: $t-t_{\rm merger}$; SNe Ia: "
            + snia_reference + ")"
        )
    else:
        ax.set_xlabel(snia_reference + " [days]")
    ax.set_ylabel(f"{pair} [mag]")
    ax.set_xlim(t_min, t_max)
    ax.grid(alpha=0.18)


def first_existing_column(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    return next((column for column in aliases if column in columns), None)


def build_snia_sky_mapping(
    sn_ids: list[int],
    kne_metadata: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Assign each SN Ia a reproducible KNe sky position and MW extinction."""
    ra_column = first_existing_column(kne_metadata.columns, RA_COLUMNS)
    dec_column = first_existing_column(kne_metadata.columns, DEC_COLUMNS)
    if ra_column is None or dec_column is None:
        raise ValueError(
            "SN Ia sky projection requires KNe coordinates in summary.csv. "
            f"Accepted RA columns: {RA_COLUMNS}; DEC columns: {DEC_COLUMNS}."
        )

    pool = kne_metadata.copy()
    pool[ra_column] = pd.to_numeric(pool[ra_column], errors="coerce")
    pool[dec_column] = pd.to_numeric(pool[dec_column], errors="coerce")
    pool["ebv_mw"] = pd.to_numeric(pool["ebv_mw"], errors="coerce")
    pool = pool.loc[
        np.isfinite(pool[ra_column])
        & np.isfinite(pool[dec_column])
        & np.isfinite(pool["ebv_mw"])
    ].reset_index(drop=True)
    if pool.empty:
        raise ValueError(
            "No KNe event has finite sky coordinates and ebv_mw; "
            "the SN Ia projection cannot be constructed."
        )

    rng = np.random.default_rng(seed)
    replace = len(pool) < len(sn_ids)
    if replace:
        selected_indices = rng.choice(len(pool), size=len(sn_ids), replace=True)
        print(
            "Warning: fewer valid KNe positions than SNe Ia; KNe positions "
            "are sampled with replacement."
        )
    else:
        selected_indices = rng.permutation(len(pool))[:len(sn_ids)]
    selected = pool.iloc[selected_indices].reset_index(drop=True)

    mapping = pd.DataFrame({
        "sn_id": np.asarray(sn_ids, dtype=np.int64),
        "projected_kne_event_id": selected["event_id"].astype(str).to_numpy(),
        "fieldRA": selected[ra_column].to_numpy(float),
        "fieldDec": selected[dec_column].to_numpy(float),
        "ebv_mw": selected["ebv_mw"].to_numpy(float),
        "position_sampled_with_replacement": replace,
        "position_assignment_seed": int(seed),
    })
    for band, coefficient in RUBIN_MW_R.items():
        exact_column = f"A_lsst_{band}_mw"
        if exact_column in selected:
            exact = pd.to_numeric(selected[exact_column], errors="coerce").to_numpy(float)
        else:
            exact = np.full(len(selected), np.nan)
        fallback = mapping["ebv_mw"].to_numpy(float) * float(coefficient)
        mapping[f"A_lsst_{band}_mw"] = np.where(np.isfinite(exact), exact, fallback)
    return mapping


def interpolate_snia_band(
    curve: pd.DataFrame,
    band: str,
    centres: np.ndarray,
    magnitude_column: str,
    time_reference: str,
) -> np.ndarray:
    """Interpolate one SN Ia band without extrapolation.

    Peak-relative phases retain the historical log10-time interpolation on
    strictly positive phases. First-observation phases include t=0 and use
    linear time, since log10(0) is undefined.
    """
    subset = curve.loc[curve["band"] == band, ["t_days", magnitude_column]].copy()
    subset["t_days"] = pd.to_numeric(subset["t_days"], errors="coerce")
    subset[magnitude_column] = pd.to_numeric(
        subset[magnitude_column], errors="coerce"
    )
    valid_time = (
        subset["t_days"] >= 0
        if time_reference == "first_observation"
        else subset["t_days"] > 0
    )
    subset = subset.loc[
        np.isfinite(subset["t_days"])
        & valid_time
        & np.isfinite(subset[magnitude_column])
    ]
    subset = (
        subset.groupby("t_days", as_index=False)[magnitude_column]
        .median()
        .sort_values("t_days")
    )
    result = np.full(len(centres), np.nan, dtype=float)
    if len(subset) == 0:
        return result
    times = subset["t_days"].to_numpy(float)
    values = subset[magnitude_column].to_numpy(float)
    supported = (centres >= times.min()) & (centres <= times.max())
    if len(times) == 1:
        supported &= np.isclose(centres, times[0], atol=1e-10)
        result[supported] = values[0]
    elif time_reference == "first_observation":
        result[supported] = np.interp(centres[supported], times, values)
    else:
        result[supported] = np.interp(
            np.log10(centres[supported]), np.log10(times), values
        )
    return result


def prepare_snia_population(
    args: argparse.Namespace,
    kne_metadata: pd.DataFrame,
    band1: str,
    band2: str,
    centres: np.ndarray,
    m5: dict[str, float],
    output: Path,
) -> dict[str, object]:
    """Adapt an LSST SN Ia parquet and build colour matrices/tracks."""
    if args.snia_parquet is None:
        return {}
    path = args.snia_parquet.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SN Ia parquet not found: {path}")

    requested_columns = [
        "sn_id", "mjd", "band", args.snia_magnitude_column, "magerr",
        "redshift", "t0", "ra", "dec",
    ]
    try:
        measurements = pd.read_parquet(
            path,
            columns=requested_columns,
            filters=[("band", "in", [band1, band2])],
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Reading the SN Ia parquet requires pyarrow or fastparquet. "
            "Install pyarrow in the environment used to run this script."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Could not read the required SN Ia columns {requested_columns} "
            f"from {path}: {exc}"
        ) from exc

    measurements["band"] = measurements["band"].astype(str).str.strip().str.lower()
    measurements = measurements.loc[measurements["band"].isin((band1, band2))].copy()
    for column in (
        "sn_id", "mjd", args.snia_magnitude_column, "magerr", "redshift",
        "t0", "ra", "dec",
    ):
        measurements[column] = pd.to_numeric(measurements[column], errors="coerce")
    required_columns = ["sn_id", "mjd", "band", args.snia_magnitude_column]
    if args.snia_time_reference == "peak":
        required_columns.append("t0")
    measurements = measurements.dropna(subset=required_columns)
    measurements["sn_id"] = measurements["sn_id"].astype(np.int64)
    sn_ids = sorted(measurements["sn_id"].unique().tolist())
    if args.snia_max_events > 0:
        sn_ids = sn_ids[:args.snia_max_events]
        measurements = measurements.loc[measurements["sn_id"].isin(sn_ids)].copy()
    if not sn_ids:
        raise ValueError("The SN Ia parquet contains no usable rows in the requested bands")

    mapping = build_snia_sky_mapping(sn_ids, kne_metadata, args.snia_position_seed)
    mapping.to_csv(output / "snia_to_kne_sky_mapping.csv", index=False)
    measurements = measurements.rename(columns={"ra": "source_ra", "dec": "source_dec"})
    mapping_columns = [
        "sn_id", "projected_kne_event_id", "fieldRA", "fieldDec", "ebv_mw",
        f"A_lsst_{band1}_mw", f"A_lsst_{band2}_mw",
    ]
    measurements = measurements.merge(
        mapping[mapping_columns], on="sn_id", how="left", validate="many_to_one"
    )
    if args.snia_time_reference == "first_observation":
        measurements["snia_time_reference_mjd"] = measurements.groupby(
            "sn_id"
        )["mjd"].transform("min")
    else:
        measurements["snia_time_reference_mjd"] = measurements["t0"]
    measurements["snia_time_reference"] = args.snia_time_reference
    measurements["t_days"] = (
        measurements["mjd"] - measurements["snia_time_reference_mjd"]
    )
    measurements["event_id"] = measurements["sn_id"].map(
        lambda value: f"snia_{int(value):06d}"
    )
    measurements["photometric_system"] = "lsst"
    extinction_columns = {
        band: f"A_lsst_{band}_mw" for band in (band1, band2)
    }
    measurements["A_mw"] = np.nan
    for band, column in extinction_columns.items():
        mask = measurements["band"] == band
        measurements.loc[mask, "A_mw"] = measurements.loc[mask, column]
    source_magnitude = pd.to_numeric(
        measurements[args.snia_magnitude_column], errors="coerce"
    )
    if args.snia_input_mw_mode == "dereddened":
        measurements["mag_mw_corrected"] = source_magnitude
        measurements["mag_mw_present"] = source_magnitude + measurements["A_mw"]
    else:
        measurements["mag_mw_present"] = source_magnitude
        measurements["mag_mw_corrected"] = source_magnitude - measurements["A_mw"]
    # compact-extincted compatibility: ``mag`` is the magnitude with MW dust.
    measurements["mag"] = measurements["mag_mw_present"]
    measurements["time_mjd"] = measurements["mjd"]

    adapted_columns = [
        "event_id", "sn_id", "time_mjd", "t_days", "band", "mag",
        "mag_mw_corrected", "mag_mw_present", "magerr", "redshift", "t0",
        "snia_time_reference", "snia_time_reference_mjd",
        "fieldRA", "fieldDec", "source_ra", "source_dec", "ebv_mw", "A_mw",
        "projected_kne_event_id", "photometric_system",
    ]
    adapted_path = output / f"snia_adapted_{band1}_{band2}.parquet"
    if args.write_adapted_snia_parquet:
        measurements[adapted_columns].to_parquet(adapted_path, index=False)
    else:
        adapted_path = None

    event_ids = [f"snia_{int(sn_id):06d}" for sn_id in sn_ids]
    event_position = {event_id: index for index, event_id in enumerate(event_ids)}
    n_events = len(event_ids)
    matrices = {
        convention: np.full((n_events, len(centres)), np.nan, dtype=np.float32)
        for convention in SNIA_CONVENTIONS
    }
    population_times = {
        convention: [np.array([], dtype=np.float32) for _ in event_ids]
        for convention in SNIA_CONVENTIONS
    }
    population_colors = {
        convention: [np.array([], dtype=np.float32) for _ in event_ids]
        for convention in SNIA_CONVENTIONS
    }

    for event_id, curve in measurements.groupby("event_id", sort=False):
        index = event_position[event_id]
        corrected1 = interpolate_snia_band(
            curve, band1, centres, "mag_mw_corrected",
            args.snia_time_reference,
        )
        corrected2 = interpolate_snia_band(
            curve, band2, centres, "mag_mw_corrected",
            args.snia_time_reference,
        )
        corrected_colour = corrected1 - corrected2
        extinction1 = float(curve[f"A_lsst_{band1}_mw"].iloc[0])
        extinction2 = float(curve[f"A_lsst_{band2}_mw"].iloc[0])
        differential_extinction = extinction1 - extinction2
        present1 = corrected1 + extinction1
        present2 = corrected2 + extinction2
        present_colour = present1 - present2
        matrices["mw_corrected"][index] = corrected_colour.astype(np.float32)
        matrices["mw_present"][index] = present_colour.astype(np.float32)
        matrices["m5_no_mw"][index] = np.where(
            np.isfinite(corrected1)
            & np.isfinite(corrected2)
            & (corrected1 <= m5[band1])
            & (corrected2 <= m5[band2]),
            corrected_colour,
            np.nan,
        ).astype(np.float32)
        matrices["m5_with_mw"][index] = np.where(
            np.isfinite(present1)
            & np.isfinite(present2)
            & (present1 <= m5[band1])
            & (present2 <= m5[band2]),
            present_colour,
            np.nan,
        ).astype(np.float32)

        paired = (
            curve.groupby(["t_days", "band"], as_index=False)["mag_mw_corrected"]
            .median()
            .pivot(index="t_days", columns="band", values="mag_mw_corrected")
        )
        if band1 in paired.columns and band2 in paired.columns:
            paired = paired.dropna(subset=[band1, band2]).sort_index()
            times = paired.index.to_numpy(float)
            corrected_colours = (paired[band1] - paired[band2]).to_numpy(float)
            keep = (
                np.isfinite(times) & np.isfinite(corrected_colours)
                & (times >= args.t_min) & (times <= args.t_max)
            )
            population_times["mw_corrected"][index] = times[keep].astype(np.float32)
            population_colors["mw_corrected"][index] = (
                corrected_colours[keep].astype(np.float32)
            )
            population_times["mw_present"][index] = times[keep].astype(np.float32)
            population_colors["mw_present"][index] = (
                corrected_colours[keep] + differential_extinction
            ).astype(np.float32)
            # As for the KNe m5 stages, the unbinned background remains the
            # complete saved population; only the matrix used to estimate the
            # envelope is depth-selected.
            population_times["m5_no_mw"][index] = times[keep].astype(np.float32)
            population_colors["m5_no_mw"][index] = (
                corrected_colours[keep].astype(np.float32)
            )
            population_times["m5_with_mw"][index] = times[keep].astype(np.float32)
            population_colors["m5_with_mw"][index] = (
                corrected_colours[keep] + differential_extinction
            ).astype(np.float32)

    object_metadata = (
        measurements.sort_values(["sn_id", "mjd"])
        .drop_duplicates("event_id")[[
            "event_id", "sn_id", "redshift", "t0", "snia_time_reference",
            "snia_time_reference_mjd", "fieldRA", "fieldDec",
            "source_ra", "source_dec", "ebv_mw", "projected_kne_event_id",
        ]]
        .reset_index(drop=True)
    )
    split = assign_event_splits(
        event_ids, args.train_fraction, args.validation_fraction,
        args.seed + 7000,
    )
    split_map = split.set_index("event_id")["split"].to_dict()
    train = np.array([split_map[event_id] == "train" for event_id in event_ids])
    split.to_csv(output / "snia_event_split.csv", index=False)
    print(
        f"SN Ia: {n_events:,} objects projected onto KNe sky positions; "
        f"magnitude={args.snia_magnitude_column}; input MW mode="
        f"{args.snia_input_mw_mode}; time reference={args.snia_time_reference}"
    )
    return {
        "input_path": path,
        "adapted_path": adapted_path,
        "n_objects_with_requested_band_rows": n_events,
        "n_objects_with_both_requested_bands": int(
            measurements.groupby("sn_id")["band"].nunique().ge(2).sum()
        ),
        "event_ids": event_ids,
        "metadata": object_metadata,
        "train": train,
        "matrices": matrices,
        "population_times": population_times,
        "population_colors": population_colors,
        "mapping": mapping,
        "m5_thresholds": {band1: float(m5[band1]), band2: float(m5[band2])},
        "time_reference": args.snia_time_reference,
    }


def build_population_envelope_tables(
    matrix: np.ndarray,
    train: np.ndarray,
    edges: np.ndarray,
    centres: np.ndarray,
    tails: list[float],
    args: argparse.Namespace,
    rng: np.random.Generator,
    stage: str,
    interpolation_method: str = "linear_in_log10_positive_phase",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build envelope rules and diagnostics for an already prepared matrix."""
    rules: list[dict[str, object]] = []
    diagnostics_frames: list[pd.DataFrame] = []
    availability: list[dict[str, object]] = []
    consecutive_open = True
    for bin_index, centre in enumerate(centres):
        values = matrix[train, bin_index]
        values = values[np.isfinite(values)].astype(float)
        if len(values) >= args.minimum_events:
            diagnostics, selected = percentile_diagnostics(
                values,
                tails,
                args.n_boot,
                args.bootstrap_threshold_mag,
                args.minimum_tail_events,
                rng,
            )
        else:
            diagnostics, selected = pd.DataFrame(), None
        if not diagnostics.empty:
            diagnostics.insert(0, "bin_index", bin_index)
            diagnostics.insert(1, "t_min_days", edges[bin_index])
            diagnostics.insert(2, "t_max_days", edges[bin_index + 1])
            diagnostics.insert(3, "t_center_days", centre)
            diagnostics.insert(4, "N_train_events", len(values))
            diagnostics_frames.append(diagnostics)
        valid = selected is not None
        if not valid:
            consecutive_open = False
        availability.append({
            "bin_index": bin_index,
            "t_min_days": edges[bin_index],
            "t_max_days": edges[bin_index + 1],
            "t_center_days": centre,
            "N_train_total": int(train.sum()),
            "N_train_stage_available": len(values),
            "candidate_envelope_valid": valid,
            "initial_consecutive_valid": bool(valid and consecutive_open),
        })
        if valid:
            rules.append({
                "rule_id": f"{stage}_bin_{bin_index:03d}",
                "bin_index": bin_index,
                "stage": stage,
                "t_min_days": edges[bin_index],
                "t_max_days": edges[bin_index + 1],
                "t_center_days": centre,
                "bin_width_days": args.bin_width,
                "N_train_events": len(values),
                "initial_consecutive_valid": bool(consecutive_open),
                **selected,
                "bootstrap_threshold_mag": args.bootstrap_threshold_mag,
                "minimum_tail_events": args.minimum_tail_events,
                "target_interpolation_time_days": centre,
                "interpolation_method": interpolation_method,
            })
    rules_all = pd.DataFrame(rules)
    if rules_all.empty:
        rules_all = pd.DataFrame(columns=[
            "rule_id", "bin_index", "stage", "t_min_days", "t_max_days",
            "t_center_days", "bin_width_days", "N_train_events",
            "initial_consecutive_valid", "low_percentile", "high_percentile",
            "color_min", "color_median", "color_max",
        ])
    rules_consecutive = rules_all.loc[
        rules_all["initial_consecutive_valid"].astype(bool)
    ].copy()
    availability_frame = pd.DataFrame(availability)
    diagnostics_frame = (
        pd.concat(diagnostics_frames, ignore_index=True)
        if diagnostics_frames else pd.DataFrame()
    )
    return rules_all, rules_consecutive, availability_frame, diagnostics_frame


def draw_population_style(
    ax: plt.Axes,
    time_tracks: list[np.ndarray],
    color_tracks: list[np.ndarray],
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    segments = []
    for times, colours in zip(time_tracks, color_tracks):
        times = np.asarray(times, dtype=float)
        colours = np.asarray(colours, dtype=float)
        finite = np.isfinite(times) & np.isfinite(colours)
        points = np.column_stack((times[finite], colours[finite]))
        if len(points) >= 2:
            segments.append(points)
        elif len(points) == 1:
            ax.scatter(
                points[0, 0], points[0, 1], s=0.2, color=color,
                alpha=alpha, edgecolors="none", rasterized=True,
            )
    if segments:
        ax.add_collection(LineCollection(
            segments, colors=color, linewidths=linewidth, alpha=alpha,
            rasterized=True,
        ))


def draw_comparison_envelope(
    ax: plt.Axes,
    rules: pd.DataFrame,
    color: str,
    linestyle: str,
    alpha: float,
) -> None:
    for row in rules.itertuples():
        x = [row.t_min_days, row.t_max_days]
        ax.fill_between(
            x, [row.color_min] * 2, [row.color_max] * 2,
            color=color, alpha=alpha,
        )
        ax.plot(x, [row.color_min] * 2, color=color, lw=0.9, linestyle=linestyle)
        ax.plot(x, [row.color_max] * 2, color=color, lw=0.9, linestyle=linestyle)
        ax.plot(x, [row.color_median] * 2, color=color, lw=1.4, linestyle=linestyle)


def comparison_ylim(
    *collections: list[np.ndarray] | pd.DataFrame,
) -> tuple[float, float] | None:
    arrays: list[np.ndarray] = []
    for collection in collections:
        if isinstance(collection, pd.DataFrame):
            if not collection.empty:
                arrays.append(
                    collection[["color_min", "color_max"]].to_numpy(float).ravel()
                )
        else:
            finite_tracks = [
                np.asarray(values, dtype=float) for values in collection
                if len(values)
            ]
            if finite_tracks:
                values = np.concatenate(finite_tracks)
                values = values[np.isfinite(values)]
                if values.size:
                    arrays.append(np.nanpercentile(values, [0.05, 99.95]))
    if not arrays:
        return None
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    low, high = float(values.min()), float(values.max())
    padding = max(0.1, 0.05 * max(high - low, 0.1))
    return low - padding, high + padding


def compute_kne_snia_overlap(
    kne_rules: pd.DataFrame,
    snia_rules: pd.DataFrame,
    kne_matrix: np.ndarray,
    snia_matrix: np.ndarray,
) -> pd.DataFrame:
    """Quantify envelope intersection and cross-population containment by bin."""
    columns = [
        "bin_index", "t_min_days", "t_max_days", "t_center_days",
        "kne_color_min", "kne_color_max", "snia_color_min", "snia_color_max",
        "overlap_color_min", "overlap_color_max", "overlap_width_mag",
        "union_width_mag", "overlap_fraction_union",
        "overlap_fraction_kne_width", "overlap_fraction_snia_width",
        "N_snia_available", "N_snia_inside_kne", "fraction_snia_inside_kne",
        "N_kne_available", "N_kne_inside_snia", "fraction_kne_inside_snia",
    ]
    if kne_rules.empty or snia_rules.empty:
        return pd.DataFrame(columns=columns)
    merged = kne_rules.merge(
        snia_rules,
        on="bin_index",
        how="inner",
        suffixes=("_kne", "_snia"),
    )
    rows = []
    for row in merged.itertuples():
        bin_index = int(row.bin_index)
        kne_low, kne_high = float(row.color_min_kne), float(row.color_max_kne)
        sn_low, sn_high = float(row.color_min_snia), float(row.color_max_snia)
        overlap_low = max(kne_low, sn_low)
        overlap_high = min(kne_high, sn_high)
        overlap_width = max(0.0, overlap_high - overlap_low)
        if overlap_width == 0.0:
            stored_low = np.nan
            stored_high = np.nan
        else:
            stored_low = overlap_low
            stored_high = overlap_high
        union_width = max(kne_high, sn_high) - min(kne_low, sn_low)
        kne_width = kne_high - kne_low
        sn_width = sn_high - sn_low
        kne_values = kne_matrix[:, bin_index]
        kne_values = kne_values[np.isfinite(kne_values)]
        sn_values = snia_matrix[:, bin_index]
        sn_values = sn_values[np.isfinite(sn_values)]
        n_sn_inside = int(((sn_values >= kne_low) & (sn_values <= kne_high)).sum())
        n_kne_inside = int(((kne_values >= sn_low) & (kne_values <= sn_high)).sum())
        rows.append({
            "bin_index": bin_index,
            "t_min_days": float(row.t_min_days_kne),
            "t_max_days": float(row.t_max_days_kne),
            "t_center_days": float(row.t_center_days_kne),
            "kne_color_min": kne_low,
            "kne_color_max": kne_high,
            "snia_color_min": sn_low,
            "snia_color_max": sn_high,
            "overlap_color_min": stored_low,
            "overlap_color_max": stored_high,
            "overlap_width_mag": overlap_width,
            "union_width_mag": union_width,
            "overlap_fraction_union": (
                overlap_width / union_width if union_width > 0 else float(
                    kne_low == sn_low and kne_high == sn_high
                )
            ),
            "overlap_fraction_kne_width": (
                overlap_width / kne_width if kne_width > 0 else np.nan
            ),
            "overlap_fraction_snia_width": (
                overlap_width / sn_width if sn_width > 0 else np.nan
            ),
            "N_snia_available": len(sn_values),
            "N_snia_inside_kne": n_sn_inside,
            "fraction_snia_inside_kne": (
                n_sn_inside / len(sn_values) if len(sn_values) else np.nan
            ),
            "N_kne_available": len(kne_values),
            "N_kne_inside_snia": n_kne_inside,
            "fraction_kne_inside_snia": (
                n_kne_inside / len(kne_values) if len(kne_values) else np.nan
            ),
        })
    return pd.DataFrame(rows, columns=columns)


def summarize_overlap(
    table: pd.DataFrame,
    convention: str,
    envelope_name: str,
) -> dict[str, object]:
    if table.empty:
        return {
            "snia_convention": convention,
            "envelope": envelope_name,
            "N_common_valid_bins": 0,
            "N_bins_with_nonzero_envelope_overlap": 0,
            "fraction_bins_with_nonzero_envelope_overlap": np.nan,
            "total_overlap_fraction_union": np.nan,
            "mean_fraction_snia_inside_kne": np.nan,
            "mean_fraction_kne_inside_snia": np.nan,
        }
    total_union = table["union_width_mag"].sum()
    overlapping = table["overlap_width_mag"] > 0
    return {
        "snia_convention": convention,
        "envelope": envelope_name,
        "N_common_valid_bins": len(table),
        "N_bins_with_nonzero_envelope_overlap": int(overlapping.sum()),
        "fraction_bins_with_nonzero_envelope_overlap": float(overlapping.mean()),
        "total_overlap_fraction_union": (
            float(table["overlap_width_mag"].sum() / total_union)
            if total_union > 0 else np.nan
        ),
        "mean_overlap_fraction_union": float(
            table["overlap_fraction_union"].mean()
        ),
        "mean_fraction_snia_inside_kne": float(
            table["fraction_snia_inside_kne"].mean()
        ),
        "mean_fraction_kne_inside_snia": float(
            table["fraction_kne_inside_snia"].mean()
        ),
    }


def save_snia_population_plots(
    output: Path,
    pair: str,
    t_min: float,
    t_max: float,
    times: list[np.ndarray],
    colours: list[np.ndarray],
    rules_all: pd.DataFrame,
    rules_consecutive: pd.DataFrame,
    time_reference: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
    draw_population_style(ax, times, colours, "#d97706", 0.025, 0.20)
    format_snia_axes(
        ax, pair, t_min, t_max, comparison=False,
        time_reference=time_reference,
    )
    limits = comparison_ylim(colours)
    if limits is not None:
        ax.set_ylim(*limits)
    ax.legend(handles=[
        Line2D([0], [0], color="#d97706", lw=1.4, label="SN Ia population")
    ], frameon=False)
    fig.savefig(
        output / "snia_population_unbinned.png",
        dpi=220, bbox_inches="tight",
    )
    plt.close(fig)

    for envelope_name, rules in (
        ("all_valid", rules_all),
        ("consecutive", rules_consecutive),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
        draw_population_style(ax, times, colours, "#d97706", 0.025, 0.20)
        draw_comparison_envelope(ax, rules, "#d97706", "-", 0.20)
        format_snia_axes(
            ax, pair, t_min, t_max, comparison=False,
            time_reference=time_reference,
        )
        limits = comparison_ylim(colours, rules)
        if limits is not None:
            ax.set_ylim(*limits)
        ax.legend(handles=[
            Line2D([0], [0], color="#d97706", lw=1.6, label="SN Ia envelope")
        ], frameon=False)
        fig.savefig(
            output / f"snia_population_with_envelope_{envelope_name}.png",
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)


def save_kne_snia_overlay_plots(
    output: Path,
    pair: str,
    t_min: float,
    t_max: float,
    kne_times: list[np.ndarray],
    kne_colours: list[np.ndarray],
    snia_times: list[np.ndarray],
    snia_colours: list[np.ndarray],
    kne_rules: pd.DataFrame,
    snia_rules: pd.DataFrame,
    overlap: pd.DataFrame,
    envelope_name: str,
    time_reference: str,
) -> None:
    for show_population, stem in (
        (False, "KNe_SNIa_envelopes"),
        (True, "KNe_SNIa_population_and_envelopes"),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
        if show_population:
            draw_population_style(ax, kne_times, kne_colours, "#111111", 0.010, 0.14)
            draw_population_style(ax, snia_times, snia_colours, "#d97706", 0.018, 0.18)
        draw_comparison_envelope(ax, kne_rules, "#2f5f98", "-", 0.16)
        draw_comparison_envelope(ax, snia_rules, "#d97706", "--", 0.14)
        for row in overlap.dropna(
            subset=["overlap_color_min", "overlap_color_max"]
        ).itertuples():
            ax.fill_between(
                [row.t_min_days, row.t_max_days],
                [row.overlap_color_min] * 2,
                [row.overlap_color_max] * 2,
                color="#2ca02c", alpha=0.26,
            )
        format_snia_axes(
            ax, pair, t_min, t_max, comparison=True,
            time_reference=time_reference,
        )
        limits = comparison_ylim(
            kne_colours if show_population else [],
            snia_colours if show_population else [],
            kne_rules,
            snia_rules,
        )
        if limits is not None:
            ax.set_ylim(*limits)
        ax.legend(handles=[
            Line2D([0], [0], color="#2f5f98", lw=1.7, label="KNe envelope"),
            Line2D([0], [0], color="#d97706", lw=1.7, linestyle="--", label="SN Ia envelope"),
            Patch(facecolor="#2ca02c", alpha=0.26, label="Envelope intersection"),
        ], frameon=False, fontsize=8)
        fig.savefig(
            output / f"{stem}_{envelope_name}.png",
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)


def save_snia_comparison_outputs(
    root_output: Path,
    convention: str,
    pair: str,
    edges: np.ndarray,
    centres: np.ndarray,
    tails: list[float],
    args: argparse.Namespace,
    snia_population: dict[str, object],
    kne_matrix: np.ndarray,
    kne_times: list[np.ndarray],
    kne_colours: list[np.ndarray],
    kne_rules_all: pd.DataFrame,
    kne_rules_consecutive: pd.DataFrame,
) -> list[dict[str, object]]:
    output_name = (
        f"{convention}_{args.m5_scenario}"
        if convention in {"m5_no_mw", "m5_with_mw"}
        else convention
    )
    output = root_output / "snia_comparison" / output_name
    output.mkdir(parents=True, exist_ok=True)
    snia_matrix = snia_population["matrices"][convention]
    snia_times = snia_population["population_times"][convention]
    snia_colours = snia_population["population_colors"][convention]
    seed_offset = SNIA_BOOTSTRAP_SEED_OFFSETS[convention]
    rules_all, rules_consecutive, availability, diagnostics = (
        build_population_envelope_tables(
            snia_matrix,
            snia_population["train"],
            edges,
            centres,
            tails,
            args,
            np.random.default_rng(args.seed + 1000 * seed_offset),
            f"snia_{convention}",
            (
                "linear_in_time_since_first_observation"
                if args.snia_time_reference == "first_observation"
                else "linear_in_log10_positive_phase_since_salt2_t0"
            ),
        )
    )
    rules_all.to_csv(output / "snia_color_rules_all_valid.csv", index=False)
    rules_consecutive.to_csv(output / "snia_color_rules_consecutive.csv", index=False)
    availability.to_csv(output / "snia_envelope_availability.csv", index=False)
    if not diagnostics.empty:
        diagnostics.to_csv(output / "snia_candidate_percentile_bootstrap.csv", index=False)
    save_snia_population_plots(
        output, pair, args.t_min, args.t_max,
        snia_times, snia_colours, rules_all, rules_consecutive,
        args.snia_time_reference,
    )

    summaries = []
    for envelope_name, kne_rules, snia_rules in (
        ("all_valid", kne_rules_all, rules_all),
        ("consecutive", kne_rules_consecutive, rules_consecutive),
    ):
        overlap = compute_kne_snia_overlap(
            kne_rules, snia_rules, kne_matrix, snia_matrix
        )
        overlap.to_csv(
            output / f"KNe_SNIa_overlap_{envelope_name}.csv", index=False
        )
        save_kne_snia_overlay_plots(
            output, pair, args.t_min, args.t_max,
            kne_times, kne_colours, snia_times, snia_colours,
            kne_rules, snia_rules, overlap, envelope_name,
            args.snia_time_reference,
        )
        summaries.append(summarize_overlap(overlap, convention, envelope_name))
    last_edge = (
        float(rules_consecutive["t_max_days"].max())
        if not rules_consecutive.empty else np.nan
    )
    print(
        f"SN Ia {output_name}: {len(rules_consecutive)} consecutive valid bins; "
        f"last edge={last_edge}"
    )
    return summaries


def load_at2017gfo_colour_points(
    path: Path,
    band1: str,
    band2: str,
    merger_mjd: float,
    maximum_separation_days: float,
) -> pd.DataFrame:
    """Read AT2017gfo photometry and build unique nearest-neighbour colours.

    The supplied data use PS1 grizy and SDSS u.  These filters are treated as
    proxies for the corresponding Rubin bands; the output records the exact
    source-filter names and the inter-band time separation.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if maximum_separation_days < 0:
        raise ValueError("AT2017gfo matching separation must be non-negative")

    raw = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["date_utc", "source_filter", "magnitude", "magnitude_error"],
        usecols=[0, 1, 2, 3],
        engine="python",
    )
    raw["timestamp"] = pd.to_datetime(raw["date_utc"], utc=True, errors="coerce")
    raw["source_filter"] = raw["source_filter"].astype(str).str.strip().str.lower()
    raw["magnitude"] = pd.to_numeric(raw["magnitude"], errors="coerce")
    raw["magnitude_error"] = pd.to_numeric(raw["magnitude_error"], errors="coerce")
    raw = raw.loc[
        raw["timestamp"].notna()
        & np.isfinite(raw["magnitude"])
        & np.isfinite(raw["magnitude_error"])
        & (raw["magnitude_error"] > 0)
    ].copy()
    raw["mjd"] = raw["timestamp"].map(
        lambda value: value.to_julian_date() - 2400000.5
    )

    first = raw.loc[
        raw["source_filter"].isin(AT2017GFO_FILTER_ALIASES[band1])
    ].sort_values("mjd").reset_index(drop=True)
    second = raw.loc[
        raw["source_filter"].isin(AT2017GFO_FILTER_ALIASES[band2])
    ].sort_values("mjd").reset_index(drop=True)
    if first.empty or second.empty:
        raise ValueError(
            f"AT2017gfo file has no usable {band1!r} or {band2!r} measurements"
        )

    # Greedy one-to-one assignment from all admissible pairs, sorted by the
    # smallest separation. Most AT2017gfo pairs are exactly simultaneous.
    candidates: list[tuple[float, int, int]] = []
    for first_index, first_mjd in enumerate(first["mjd"].to_numpy(float)):
        separations = np.abs(second["mjd"].to_numpy(float) - first_mjd)
        for second_index in np.flatnonzero(
            separations <= maximum_separation_days + 1e-12
        ):
            candidates.append(
                (float(separations[second_index]), first_index, int(second_index))
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_first: set[int] = set()
    used_second: set[int] = set()
    matched: list[dict[str, object]] = []
    for separation, first_index, second_index in candidates:
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        row1 = first.iloc[first_index]
        row2 = second.iloc[second_index]
        colour_mjd = 0.5 * (float(row1["mjd"]) + float(row2["mjd"]))
        colour_timestamp = row1["timestamp"] + 0.5 * (
            row2["timestamp"] - row1["timestamp"]
        )
        colour = float(row1["magnitude"] - row2["magnitude"])
        colour_error = float(np.hypot(
            row1["magnitude_error"], row2["magnitude_error"]
        ))
        matched.append({
            "date_utc": colour_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "phase_days": colour_mjd - merger_mjd,
            "time_separation_days": separation,
            "band1": band1,
            "band2": band2,
            "source_filter_band1": row1["source_filter"],
            "source_filter_band2": row2["source_filter"],
            "date_band1_utc": row1["timestamp"].strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "date_band2_utc": row2["timestamp"].strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "magnitude_band1": float(row1["magnitude"]),
            "magnitude_error_band1": float(row1["magnitude_error"]),
            "magnitude_band2": float(row2["magnitude"]),
            "magnitude_error_band2": float(row2["magnitude_error"]),
            "color_input": colour,
            "color_error": colour_error,
        })
    if not matched:
        return pd.DataFrame(columns=[
            "date_utc", "phase_days", "time_separation_days", "band1", "band2",
            "source_filter_band1", "source_filter_band2", "date_band1_utc",
            "date_band2_utc", "magnitude_band1", "magnitude_error_band1",
            "magnitude_band2", "magnitude_error_band2", "color_input",
            "color_error",
        ])
    return pd.DataFrame(matched).sort_values("phase_days").reset_index(drop=True)


def at2017gfo_stage_colours(
    points: pd.DataFrame,
    stage: str,
    band1: str,
    band2: str,
    ebv: float,
    photometry_mode: str,
    photometric_system: str,
) -> pd.DataFrame:
    """Transform input colours to the extinction convention of one stage."""
    transformed = points.copy()
    coefficients = MW_R_BY_SYSTEM[photometric_system]
    differential_extinction = (
        coefficients[band1] - coefficients[band2]
    ) * ebv
    stage_has_mw = stage in MW_EXTINCTED_STAGES
    if photometry_mode == "dereddened":
        correction = differential_extinction if stage_has_mw else 0.0
    else:
        correction = 0.0 if stage_has_mw else -differential_extinction
    transformed["color"] = transformed["color_input"] + correction
    transformed["color_extinction_adjustment"] = correction
    transformed["photometric_stage"] = stage
    transformed["envelope_photometric_system"] = photometric_system
    return transformed


def evaluate_at2017gfo_points(
    points: pd.DataFrame,
    rules: pd.DataFrame,
    t_min: float,
    t_max: float,
    error_sigma: float,
) -> pd.DataFrame:
    """Evaluate colour/error intervals against the bin containing each epoch.

    A point is compatible when ``colour +/- error_sigma * colour_error``
    intersects the envelope.  The central-value verdict is retained separately
    so points rescued by their photometric uncertainty remain identifiable.
    """
    evaluated_rows: list[dict[str, object]] = []
    ordered = rules.sort_values("bin_index").reset_index(drop=True)
    for point in points.to_dict(orient="records"):
        row = dict(point)
        phase = float(row["phase_days"])
        row.update({
            "evaluated": False,
            "passes_filter": pd.NA,
            "passes_filter_central_value": pd.NA,
            "passes_filter_with_photometric_error": pd.NA,
            "result": "outside_requested_time_range",
            "photometric_error_sigma": float(error_sigma),
            "color_interval_min": np.nan,
            "color_interval_max": np.nan,
            "rule_id": "",
            "bin_index": pd.NA,
            "bin_t_min_days": np.nan,
            "bin_t_max_days": np.nan,
            "color_min": np.nan,
            "color_median": np.nan,
            "color_max": np.nan,
            "excursion_beyond_envelope_mag": np.nan,
        })
        if t_min <= phase <= t_max:
            matching = ordered.loc[
                (ordered["t_min_days"].astype(float) <= phase)
                & (
                    (phase < ordered["t_max_days"].astype(float))
                    | np.isclose(phase, ordered["t_max_days"].astype(float))
                )
            ]
            if matching.empty:
                row["result"] = "no_valid_envelope_bin"
            else:
                rule = matching.iloc[0]
                colour = float(row["color"])
                colour_error = float(row["color_error"])
                lower = float(rule["color_min"])
                upper = float(rule["color_max"])
                interval_lower = colour - error_sigma * colour_error
                interval_upper = colour + error_sigma * colour_error
                passes_central = lower <= colour <= upper
                passes_with_error = (
                    interval_upper >= lower and interval_lower <= upper
                )
                if interval_upper < lower:
                    result = "bluer"
                    excursion = lower - interval_upper
                elif interval_lower > upper:
                    result = "redder"
                    excursion = interval_lower - upper
                elif passes_central:
                    result = "pass_central"
                    excursion = 0.0
                else:
                    result = "pass_with_photometric_error"
                    excursion = 0.0
                row.update({
                    "evaluated": True,
                    "passes_filter": bool(passes_with_error),
                    "passes_filter_central_value": bool(passes_central),
                    "passes_filter_with_photometric_error": bool(passes_with_error),
                    "result": result,
                    "color_interval_min": float(interval_lower),
                    "color_interval_max": float(interval_upper),
                    "rule_id": str(rule["rule_id"]),
                    "bin_index": int(rule["bin_index"]),
                    "bin_t_min_days": float(rule["t_min_days"]),
                    "bin_t_max_days": float(rule["t_max_days"]),
                    "color_min": lower,
                    "color_median": float(rule["color_median"]),
                    "color_max": upper,
                    "excursion_beyond_envelope_mag": float(excursion),
                })
        evaluated_rows.append(row)
    return pd.DataFrame(evaluated_rows)


def save_at2017gfo_overlay(
    results: pd.DataFrame,
    rules: pd.DataFrame,
    path: Path,
    pair: str,
    t_min: float,
    t_max: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
    if not rules.empty:
        draw_envelope(ax, rules)
    finite = results.loc[
        np.isfinite(pd.to_numeric(results["phase_days"], errors="coerce"))
        & np.isfinite(pd.to_numeric(results["color"], errors="coerce"))
    ].sort_values("phase_days")
    if not finite.empty:
        ax.plot(
            finite["phase_days"], finite["color"], color="#222222", lw=1.1,
            alpha=0.75, zorder=4,
        )
        evaluated = finite["evaluated"].astype(bool)
        pass_values = finite["passes_filter"].astype("boolean").fillna(False).astype(bool)
        central_values = (
            finite["passes_filter_central_value"]
            .astype("boolean").fillna(False).astype(bool)
        )
        passed_central = evaluated & pass_values & central_values
        passed_via_error = evaluated & pass_values & ~central_values
        failed = evaluated & ~pass_values
        unavailable = ~evaluated
        for mask, color, marker, label, zorder in (
            (passed_central, "#009e73", "o", "AT2017gfo: central value passes", 6),
            (
                passed_via_error, "#e69f00", "D",
                "AT2017gfo: compatible within photometric error", 6,
            ),
            (failed, "#d62728", "X", "AT2017gfo: outside", 7),
            (unavailable, "#777777", "s", "AT2017gfo: no valid envelope", 5),
        ):
            subset = finite.loc[mask]
            if subset.empty:
                continue
            ax.errorbar(
                subset["phase_days"], subset["color"],
                yerr=subset["color_error"], fmt=marker, ms=5.5,
                color=color, ecolor=color, elinewidth=0.8, capsize=2,
                linestyle="none", label=label, zorder=zorder,
            )
    format_axes(ax, pair, t_min, t_max)
    arrays = []
    if not rules.empty:
        arrays.append(rules[["color_min", "color_max"]].to_numpy(float).ravel())
    if not finite.empty:
        arrays.append(finite["color"].to_numpy(float))
    if arrays:
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        if values.size:
            low, high = np.min(values), np.max(values)
            padding = max(0.1, 0.06 * max(high - low, 0.1))
            ax.set_ylim(low - padding, high + padding)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_at2017gfo_failed_table(results: pd.DataFrame, path: Path) -> None:
    failed = results.loc[
        results["evaluated"].astype(bool)
        & ~results["passes_filter"].astype("boolean").fillna(False).astype(bool)
    ].copy()
    figure_height = max(1.8, 0.42 * (len(failed) + 2))
    fig, ax = plt.subplots(figsize=(13.5, figure_height), constrained_layout=True)
    ax.axis("off")
    if failed.empty:
        ax.text(
            0.5, 0.5, "All evaluated AT2017gfo points pass",
            ha="center", va="center", fontsize=13,
        )
    else:
        display = pd.DataFrame({
            "date UTC": failed["date_utc"].str.replace("T", " ").str.slice(0, 19),
            "phase [d]": failed["phase_days"].map(lambda x: f"{x:.3f}"),
            "colour [mag]": failed.apply(
                lambda row: f"{row['color']:.3f} +/- {row['color_error']:.3f}", axis=1
            ),
            "allowed [mag]": failed.apply(
                lambda row: f"[{row['color_min']:.3f}, {row['color_max']:.3f}]", axis=1
            ),
            "side": failed["result"].astype(str),
            "excursion [mag]": failed["excursion_beyond_envelope_mag"].map(
                lambda x: f"{x:.3f}"
            ),
        })
        table = ax.table(
            cellText=display.to_numpy(), colLabels=display.columns,
            cellLoc="center", colLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.3)
        table.scale(1.0, 1.3)
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#dbe8f5")
                cell.set_text_props(weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f4f6f8")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_at2017gfo_outputs(
    stage_dir: Path,
    points: pd.DataFrame,
    rules: pd.DataFrame,
    envelope_name: str,
    pair: str,
    t_min: float,
    t_max: float,
    error_sigma: float,
) -> dict[str, int]:
    results = evaluate_at2017gfo_points(
        points, rules, t_min, t_max, error_sigma
    )
    prefix = f"AT2017gfo_{envelope_name}"
    results.to_csv(stage_dir / f"{prefix}_color_points.csv", index=False)
    failed = results.loc[
        results["evaluated"].astype(bool)
        & ~results["passes_filter"].astype("boolean").fillna(False).astype(bool)
    ].copy()
    failed.to_csv(stage_dir / f"{prefix}_failed_points.csv", index=False)
    save_at2017gfo_overlay(
        results, rules, stage_dir / f"{prefix}_with_envelope.png",
        pair, t_min, t_max,
    )
    save_at2017gfo_failed_table(
        results, stage_dir / f"{prefix}_failed_points.png"
    )
    evaluated = results["evaluated"].astype(bool)
    passed = (
        evaluated
        & results["passes_filter"].astype("boolean").fillna(False).astype(bool)
    )
    passed_central = (
        evaluated
        & results["passes_filter_central_value"]
        .astype("boolean").fillna(False).astype(bool)
    )
    passed_via_error = passed & ~passed_central
    return {
        "N_paired_points": int(len(results)),
        "N_evaluated_points": int(evaluated.sum()),
        "N_passing_points": int(passed.sum()),
        "N_passing_central_points": int(passed_central.sum()),
        "N_passing_via_photometric_error_points": int(passed_via_error.sum()),
        "N_failing_points": int((evaluated & ~passed).sum()),
    }


PERMANENT_EXIT_COLUMNS = [
    "event_id", "split", "first_permanent_outside_bin_index",
    "first_permanent_outside_time_days", "last_outside_bin_index",
    "last_outside_time_days",
    "N_final_consecutive_outside_bins", "exit_direction",
    "maximum_excursion_beyond_envelope_mag", "right_censored",
    "M_ej_dyn_Msun", "v_ej_dyn_c", "Ye_dyn", "M_ej_wind_Msun",
    "v_ej_wind_c", "Ye_wind", "inclination_deg",
    "luminosity_distance_Mpc", "redshift",
]


def identify_permanent_exits(
    matrix: np.ndarray,
    rules: pd.DataFrame,
    metadata: pd.DataFrame,
    split_map: dict[str, str],
    minimum_final_outside_bins: int,
) -> pd.DataFrame:
    """Find events whose final comparable run remains outside the envelope.

    A genuine exit must have at least one earlier comparable bin inside the
    envelope.  The final outside bins must be consecutive on the requested
    temporal grid.  Missing event values and bins without an envelope are not
    interpreted as either inside or outside.
    """
    if rules.empty:
        return pd.DataFrame(columns=PERMANENT_EXIT_COLUMNS)

    ordered = rules.sort_values("bin_index").reset_index(drop=True)
    rule_bins = ordered["bin_index"].to_numpy(dtype=int)
    lower = ordered["color_min"].to_numpy(dtype=float)
    upper = ordered["color_max"].to_numpy(dtype=float)
    times = ordered["t_center_days"].to_numpy(dtype=float)
    final_rule_bin = int(rule_bins.max())
    rows: list[dict[str, object]] = []

    for event_index, metadata_row in metadata.reset_index(drop=True).iterrows():
        values = matrix[event_index, rule_bins].astype(float)
        comparable = np.isfinite(values) & np.isfinite(lower) & np.isfinite(upper)
        comparable_indices = np.flatnonzero(comparable)
        if comparable_indices.size == 0:
            continue

        outside = (values < lower) | (values > upper)
        last_position = int(comparable_indices[-1])
        if not outside[last_position]:
            continue

        # Walk backwards over the final run. Both the rule positions and the
        # original requested-bin indices must be consecutive.
        run_positions = [last_position]
        cursor = last_position
        while cursor > 0:
            previous = cursor - 1
            if (
                not comparable[previous]
                or not outside[previous]
                or rule_bins[cursor] - rule_bins[previous] != 1
            ):
                break
            run_positions.append(previous)
            cursor = previous
        run_positions = np.array(run_positions[::-1], dtype=int)
        if len(run_positions) < minimum_final_outside_bins:
            continue

        exit_position = int(run_positions[0])
        earlier_comparable = comparable_indices[comparable_indices < exit_position]
        if not np.any(~outside[earlier_comparable]):
            # Always outside is not an "exit" from the envelope.
            continue

        run_values = values[run_positions]
        run_lower = lower[run_positions]
        run_upper = upper[run_positions]
        below = run_values < run_lower
        above = run_values > run_upper
        if np.all(above):
            direction = "redder"
        elif np.all(below):
            direction = "bluer"
        else:
            direction = "mixed"
        excursion = np.where(
            below,
            run_lower - run_values,
            np.where(above, run_values - run_upper, 0.0),
        )

        event_id = canonical_event_id(metadata_row["event_id"])
        row = {
            "event_id": event_id,
            "split": split_map.get(event_id, ""),
            "first_permanent_outside_bin_index": int(rule_bins[exit_position]),
            "first_permanent_outside_time_days": float(times[exit_position]),
            "last_outside_bin_index": int(rule_bins[last_position]),
            "last_outside_time_days": float(times[last_position]),
            "N_final_consecutive_outside_bins": int(len(run_positions)),
            "exit_direction": direction,
            "maximum_excursion_beyond_envelope_mag": float(np.max(excursion)),
            "right_censored": bool(rule_bins[last_position] < final_rule_bin),
            "M_ej_dyn_Msun": 10.0 ** float(metadata_row["log10_mej_dyn"]),
            "v_ej_dyn_c": float(metadata_row["v_ej_dyn"]),
            "Ye_dyn": float(metadata_row["Ye_dyn"]),
            "M_ej_wind_Msun": 10.0 ** float(metadata_row["log10_mej_wind"]),
            "v_ej_wind_c": float(metadata_row["v_ej_wind"]),
            "Ye_wind": float(metadata_row["Ye_wind"]),
            "inclination_deg": float(np.degrees(metadata_row["inclination_EM"])),
            "luminosity_distance_Mpc": float(metadata_row["luminosity_distance"]),
            "redshift": float(metadata_row["redshift"]),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=PERMANENT_EXIT_COLUMNS)
    return (
        pd.DataFrame(rows, columns=PERMANENT_EXIT_COLUMNS)
        .sort_values(
            ["N_final_consecutive_outside_bins", "last_outside_time_days",
             "first_permanent_outside_time_days", "event_id"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )


def save_permanent_exit_table_png(
    exits: pd.DataFrame,
    path: Path,
    maximum_rows: int,
    event_colors: dict[str, str] | None = None,
) -> None:
    """Save a compact, presentation-ready physical-parameter table."""
    display_columns = [
        "event_id", "last_outside_time_days", "exit_direction",
        "M_ej_dyn_Msun", "v_ej_dyn_c", "Ye_dyn", "M_ej_wind_Msun",
        "v_ej_wind_c", "Ye_wind", "inclination_deg",
        "luminosity_distance_Mpc",
    ]
    labels = [
        "event", "last out [d]", "side",
        r"Mdyn [$M_\odot$]", "vdyn/c", "Ye dyn", r"Mwind [$M_\odot$]",
        "vwind/c", "Ye wind", "incl. [deg]", "DL [Mpc]",
    ]
    shown = exits.head(maximum_rows).copy()
    figure_height = max(1.8, 0.38 * (len(shown) + 2))
    fig, ax = plt.subplots(figsize=(18.0, figure_height), constrained_layout=True)
    ax.axis("off")
    if shown.empty:
        ax.text(0.5, 0.5, "No permanent exits", ha="center", va="center", fontsize=13)
    else:
        formatted = pd.DataFrame({
            "event_id": shown["event_id"].astype(str),
            "last_outside_time_days": shown["last_outside_time_days"].map(
                lambda x: f"{x:.2f}"
            ),
            "exit_direction": shown["exit_direction"].astype(str),
            "M_ej_dyn_Msun": shown["M_ej_dyn_Msun"].map(lambda x: f"{x:.3e}"),
            "v_ej_dyn_c": shown["v_ej_dyn_c"].map(lambda x: f"{x:.3f}"),
            "Ye_dyn": shown["Ye_dyn"].map(lambda x: f"{x:.3f}"),
            "M_ej_wind_Msun": shown["M_ej_wind_Msun"].map(lambda x: f"{x:.3e}"),
            "v_ej_wind_c": shown["v_ej_wind_c"].map(lambda x: f"{x:.3f}"),
            "Ye_wind": shown["Ye_wind"].map(lambda x: f"{x:.3f}"),
            "inclination_deg": shown["inclination_deg"].map(lambda x: f"{x:.1f}"),
            "luminosity_distance_Mpc": shown["luminosity_distance_Mpc"].map(
                lambda x: f"{x:.1f}"
            ),
        })
        table = ax.table(
            cellText=formatted[display_columns].to_numpy(),
            colLabels=labels,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1.0, 1.25)
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#dbe8f5")
                cell.set_text_props(weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f4f6f8")
        if event_colors:
            for row_index, event_id in enumerate(shown["event_id"].astype(str), start=1):
                table[(row_index, 0)].set_text_props(
                    color=event_colors.get(event_id, "black"),
                    weight="bold",
                )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stage_plots(
    stage_dir: Path,
    pair: str,
    t_min: float,
    t_max: float,
    population_times: list[np.ndarray],
    population_colors: list[np.ndarray],
    all_valid: pd.DataFrame,
    consecutive: pd.DataFrame,
    coloured_samples: dict[int, np.ndarray],
    colour_by_event_index: dict[int, str],
    event_ids: list[str],
) -> None:
    specifications = [
        ("complete_population_unbinned.png", True, None),
        ("envelope_all_valid.png", False, all_valid),
        ("envelope_consecutive.png", False, consecutive),
        ("complete_population_unbinned_with_envelope_all_valid.png", True, all_valid),
        ("complete_population_unbinned_with_envelope_consecutive.png", True, consecutive),
    ]
    flattened_colors = (
        np.concatenate(population_colors) if population_colors else np.array([])
    )
    flattened_colors = flattened_colors[np.isfinite(flattened_colors)]
    rule_arrays = []
    for table in (all_valid, consecutive):
        if table is not None and not table.empty:
            for column in ("color_min", "color_median", "color_max"):
                if column in table:
                    rule_arrays.append(table[column].to_numpy(dtype=float))
    finite_rule_colors = (
        np.concatenate(rule_arrays) if rule_arrays else np.array([], dtype=float)
    )
    finite_rule_colors = finite_rule_colors[np.isfinite(finite_rule_colors)]
    for filename, show_population, rules in specifications:
        fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
        if show_population:
            draw_population(ax, population_times, population_colors)
        if rules is not None and not rules.empty:
            draw_envelope(ax, rules)
        format_axes(ax, pair, t_min, t_max)
        if len(flattened_colors):
            low, high = np.nanpercentile(flattened_colors, [0.05, 99.95])
            if finite_rule_colors.size:
                low = min(low, float(np.min(finite_rule_colors)))
                high = max(high, float(np.max(finite_rule_colors)))
            padding = max(0.1, 0.05 * (high - low))
            ax.set_ylim(low - padding, high + padding)
        fig.savefig(stage_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Readable event-level views.  The samples are nested and identical for
    # every photometric stage; colours are also kept fixed between stages.
    for sample_size, indices in coloured_samples.items():
        selected_times = [population_times[index] for index in indices]
        selected_colors = [population_colors[index] for index in indices]
        trajectory_colors = [colour_by_event_index[index] for index in indices]
        selected_event_ids = [event_ids[int(index)] for index in indices]
        legend_handles = [
            Line2D(
                [0], [0], color=color, lw=1.6,
                label=f"event {event_id}",
            )
            for event_id, color in zip(selected_event_ids, trajectory_colors)
        ]
        legend_columns = min(10, max(1, int(np.ceil(np.sqrt(sample_size)))))
        legend_rows = int(np.ceil(sample_size / legend_columns))
        show_legend = sample_size <= 10
        coloured_specifications = [
            (f"population_{sample_size}_random_coloured.png", None),
            (
                f"population_{sample_size}_random_coloured_with_envelope_all_valid.png",
                all_valid,
            ),
            (
                f"population_{sample_size}_random_coloured_with_envelope_consecutive.png",
                consecutive,
            ),
        ]
        for filename, rules in coloured_specifications:
            if show_legend:
                figure_height = 5.7 + 0.10 * max(0, legend_rows - 3)
                fig, ax = plt.subplots(figsize=(9.2, figure_height))
                fig.subplots_adjust(bottom=min(0.32, 0.16 + 0.023 * legend_rows))
            else:
                fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
            draw_population(
                ax,
                selected_times,
                selected_colors,
                trajectory_colors=trajectory_colors,
            )
            if rules is not None and not rules.empty:
                draw_envelope(ax, rules)
            format_axes(ax, pair, t_min, t_max)
            if len(flattened_colors):
                low, high = np.nanpercentile(flattened_colors, [0.05, 99.95])
                if finite_rule_colors.size:
                    low = min(low, float(np.min(finite_rule_colors)))
                    high = max(high, float(np.max(finite_rule_colors)))
                padding = max(0.1, 0.05 * (high - low))
                ax.set_ylim(low - padding, high + padding)
            if show_legend:
                fig.legend(
                    handles=legend_handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.015),
                    ncol=legend_columns,
                    frameon=False,
                    fontsize=7.2,
                    handlelength=1.6,
                    columnspacing=1.0,
                )
            fig.savefig(stage_dir / filename, dpi=220, bbox_inches="tight")
            plt.close(fig)


def stage_output_name(stage: str, scenario: str) -> str:
    return f"{stage}_{scenario}" if stage in M5_STAGES else stage


def save_dual_system_comparison_plots(
    output: Path,
    pair: str,
    scenario: str,
    t_min: float,
    t_max: float,
    at2017gfo_enabled: bool,
    systems: tuple[str, str] = ("lsst", "ps1"),
) -> None:
    """Overlay two survey envelopes built from identical physical events.

    AT2017gfo measurements are plotted only for the historical LSST/PS1
    comparison. They are PS1 photometry and must not be relabelled as ZTF.
    """
    system_style = {
        "lsst": ("#2f5f98", "Rubin/LSST envelope"),
        "ps1": ("#cc6677", "Pan-STARRS1 envelope"),
        "ztf": ("#dd8d29", "ZTF envelope"),
    }
    for stage in STAGES:
        stage_name = stage_output_name(stage, scenario)
        for envelope_name, rules_filename in (
            ("all_valid", "dynamic_color_cut_rules_all_valid.csv"),
            ("consecutive", "dynamic_color_cut_rules.csv"),
        ):
            rules_by_system: dict[str, pd.DataFrame] = {}
            for system in systems:
                path = output / system / stage_name / rules_filename
                if path.exists():
                    rules_by_system[system] = pd.read_csv(path)
            if not rules_by_system:
                continue

            fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
            handles: list[Line2D] = []
            color_arrays = []
            for system, rules in rules_by_system.items():
                color, label = system_style[system]
                for row in rules.itertuples():
                    x = [row.t_min_days, row.t_max_days]
                    ax.fill_between(
                        x,
                        [row.color_min, row.color_min],
                        [row.color_max, row.color_max],
                        color=color,
                        alpha=0.14,
                    )
                    ax.plot(x, [row.color_min] * 2, color=color, lw=0.9)
                    ax.plot(x, [row.color_max] * 2, color=color, lw=0.9)
                    ax.plot(
                        x, [row.color_median] * 2, color=color, lw=1.35,
                        linestyle="-" if system == systems[0] else "--",
                    )
                if not rules.empty:
                    color_arrays.append(
                        rules[["color_min", "color_max"]].to_numpy(float).ravel()
                    )
                handles.append(Line2D(
                    [0], [0], color=color, lw=1.8,
                    linestyle="-" if system == systems[0] else "--",
                    label=label,
                ))

            at_points = pd.DataFrame()
            if at2017gfo_enabled and "ps1" in systems:
                candidate = (
                    output / "ps1" / stage_name
                    / f"AT2017gfo_{envelope_name}_color_points.csv"
                )
                if candidate.exists():
                    at_points = pd.read_csv(candidate)
                    at_points = at_points.loc[
                        np.isfinite(pd.to_numeric(at_points["phase_days"], errors="coerce"))
                        & np.isfinite(pd.to_numeric(at_points["color"], errors="coerce"))
                    ].sort_values("phase_days")
                    if not at_points.empty:
                        ax.errorbar(
                            at_points["phase_days"], at_points["color"],
                            yerr=at_points["color_error"], color="#111111",
                            marker="o", ms=4.5, lw=1.0, elinewidth=0.8,
                            capsize=2, zorder=6,
                        )
                        handles.append(Line2D(
                            [0], [0], color="#111111", marker="o", lw=1.0,
                            label="AT2017gfo (PS1 photometry)",
                        ))
                        color_arrays.append(at_points["color"].to_numpy(float))

            format_axes(ax, pair, t_min, t_max)
            if color_arrays:
                values = np.concatenate(color_arrays)
                values = values[np.isfinite(values)]
                if values.size:
                    low, high = float(values.min()), float(values.max())
                    padding = max(0.1, 0.06 * max(high - low, 0.1))
                    ax.set_ylim(low - padding, high + padding)
            ax.legend(handles=handles, frameon=False, fontsize=8)
            comparison_prefix = "_".join(system.upper() for system in systems)
            fig.savefig(
                output
                / f"{comparison_prefix}_{stage}_{envelope_name}_comparison.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)


def main(args: argparse.Namespace | None = None) -> Path:
    """Build one system, or dispatch a paired two-system analysis.

    The dual-system branch deliberately reuses the same event split, random
    seed, temporal grid and physical population.  Only the photometric
    throughput used for the colour changes. PS1 uses Rubin detectability by
    construction; ZTF uses its own empirical limiting-magnitude distribution.
    """
    if args is None:
        args = build_parser().parse_args()
        if args.interactive or len(sys.argv) == 1:
            args = interactive(args)

    pieces = args.color_pair.lower().replace("_", "-").split("-")
    if len(pieces) != 2 or pieces[0] == pieces[1] or any(b not in RUBIN_MW_R for b in pieces):
        raise ValueError("Colour pair must look like g-r")
    band1, band2 = pieces
    pair = f"{band1}-{band2}"

    if args.photometric_system in {"ps1", "both"} and "u" in pieces:
        raise ValueError(
            "PS1 has no u band. Choose an overlap colour such as g-r, r-i, "
            "i-z or z-y, or run --photometric-system lsst."
        )
    if args.photometric_system in {"ztf", "lsst+ztf"} and any(
        band not in ZTF_MW_R for band in pieces
    ):
        raise ValueError(
            "ZTF supports only g, r and i. Choose g-r, g-i or r-i."
        )

    if args.bin_width <= 0 or args.t_max <= args.t_min:
        raise ValueError("Require bin-width > 0 and t-max > t-min")
    if args.minimum_final_outside_bins < 1:
        raise ValueError("minimum-final-outside-bins must be at least 1")
    if args.max_outlier_table_events < 1:
        raise ValueError("max-outlier-table-events must be at least 1")
    if args.outlier_table_plot_size < 1:
        raise ValueError("outlier-table-plot-size must be at least 1")
    if args.at2017gfo_match_window_days < 0:
        raise ValueError("at2017gfo-match-window-days must be non-negative")
    if args.at2017gfo_ebv < 0:
        raise ValueError("at2017gfo-ebv must be non-negative")
    if args.at2017gfo_error_sigma <= 0:
        raise ValueError("at2017gfo-error-sigma must be strictly positive")
    if args.at2017gfo_file is not None and args.photometric_system == "ztf":
        raise ValueError(
            "AT2017gfo input points are PS1 photometry, not ZTF. Build the "
            "AT2017gfo overlay with --photometric-system ps1 (or both)."
        )
    if args.snia_max_events < 0:
        raise ValueError("snia-max-events must be non-negative")
    if args.snia_parquet is not None and args.photometric_system in {"ps1", "ztf"}:
        raise ValueError(
            "The supplied SN Ia mock contains Rubin/LSST ugrizy photometry, "
            f"not {args.photometric_system.upper()} photometry. Use an LSST mode."
        )
    if not 0.0 <= args.latitude_threshold_deg <= 90.0:
        raise ValueError("latitude-threshold-deg must lie in [0, 90]")
    edges = np.arange(args.t_min, args.t_max + 0.5 * args.bin_width, args.bin_width)
    if not np.isclose(edges[-1], args.t_max, atol=1e-8):
        raise ValueError("(t-max - t-min) must be an integer multiple of bin-width")
    centres = 0.5 * (edges[:-1] + edges[1:])

    if args.photometric_system in {"both", "lsst+ztf"}:
        systems = (
            ("lsst", "ps1")
            if args.photometric_system == "both"
            else ("lsst", "ztf")
        )
        comparison_slug = "_".join(systems)
        comparison_label = "+".join(system.upper() for system in systems)
        output = args.output_dir.expanduser().resolve() if args.output_dir else (
            run_root(args.run_dir)
            / "analysis"
            / (
                f"four_stage_{band1}_{band2}_envelopes_"
                f"{args.m5_scenario}_{comparison_slug}_{population_latitude_tag(args)}"
            )
        )
        output.mkdir(parents=True, exist_ok=True)
        for system in systems:
            child_args = copy.deepcopy(args)
            child_args.photometric_system = system
            child_args.output_dir = output / system
            child_args.interactive = False
            if system != "lsst":
                # The supplied SN Ia population is Rubin/LSST only.
                child_args.snia_parquet = None
            if args.photometric_system == "lsst+ztf":
                # AT2017gfo points in this pipeline are PS1 photometry, not ZTF.
                child_args.at2017gfo_file = None
            main(child_args)
        save_dual_system_comparison_plots(
            output,
            pair,
            args.m5_scenario,
            args.t_min,
            args.t_max,
            args.at2017gfo_file is not None and "ps1" in systems,
            systems=systems,
        )
        (output / f"{comparison_label.replace('+', '_')}_comparison_configuration.json").write_text(
            json.dumps(
                {
                    "color_pair": pair,
                    "systems": list(systems),
                    "same_injected_events": True,
                    "same_event_split_and_random_seed": True,
                    "same_OpSim_field_per_event": True,
                    "depth_selection_system_by_envelope": {
                        "lsst": "lsst",
                        systems[1]: "lsst" if systems[1] == "ps1" else "ztf",
                    },
                    "m5_scenario": args.m5_scenario,
                    "population_latitude_selection": (
                        population_latitude_label(args)
                    ),
                    "population_latitude_threshold_deg": (
                        args.latitude_threshold_deg
                        if args.population_latitude_selection != "all"
                        else None
                    ),
                    "AT2017gfo_photometric_system": (
                        "ps1" if "ps1" in systems and args.at2017gfo_file is not None
                        else None
                    ),
                    "SN_Ia_comparison_system": (
                        "lsst" if args.snia_parquet is not None else None
                    ),
                    "interpretation": (
                        "PS1 uses the corresponding Rubin LSST bands for the "
                        "OpSim depth selection. ZTF uses its own supplied "
                        "per-band limiting-magnitude thresholds."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Combined {comparison_label} output:", output)
        return output

    synthetic_source = SyntheticPopulationSource(args.run_dir)
    if synthetic_source.kind == "missing":
        raise FileNotFoundError(
            "No synthetic population was found. Expected either "
            "parquet/synthetic_part_*.parquet or csv/lightcurve_XXXX.csv below "
            f"{run_root(args.run_dir)}"
        )
    available_event_ids = synthetic_source.available_event_ids()
    metadata = prepare_metadata(args, available_event_ids)
    print(f"Synthetic input source: {synthetic_source.kind}")
    event_ids = metadata["event_id"].tolist()
    requested_plot_sizes_float = parse_floats(args.colored_plot_sizes)
    if any(
        value <= 0 or not np.isclose(value, round(value))
        for value in requested_plot_sizes_float
    ):
        raise ValueError("Coloured plot sizes must be positive integers")
    requested_plot_sizes = sorted(
        set(int(round(value)) for value in requested_plot_sizes_float)
    )
    if args.outlier_table_plot_size not in requested_plot_sizes:
        requested_plot_sizes.append(args.outlier_table_plot_size)
        requested_plot_sizes.sort()
    plot_rng = np.random.default_rng(args.population_plot_seed)
    plot_permutation = plot_rng.permutation(len(event_ids))
    coloured_samples: dict[int, np.ndarray] = {}
    for requested_size in requested_plot_sizes:
        actual_size = min(requested_size, len(event_ids))
        coloured_samples[actual_size] = plot_permutation[:actual_size].copy()
    table_plot_size = min(args.outlier_table_plot_size, len(event_ids))
    table_plot_indices = coloured_samples[table_plot_size]
    table_plot_event_ids = [event_ids[int(index)] for index in table_plot_indices]
    table_plot_event_id_set = set(table_plot_event_ids)
    maximum_plot_size = max(coloured_samples, default=0)
    maximum_plot_indices = plot_permutation[:maximum_plot_size]
    golden_ratio_conjugate = 0.6180339887498949
    colour_by_event_index = {
        int(event_index): to_hex(
            hsv_to_rgb(
                (
                    (rank * golden_ratio_conjugate) % 1.0,
                    0.72,
                    0.82,
                )
            )
        )
        for rank, event_index in enumerate(maximum_plot_indices)
    }
    table_event_colors = {
        event_ids[int(event_index)]: colour_by_event_index[int(event_index)]
        for event_index in table_plot_indices
    }
    split = assign_event_splits(
        event_ids, args.train_fraction, args.validation_fraction, args.seed
    )
    split_map = split.set_index("event_id")["split"].to_dict()
    train = np.array([split_map[eid] == "train" for eid in event_ids], dtype=bool)

    output = args.output_dir.expanduser().resolve() if args.output_dir else (
        run_root(args.run_dir)
        / "analysis"
        / (
            f"four_stage_{band1}_{band2}_envelopes_"
            f"{args.m5_scenario}_{args.photometric_system}_"
            f"{population_latitude_tag(args)}"
        )
    )
    output.mkdir(parents=True, exist_ok=True)

    ztf_depth_table = pd.DataFrame()
    if args.photometric_system == "ztf":
        manual_ztf_depths = {
            "g": args.ztf_m5_g,
            "r": args.ztf_m5_r,
            "i": args.ztf_m5_i,
        }
        m5, ztf_depth_table = load_ztf_depth_thresholds(
            directory=args.ztf_depth_dir,
            quantile_table=args.ztf_depth_quantile_table,
            scenario=args.m5_scenario,
            bands=(band1, band2),
            n_samples=args.ztf_depth_samples,
            seed=args.ztf_depth_seed,
            depth_source=args.ztf_depth_source,
            manual_thresholds=manual_ztf_depths,
        )
        recorded_ztf_depth_table = output / "ztf_depth_thresholds_used.csv"
        ztf_depth_table.to_csv(recorded_ztf_depth_table, index=False)
        m5_table = (
            args.ztf_depth_quantile_table.expanduser().resolve()
            if args.ztf_depth_quantile_table is not None
            and (args.ztf_depth_source in {None, "table"})
            else recorded_ztf_depth_table
        )
        print(
            "ZTF depth thresholds: "
            + ", ".join(f"{band}={value:.4f}" for band, value in m5.items())
        )
    else:
        m5_table = discover_m5_table(args.run_dir, args.m5_quantile_table)
        m5 = load_m5(
            m5_table, args.threshold_scope, args.m5_scenario, (band1, band2)
        )

    split.to_csv(output / "event_split.csv", index=False)
    at2017gfo_input = None
    if args.at2017gfo_file is not None:
        at2017gfo_input = load_at2017gfo_colour_points(
            args.at2017gfo_file,
            band1,
            band2,
            args.at2017gfo_merger_mjd,
            args.at2017gfo_match_window_days,
        )
        at2017gfo_input.to_csv(
            output / "AT2017gfo_paired_input_photometry.csv", index=False
        )
        print(
            f"AT2017gfo: {len(at2017gfo_input)} unique {pair} pairs within "
            f"{args.at2017gfo_match_window_days:g} days"
        )
    selection_rows = []
    for rank, event_index in enumerate(maximum_plot_indices, start=1):
        row = {
            "event_id": event_ids[int(event_index)],
            "random_order": rank,
            "trajectory_color": colour_by_event_index[int(event_index)],
        }
        for sample_size, indices in coloured_samples.items():
            row[f"included_in_{sample_size}"] = bool(event_index in set(indices))
        selection_rows.append(row)
    pd.DataFrame(selection_rows).to_csv(
        output / "population_plot_random_selection.csv", index=False
    )

    n_events, n_times = len(metadata), len(centres)
    int1 = np.full((n_events, n_times), np.nan, dtype=np.float32)
    int2 = np.full_like(int1, np.nan)
    ext1 = np.full_like(int1, np.nan)
    ext2 = np.full_like(int1, np.nan)
    # The colour system and the survey selection system are deliberately
    # separate only for PS1. ZTF and LSST are selected with their own depths;
    # PS1 detectability is evaluated in corresponding saved Rubin bands.
    selection_int1 = np.full_like(int1, np.nan)
    selection_int2 = np.full_like(int1, np.nan)
    selection_ext1 = np.full_like(int1, np.nan)
    selection_ext2 = np.full_like(int1, np.nan)
    population_times = {stage: [None] * n_events for stage in STAGES}
    population_colors = {stage: [None] * n_events for stage in STAGES}

    print(
        f"Reading {n_events:,} saved synthetic light curves; numerical support "
        "is inherited from generate_kne_lightcurves.py"
    )
    event_position = {
        canonical_event_id(event_id): index
        for index, event_id in enumerate(event_ids)
    }
    metadata_by_event = metadata.set_index("event_id", drop=False)
    processed_events = 0
    processed_event_ids: set[str] = set()
    progress_interval = max(500, n_events // 20)
    selection_system = (
        "lsst" if args.photometric_system == "ps1" else args.photometric_system
    )
    requested_systems = tuple(dict.fromkeys(
        (args.photometric_system, selection_system)
    ))
    for event_id, lightcurve in synthetic_source.iter_events(
        event_ids, (band1, band2), requested_systems
    ):
        event_id = canonical_event_id(event_id)
        if event_id in processed_event_ids:
            raise RuntimeError(
                f"Synthetic event {event_id} occurs in more than one input group/shard"
            )
        processed_event_ids.add(event_id)
        matrix_index = event_position[event_id]
        row = metadata_by_event.loc[event_id]
        required_columns = {"t_days", "band", "mag"}
        missing_columns = required_columns.difference(lightcurve.columns)
        if missing_columns:
            raise ValueError(
                f"Synthetic event {event_id} is missing {sorted(missing_columns)}"
            )
        lightcurve = pd.DataFrame({
            "t_days": pd.to_numeric(lightcurve["t_days"], errors="coerce"),
            "band": lightcurve["band"].astype(str).str.strip().str.lower(),
            "magnitude": pd.to_numeric(lightcurve["mag"], errors="coerce"),
            "photometric_system": (
                lightcurve["photometric_system"].astype(str).str.strip().str.lower()
                if "photometric_system" in lightcurve
                else "lsst"
            ),
        })
        lightcurve = lightcurve.loc[
            np.isfinite(lightcurve["t_days"])
            & (lightcurve["t_days"] > 0)
            & np.isfinite(lightcurve["magnitude"])
            & lightcurve["band"].isin((band1, band2))
            & lightcurve["photometric_system"].isin(requested_systems)
        ].copy()

        target_curve = lightcurve.loc[
            lightcurve["photometric_system"] == args.photometric_system
        ].copy()
        selection_curve = lightcurve.loc[
            lightcurve["photometric_system"] == selection_system
        ].copy()
        mw1 = event_extinction_mag(row, args.photometric_system, band1)
        mw2 = event_extinction_mag(row, args.photometric_system, band2)
        selection_mw1 = event_extinction_mag(row, selection_system, band1)
        selection_mw2 = event_extinction_mag(row, selection_system, band2)

        # Saved magnitudes already include MW extinction. Interpolation is
        # restricted to the saved support of each band, so it cannot recreate
        # samples beyond the numerical-floor cut made during generation.
        centre_ext1 = interpolate_band(target_curve, band1, centres)
        centre_ext2 = interpolate_band(target_curve, band2, centres)
        ext1[matrix_index] = centre_ext1
        ext2[matrix_index] = centre_ext2
        int1[matrix_index] = centre_ext1 - mw1
        int2[matrix_index] = centre_ext2 - mw2
        centre_selection_ext1 = interpolate_band(selection_curve, band1, centres)
        centre_selection_ext2 = interpolate_band(selection_curve, band2, centres)
        selection_ext1[matrix_index] = centre_selection_ext1
        selection_ext2[matrix_index] = centre_selection_ext2
        selection_int1[matrix_index] = centre_selection_ext1 - selection_mw1
        selection_int2[matrix_index] = centre_selection_ext2 - selection_mw2

        # Colours on the population plots are evaluated only at saved times
        # where both bands exist. No population binning or surrogate call is
        # involved.
        paired = (
            target_curve.groupby(["t_days", "band"], as_index=False)["magnitude"]
            .median()
            .pivot(index="t_days", columns="band", values="magnitude")
        )
        if band1 not in paired.columns or band2 not in paired.columns:
            times = np.array([], dtype=float)
            saved1 = np.array([], dtype=float)
            saved2 = np.array([], dtype=float)
        else:
            paired = paired.dropna(subset=[band1, band2]).sort_index()
            times = paired.index.to_numpy(float)
            saved1 = paired[band1].to_numpy(float)
            saved2 = paired[band2].to_numpy(float)
        native_support = (
            np.isfinite(times)
            & np.isfinite(saved1)
            & np.isfinite(saved2)
            & (times >= args.t_min)
            & (times <= args.t_max)
        )
        intrinsic_colour = (saved1 - mw1) - (saved2 - mw2)
        extincted_colour = saved1 - saved2
        for stage in STAGES:
            # The visual background remains the complete population.  The m5
            # masks are applied only to the samples used to build the rules.
            mask = native_support
            population_times[stage][matrix_index] = times[mask].astype(np.float32)
            colour = (
                extincted_colour
                if stage in MW_EXTINCTED_STAGES
                else intrinsic_colour
            )
            population_colors[stage][matrix_index] = colour[mask].astype(np.float32)

        processed_events += 1
        if processed_events % progress_interval == 0:
            print(f"  {processed_events:,}/{n_events:,}")

    # ``summary.csv`` records whether an event has any saved synthetic Parquet
    # product.  An event can nevertheless have no saved row in either of the
    # two requested bands (for example after the generator's numerical-support
    # cut).  Such an event is a legitimate injected-but-unavailable event for
    # this colour; it must not abort the full population analysis.
    missing_event_ids = [
        canonical_event_id(event_id) for event_id in event_ids
        if canonical_event_id(event_id) not in processed_event_ids
    ]
    if missing_event_ids:
        empty = np.array([], dtype=np.float32)
        for event_id in missing_event_ids:
            matrix_index = event_position[event_id]
            for stage in STAGES:
                population_times[stage][matrix_index] = empty.copy()
                population_colors[stage][matrix_index] = empty.copy()
        missing_table = metadata.loc[
            metadata["event_id"].isin(missing_event_ids)
        ].copy()
        missing_table.insert(
            1, "unavailability_reason", "no_saved_rows_in_requested_bands"
        )
        missing_table.insert(2, "requested_colour_pair", pair)
        missing_table.to_csv(
            output / "events_without_requested_colour_rows.csv", index=False
        )
        print(
            f"Warning: {len(missing_event_ids):,}/{n_events:,} injected events have "
            f"no saved row in either requested band ({band1}, {band2}). They are "
            "retained in N_injected and treated as colour-unavailable. Details: "
            f"{output / 'events_without_requested_colour_rows.csv'}"
        )
    print(
        f"Finished reading: {processed_events:,} events with saved {pair} rows; "
        f"{len(missing_event_ids):,} colour-unavailable events."
    )

    support = np.isfinite(int1) & np.isfinite(int2)
    selection_support = np.isfinite(selection_int1) & np.isfinite(selection_int2)
    intrinsic_color = int1 - int2
    extincted_color = ext1 - ext2
    tails = parse_floats(args.tail_percentiles)
    for fallback in (25, 30, 35, 40, 42, 44, 45, 46, 47, 48, 49, 50):
        if fallback not in tails:
            tails.append(float(fallback))

    snia_population: dict[str, object] = {}
    if args.snia_parquet is not None:
        snia_population = prepare_snia_population(
            args, metadata, band1, band2, centres, m5, output
        )

    summaries = []
    snia_overlap_summaries: list[dict[str, object]] = []
    for stage in STAGES:
        if stage == "intrinsic_no_m5_no_mw":
            keep = support
            colour_matrix = intrinsic_color
        elif stage == "no_m5_with_mw":
            keep = support
            colour_matrix = extincted_color
        elif stage == "m5_no_mw":
            keep = (
                support
                & selection_support
                & (selection_int1 <= m5[band1])
                & (selection_int2 <= m5[band2])
            )
            colour_matrix = intrinsic_color
        else:
            keep = (
                support
                & np.isfinite(selection_ext1)
                & np.isfinite(selection_ext2)
                & (selection_ext1 <= m5[band1])
                & (selection_ext2 <= m5[band2])
            )
            colour_matrix = extincted_color
        matrix = np.where(keep, colour_matrix, np.nan).astype(np.float32)

        # Make the adopted depth scenario visible in every m5-dependent path.
        # This also prevents median/worst/best products from overwriting one
        # another when an explicit common --output-dir is supplied.
        stage_dir = output / stage_output_name(stage, args.m5_scenario)
        stage_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(
            args.seed + 1000 * STAGE_BOOTSTRAP_SEED_OFFSETS[stage]
        )
        rules, diagnostic_frames, availability = [], [], []
        consecutive_open = True
        for bin_index, centre in enumerate(centres):
            values = matrix[train, bin_index]
            values = values[np.isfinite(values)].astype(float)
            if len(values) >= args.minimum_events:
                diagnostics, selected = percentile_diagnostics(
                    values,
                    tails,
                    args.n_boot,
                    args.bootstrap_threshold_mag,
                    args.minimum_tail_events,
                    rng,
                )
            else:
                diagnostics, selected = pd.DataFrame(), None
            if not diagnostics.empty:
                diagnostics.insert(0, "bin_index", bin_index)
                diagnostics.insert(1, "t_min_days", edges[bin_index])
                diagnostics.insert(2, "t_max_days", edges[bin_index + 1])
                diagnostics.insert(3, "t_center_days", centre)
                diagnostics.insert(4, "N_train_events", len(values))
                diagnostic_frames.append(diagnostics)
            valid = selected is not None
            if not valid:
                consecutive_open = False
            availability.append({
                "bin_index": bin_index,
                "t_min_days": edges[bin_index],
                "t_max_days": edges[bin_index + 1],
                "t_center_days": centre,
                "N_train_total": int(train.sum()),
                "N_train_model_supported": int(support[train, bin_index].sum()),
                "N_train_stage_available": len(values),
                "candidate_envelope_valid": valid,
                "initial_consecutive_valid": bool(valid and consecutive_open),
            })
            if valid:
                rules.append({
                    "rule_id": f"{stage}_bin_{bin_index:03d}",
                    "bin_index": bin_index,
                    "stage": stage,
                    "t_min_days": edges[bin_index],
                    "t_max_days": edges[bin_index + 1],
                    "t_center_days": centre,
                    "bin_width_days": args.bin_width,
                    "N_train_events": len(values),
                    "initial_consecutive_valid": bool(consecutive_open),
                    **selected,
                    "bootstrap_threshold_mag": args.bootstrap_threshold_mag,
                    "minimum_tail_events": args.minimum_tail_events,
                    "target_interpolation_time_days": centre,
                    "interpolation_method": "linear_in_log10_time at fixed bin centre",
                })

        rules_all = pd.DataFrame(rules)
        if rules_all.empty:
            rules_all = pd.DataFrame(columns=[
                "rule_id", "bin_index", "stage", "t_min_days", "t_max_days", "t_center_days",
                "bin_width_days", "N_train_events", "initial_consecutive_valid",
                "low_percentile", "high_percentile", "color_min", "color_median",
                "color_max",
            ])
        rules_consecutive = rules_all.loc[
            rules_all["initial_consecutive_valid"].astype(bool)
        ].copy()
        rules_all.to_csv(stage_dir / "dynamic_color_cut_rules_all_valid.csv", index=False)
        rules_consecutive.to_csv(stage_dir / "dynamic_color_cut_rules.csv", index=False)
        pd.DataFrame(availability).to_csv(
            stage_dir / "envelope_availability_diagnostics.csv", index=False
        )
        if diagnostic_frames:
            pd.concat(diagnostic_frames, ignore_index=True).to_csv(
                stage_dir / "all_candidate_percentile_bootstrap.csv", index=False
            )

        exit_counts = {}
        all_event_exit_counts = {}
        for envelope_name, envelope_rules in (
            ("all_valid", rules_all),
            ("consecutive", rules_consecutive),
        ):
            all_permanent_exits = identify_permanent_exits(
                matrix,
                envelope_rules,
                metadata,
                split_map,
                args.minimum_final_outside_bins,
            )
            all_event_exit_counts[envelope_name] = len(all_permanent_exits)
            permanent_exits = all_permanent_exits.loc[
                all_permanent_exits["event_id"].astype(str).isin(
                    table_plot_event_id_set
                )
            ].copy()
            plot_order = {
                event_id: rank for rank, event_id in enumerate(table_plot_event_ids)
            }
            permanent_exits["_plot_order"] = permanent_exits["event_id"].map(
                plot_order
            )
            permanent_exits = (
                permanent_exits.sort_values("_plot_order")
                .drop(columns="_plot_order")
                .reset_index(drop=True)
            )
            permanent_exits.to_csv(
                stage_dir / f"permanent_exits_{envelope_name}.csv", index=False
            )
            save_permanent_exit_table_png(
                permanent_exits,
                stage_dir / f"permanent_exits_{envelope_name}.png",
                args.max_outlier_table_events,
                table_event_colors,
            )
            exit_counts[envelope_name] = len(permanent_exits)

        at2017gfo_counts: dict[str, dict[str, int]] = {}
        if at2017gfo_input is not None:
            at2017gfo_stage = at2017gfo_stage_colours(
                at2017gfo_input,
                stage,
                band1,
                band2,
                args.at2017gfo_ebv,
                args.at2017gfo_photometry_mode,
                args.photometric_system,
            )
            for envelope_name, envelope_rules in (
                ("all_valid", rules_all),
                ("consecutive", rules_consecutive),
            ):
                at2017gfo_counts[envelope_name] = save_at2017gfo_outputs(
                    stage_dir,
                    at2017gfo_stage,
                    envelope_rules,
                    envelope_name,
                    pair,
                    args.t_min,
                    args.t_max,
                    args.at2017gfo_error_sigma,
                )

        save_stage_plots(
            stage_dir,
            pair,
            args.t_min,
            args.t_max,
            population_times[stage],
            population_colors[stage],
            rules_all,
            rules_consecutive,
            coloured_samples,
            colour_by_event_index,
            event_ids,
        )
        snia_convention = next(
            (
                convention
                for convention, comparison_stage in SNIA_CONVENTIONS.items()
                if comparison_stage == stage
            ),
            None,
        )
        if snia_population and snia_convention is not None:
            snia_overlap_summaries.extend(save_snia_comparison_outputs(
                output,
                snia_convention,
                pair,
                edges,
                centres,
                tails,
                args,
                snia_population,
                matrix,
                population_times[stage],
                population_colors[stage],
                rules_all,
                rules_consecutive,
            ))
        n_population_points = sum(len(values) for values in population_colors[stage])
        last_edge = (
            float(rules_consecutive["t_max_days"].max())
            if not rules_consecutive.empty else np.nan
        )
        summary_row = {
            "stage": stage,
            "N_valid_bins_anywhere": len(rules_all),
            "N_initial_consecutive_valid_bins": len(rules_consecutive),
            "last_initial_consecutive_time_days": last_edge,
            "N_native_population_points": n_population_points,
            "N_permanent_exits_in_coloured_sample_all_valid": exit_counts["all_valid"],
            "N_permanent_exits_in_coloured_sample_consecutive": exit_counts["consecutive"],
            "N_permanent_exits_all_events_all_valid": all_event_exit_counts["all_valid"],
            "N_permanent_exits_all_events_consecutive": all_event_exit_counts["consecutive"],
        }
        for envelope_name, counts in at2017gfo_counts.items():
            for key, value in counts.items():
                summary_row[f"AT2017gfo_{envelope_name}_{key}"] = value
        summaries.append(summary_row)
        print(
            f"{stage}: {len(rules_consecutive)} consecutive valid bins; "
            f"last edge={last_edge}; permanent exits in the "
            f"{table_plot_size}-event coloured sample="
            f"{exit_counts['all_valid']} (all valid), "
            f"{exit_counts['consecutive']} (consecutive)"
        )

    pd.DataFrame(summaries).to_csv(output / "stage_summary.csv", index=False)
    if snia_overlap_summaries:
        pd.DataFrame(snia_overlap_summaries).to_csv(
            output / "snia_comparison" / "KNe_SNIa_overlap_summary.csv",
            index=False,
        )
    configuration = {
        "run_dir": str(run_root(args.run_dir)),
        "model": "not re-evaluated; saved generator products used",
        "raw_model_predictions_recomputed": False,
        "synthetic_input_source": synthetic_source.kind,
        "photometric_system": args.photometric_system,
        "m5_selection_system": selection_system,
        "m5_selection_note": (
            "ZTF empirical depth is applied directly to saved ZTF magnitudes."
            if args.photometric_system == "ztf"
            else (
                "Rubin OpSim m5 is applied to the corresponding saved LSST "
                "magnitudes. For a PS1 envelope, it is not applied directly "
                "to the PS1 magnitude."
            )
        ),
        "numerical_support": "inherited exactly from generate_kne_lightcurves.py",
        "color_pair": pair,
        "population_latitude_selection_mode": (
            args.population_latitude_selection
        ),
        "population_latitude_selection": population_latitude_label(args),
        "population_latitude_threshold_deg": (
            args.latitude_threshold_deg
            if args.population_latitude_selection != "all"
            else None
        ),
        "population_latitude_selection_is_independent_of_m5_scope": True,
        "t_min_days": args.t_min,
        "t_max_days": args.t_max,
        "fixed_bin_width_days": args.bin_width,
        "bins_are_merged": False,
        "stages": list(STAGES),
        "stage_output_directories": {
            "intrinsic_no_m5_no_mw": "intrinsic_no_m5_no_mw",
            "no_m5_with_mw": "no_m5_with_mw",
            "m5_no_mw": f"m5_no_mw_{args.m5_scenario}",
            "m5_with_mw": f"m5_with_mw_{args.m5_scenario}",
        },
        "m5_quantile_table": str(m5_table),
        "m5_threshold_scope": (
            None if args.photometric_system == "ztf" else args.threshold_scope
        ),
        "m5_scenario": args.m5_scenario,
        "m5_by_band": m5,
        "ztf_depth_selection": (
            {
                "source_mode": (
                    args.ztf_depth_source
                    or (
                        "manual"
                        if any(
                            value is not None
                            for value in (
                                args.ztf_m5_g,
                                args.ztf_m5_r,
                                args.ztf_m5_i,
                            )
                        )
                        else "table"
                        if args.ztf_depth_quantile_table is not None
                        else "joblib"
                    )
                ),
                "manual_fixed_m5": {
                    "g": args.ztf_m5_g,
                    "r": args.ztf_m5_r,
                    "i": args.ztf_m5_i,
                },
                "depth_directory": (
                    str(args.ztf_depth_dir.expanduser().resolve())
                    if args.ztf_depth_dir is not None else None
                ),
                "input_quantile_table": (
                    str(args.ztf_depth_quantile_table.expanduser().resolve())
                    if args.ztf_depth_quantile_table is not None else None
                ),
                "recorded_threshold_table": str(
                    output / "ztf_depth_thresholds_used.csv"
                ),
                "draws_per_band": args.ztf_depth_samples,
                "seed": args.ztf_depth_seed,
                "interpretation": (
                    "one global limiting magnitude per band; this is not a "
                    "realization of ZTF cadence or per-visit noise"
                ),
            }
            if args.photometric_system == "ztf" else None
        ),
        "tail_percentile_grid": sorted(set(tails)),
        "median_is_stored_and_plotted": True,
        "population_plot_is_binned": False,
        "population_plot_includes_m5_rejected_points": True,
        "population_uses_saved_synthetic_times": True,
        "population_rendering": "one continuous polyline per event",
        "coloured_population_plot_sizes": sorted(coloured_samples),
        "population_plot_seed": args.population_plot_seed,
        "coloured_samples_are_nested": True,
        "event_colours_are_fixed_between_stages": True,
        "coloured_population_legend": (
            "shown only when the coloured sample contains at most 10 events; "
            "each trajectory is labelled as event <event_id>"
        ),
        "permanent_exit_definition": (
            "at least one earlier comparable inside-envelope bin followed by "
            "a final run of consecutive comparable outside-envelope bins with "
            "no later comparable re-entry"
        ),
        "minimum_final_outside_bins": args.minimum_final_outside_bins,
        "maximum_events_in_permanent_exit_png": args.max_outlier_table_events,
        "permanent_exit_table_coloured_sample_size": table_plot_size,
        "permanent_exit_table_event_ids": table_plot_event_ids,
        "permanent_exit_tables_contain": (
            "only permanent exits among the events displayed in the coloured "
            f"{table_plot_size}-event population plots"
        ),
        "permanent_exit_bin_columns": {
            "first_permanent_outside_bin_index": (
                "first bin of the final consecutive outside-envelope run"
            ),
            "last_outside_bin_index": (
                "last comparable bin of that final outside-envelope run"
            ),
            "last_outside_time_days": (
                "centre time of last_outside_bin_index"
            ),
        },
        "right_censored_definition": (
            "the event has no comparable colour in the final valid envelope bin"
        ),
        "bootstrap_repetitions": args.n_boot,
        "bootstrap_half_width_threshold_mag": args.bootstrap_threshold_mag,
        "minimum_events": args.minimum_events,
        "minimum_expected_tail_events": args.minimum_tail_events,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "split_seed": args.seed,
        "n_input_events": n_events,
        "maximum_events_requested": args.max_events,
        "event_subsample_seed": args.event_subsample_seed,
        "SN_Ia": {
            "enabled": bool(snia_population),
            "input_file": (
                str(args.snia_parquet.expanduser().resolve())
                if args.snia_parquet is not None else None
            ),
            "photometric_system": "lsst" if snia_population else None,
            "magnitude_column": args.snia_magnitude_column,
            "input_mw_mode": args.snia_input_mw_mode,
            "time_reference": (
                "first available MJD per SN Ia in either requested colour band"
                if args.snia_time_reference == "first_observation"
                else "SALT2 t0 (time of maximum light)"
            ),
            "KNe_time_reference": "merger",
            "position_projection": (
                "one deterministic KNe position per SN Ia; without replacement "
                "when enough KNe positions are available"
            ),
            "position_assignment_seed": args.snia_position_seed,
            "n_objects_with_requested_band_rows": (
                int(snia_population["n_objects_with_requested_band_rows"])
                if snia_population else 0
            ),
            "n_objects_with_both_requested_bands": (
                int(snia_population["n_objects_with_both_requested_bands"])
                if snia_population else 0
            ),
            "adapted_parquet": (
                str(snia_population.get("adapted_path"))
                if snia_population.get("adapted_path") is not None else None
            ),
            "comparison_conventions": SNIA_CONVENTIONS,
            "output_directories": {
                "mw_corrected": "mw_corrected",
                "mw_present": "mw_present",
                "m5_no_mw": f"m5_no_mw_{args.m5_scenario}",
                "m5_with_mw": f"m5_with_mw_{args.m5_scenario}",
            },
            "m5_applied_to_snia": {
                "mw_corrected": False,
                "mw_present": False,
                "m5_no_mw": True,
                "m5_with_mw": True,
            },
            "m5_threshold_scope": args.threshold_scope,
            "m5_scenario": args.m5_scenario,
            "m5_by_band": (
                snia_population.get("m5_thresholds")
                if snia_population else None
            ),
            "m5_rule": (
                "both interpolated SN Ia bands must be <= their selected "
                "Rubin m5 thresholds at the bin centre"
            ),
            "m5_population_plot_note": (
                "As for the KNe stages, the unbinned background contains the "
                "complete saved population; the m5 cut is applied to the "
                "matrix used to estimate the envelope."
            ),
            "source_dataset_selection_caveat": (
                "The supplied mock is already row-filtered by its generator; "
                "the README reports 2,327,115 rows after magnitude filtering."
            ),
            "overlap_metrics": {
                "overlap_color_min_max": "intersection bounds of both envelopes",
                "overlap_fraction_union": "intersection width divided by union width",
                "overlap_fraction_kne_width": "intersection width divided by KNe-envelope width",
                "overlap_fraction_snia_width": "intersection width divided by SN-Ia-envelope width",
                "fraction_snia_inside_kne": "fraction of available SN Ia colours inside the KNe envelope",
                "fraction_kne_inside_snia": "fraction of available KNe colours inside the SN Ia envelope",
            },
        },
        "AT2017gfo": {
            "enabled": args.at2017gfo_file is not None,
            "input_file": (
                str(args.at2017gfo_file.expanduser().resolve())
                if args.at2017gfo_file is not None else None
            ),
            "merger_mjd": args.at2017gfo_merger_mjd,
            "maximum_pair_separation_days": args.at2017gfo_match_window_days,
            "input_photometry_mode": args.at2017gfo_photometry_mode,
            "ebv_mw": args.at2017gfo_ebv,
            "photometric_error_compatibility_sigma": args.at2017gfo_error_sigma,
            "filter_mapping": {
                band1: list(AT2017GFO_FILTER_ALIASES[band1]),
                band2: list(AT2017GFO_FILTER_ALIASES[band2]),
            },
            "filter_system_caveat": (
                "AT2017gfo grizy measurements are PS1 photometry. They are "
                + (
                    "compared directly to the PS1 synthetic envelope."
                    if args.photometric_system == "ps1"
                    else "used only as proxies for Rubin/LSST bands in this branch."
                )
            ),
            "filter_verdict_uses": (
                "the observed colour interval colour +/- N_sigma * "
                "colour_error; a point passes when that interval intersects "
                "the envelope"
            ),
        },
    }
    (output / "four_stage_envelope_configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    print("Output:", output)
    return output


if __name__ == "__main__":
    main()
