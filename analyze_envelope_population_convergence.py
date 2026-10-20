#!/usr/bin/env python3
"""Measure convergence of the last valid colour-envelope bin with population size.

The script reuses one large saved KNe population.  For each repetition it
creates one random event permutation; increasing sample sizes are prefixes of
that same permutation and are therefore nested.  Missing envelope builds are
launched through ``build_color_envelopes.py`` and existing results can be
resumed without recomputation.

The convergence threshold is the first tested population size N such that N
and every larger tested size remain within a configurable time tolerance of
the largest-sample reference.  The median and 16--84 percentile interval over
subsampling repetitions are plotted.  An exponential saturation fit is a
visual guide only and is not used to define the threshold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kne_pipeline_common import prompt_choice, prompt_text
from kne_convergence_common import (
    M5_SCENARIOS,
    M5_STAGES,
    STAGES,
    aggregate_results,
    count_summary_events,
    parse_sample_sizes,
    resolve_run_root,
    save_plot,
    split_colour_pair,
    stage_directory_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run nested KNe population-size experiments and measure the last "
            "initial consecutive valid colour-envelope bin."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path(__file__).with_name("build_color_envelopes.py"),
    )
    parser.add_argument("--color-pair", default="g-r")
    parser.add_argument(
        "--photometric-system",
        choices=["lsst", "ps1", "ztf"],
        default="lsst",
        help=(
            "Analyse one envelope system at a time. A paired LSST+ZTF run can "
            "be analysed twice, once with lsst and once with ztf."
        ),
    )
    parser.add_argument("--stage", choices=STAGES, default="m5_with_mw")
    parser.add_argument(
        "--sample-sizes",
        default="100,1000,10000,100000,300000,600000,1000000",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--subsample-seed", type=int, default=20261002)
    parser.add_argument("--builder-seed", type=int, default=12345)
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--t-max", type=float, default=16.0)
    parser.add_argument("--bin-width", type=float, default=0.4)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--minimum-events", type=int, default=30)
    parser.add_argument("--minimum-tail-events", type=float, default=20.0)
    parser.add_argument("--bootstrap-threshold-mag", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--tail-percentiles",
        help="Optional override of the builder's lower-tail percentile grid.",
    )
    parser.add_argument(
        "--m5-scenario", choices=M5_SCENARIOS, default="median_p50"
    )
    parser.add_argument(
        "--threshold-scope",
        choices=["common_global", "latitude_conditioned"],
        default="common_global",
    )
    parser.add_argument("--m5-quantile-table", type=Path)
    parser.add_argument(
        "--ztf-depth-source", choices=["manual", "table"], default="manual"
    )
    parser.add_argument("--ztf-m5-g", type=float)
    parser.add_argument("--ztf-m5-r", type=float)
    parser.add_argument("--ztf-m5-i", type=float)
    parser.add_argument("--ztf-depth-quantile-table", type=Path)
    parser.add_argument(
        "--tolerance-days",
        type=float,
        default=0.0,
        help=(
            "Maximum difference from the largest-N reference. Zero requires "
            "the same final temporal bin; use one bin width for a relaxed rule."
        ),
    )
    parser.add_argument(
        "--required-fraction",
        type=float,
        default=0.90,
        help="Required fraction of repetitions within the reference tolerance.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed population-size builds.",
    )
    parser.add_argument(
        "--fit-exponential",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a phenomenological RC-like saturation fit.",
    )
    parser.add_argument("--interactive", action="store_true")
    return parser


def interactive(args: argparse.Namespace) -> argparse.Namespace:
    args.run_dir = Path(prompt_text("Parent run directory", str(args.run_dir)))
    args.color_pair = prompt_text("Colour pair", args.color_pair)
    args.photometric_system = prompt_choice(
        "Photometric system whose envelope convergence is tested",
        [
            ("lsst", "LSST"),
            ("ps1", "PS1 colour with Rubin/LSST depth selection"),
            ("ztf", "ZTF with supplied ZTF depth thresholds"),
        ],
        args.photometric_system,
    )
    args.stage = prompt_choice(
        "Envelope stage",
        [
            ("intrinsic_no_m5_no_mw", "No m5 and no Milky-Way extinction"),
            ("no_m5_with_mw", "No m5, with Milky-Way extinction"),
            ("m5_no_mw", "With m5, without Milky-Way extinction"),
            ("m5_with_mw", "With m5 and Milky-Way extinction"),
        ],
        args.stage,
    )
    args.sample_sizes = prompt_text(
        "Population-size grid", args.sample_sizes
    )
    args.repetitions = int(prompt_text(
        "Random nested-subsample repetitions", str(args.repetitions)
    ))
    args.t_min = float(prompt_text("First temporal-bin lower edge [days]", str(args.t_min)))
    args.t_max = float(prompt_text("Last temporal-bin upper edge [days]", str(args.t_max)))
    args.bin_width = float(prompt_text("Fixed bin width [days]", str(args.bin_width)))
    args.n_boot = int(prompt_text("Bootstrap repetitions per envelope", str(args.n_boot)))
    args.minimum_tail_events = float(prompt_text(
        "Minimum expected events in each percentile tail",
        str(args.minimum_tail_events),
    ))
    args.bootstrap_threshold_mag = float(prompt_text(
        "Maximum 68% bootstrap half-width per envelope bound [mag]",
        str(args.bootstrap_threshold_mag),
    ))
    args.m5_scenario = prompt_choice(
        "m5 threshold scenario",
        [
            ("worst_p25", "worst p25"),
            ("median_p50", "median p50"),
            ("best_p75", "best p75"),
        ],
        args.m5_scenario,
    )
    if args.photometric_system == "ztf":
        args.ztf_depth_source = prompt_choice(
            "ZTF limiting-depth input",
            [
                ("manual", "Enter fixed 5-sigma limiting magnitudes"),
                ("table", "Read p25/p50/p75 values from a CSV table"),
            ],
            args.ztf_depth_source,
        )
        band1, band2 = split_colour_pair(args.color_pair)
        if args.ztf_depth_source == "manual":
            for band in dict.fromkeys((band1, band2)):
                attribute = f"ztf_m5_{band}"
                current = getattr(args, attribute)
                value = prompt_text(
                    f"Fixed ZTF {band}-band 5-sigma limiting magnitude [AB mag]",
                    None if current is None else str(current),
                )
                setattr(args, attribute, float(value))
        else:
            args.ztf_depth_quantile_table = Path(prompt_text(
                "ZTF depth-quantile CSV table",
                str(args.ztf_depth_quantile_table or "ztf_depth_quantiles.csv"),
            ))
    args.tolerance_days = float(prompt_text(
        "Convergence tolerance on the last valid edge [days] (0 = same bin)",
        str(args.tolerance_days),
    ))
    args.required_fraction = float(prompt_text(
        "Required fraction of repetitions within tolerance",
        str(args.required_fraction),
    ))
    default_output = (
        resolve_run_root(args.run_dir)
        / "analysis"
        / f"envelope_population_convergence_{args.color_pair.replace('-', '_')}_{args.photometric_system}_{args.stage}"
    )
    args.output_dir = Path(prompt_text(
        "Output directory", str(args.output_dir or default_output)
    ))
    return args


def validate(args: argparse.Namespace) -> tuple[Path, list[int]]:
    run_root = resolve_run_root(args.run_dir)
    split_colour_pair(args.color_pair)
    sizes = parse_sample_sizes(args.sample_sizes)
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least one")
    if args.n_boot < 1:
        raise ValueError("--n-boot must be at least one")
    if args.bin_width <= 0 or args.t_max <= args.t_min:
        raise ValueError("Invalid temporal binning")
    if not np.isclose(
        (args.t_max - args.t_min) / args.bin_width,
        round((args.t_max - args.t_min) / args.bin_width),
        atol=1e-8,
    ):
        raise ValueError("(t-max - t-min) must be an integer multiple of bin-width")
    if args.tolerance_days < 0:
        raise ValueError("--tolerance-days must be non-negative")
    if not 0 < args.required_fraction <= 1:
        raise ValueError("--required-fraction must lie in (0, 1]")
    if not args.builder.expanduser().is_file():
        raise FileNotFoundError(f"Envelope builder not found: {args.builder}")
    available = count_summary_events(run_root)
    if sizes[-1] > available:
        raise ValueError(
            f"Largest requested population ({sizes[-1]:,}) exceeds the "
            f"{available:,} rows in summary.csv. Generate a larger parent run "
            "or reduce the grid."
        )
    if args.photometric_system == "ztf":
        band1, band2 = split_colour_pair(args.color_pair)
        if any(band not in {"g", "r", "i"} for band in (band1, band2)):
            raise ValueError("ZTF convergence supports g, r and i only")
        if args.ztf_depth_source == "manual":
            for band in (band1, band2):
                if getattr(args, f"ztf_m5_{band}") is None:
                    raise ValueError(f"Missing --ztf-m5-{band}")
        elif args.ztf_depth_quantile_table is None:
            raise ValueError("ZTF table mode requires --ztf-depth-quantile-table")
    return run_root, sizes


def builder_command(
    args: argparse.Namespace,
    run_root: Path,
    sample_size: int,
    repetition: int,
    build_output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.builder.expanduser().resolve()),
        "--run-dir", str(run_root),
        "--color-pair", args.color_pair,
        "--photometric-system", args.photometric_system,
        "--t-min", str(args.t_min),
        "--t-max", str(args.t_max),
        "--bin-width", str(args.bin_width),
        "--n-boot", str(args.n_boot),
        "--bootstrap-threshold-mag", str(args.bootstrap_threshold_mag),
        "--minimum-events", str(args.minimum_events),
        "--minimum-tail-events", str(args.minimum_tail_events),
        "--train-fraction", str(args.train_fraction),
        "--validation-fraction", str(args.validation_fraction),
        "--seed", str(args.builder_seed),
        "--max-events", str(sample_size),
        "--event-subsample-seed", str(args.subsample_seed + repetition),
        "--colored-plot-sizes", "1",
        "--outlier-table-plot-size", "1",
        "--max-outlier-table-events", "1",
        "--m5-scenario", args.m5_scenario,
        "--threshold-scope", args.threshold_scope,
        "--output-dir", str(build_output),
    ]
    if args.tail_percentiles:
        command.extend(["--tail-percentiles", args.tail_percentiles])
    if args.m5_quantile_table is not None:
        command.extend(["--m5-quantile-table", str(args.m5_quantile_table)])
    if args.photometric_system == "ztf":
        command.extend(["--ztf-depth-source", args.ztf_depth_source])
        if args.ztf_depth_source == "manual":
            for band in set(split_colour_pair(args.color_pair)):
                command.extend([
                    f"--ztf-m5-{band}",
                    str(getattr(args, f"ztf_m5_{band}")),
                ])
        else:
            command.extend([
                "--ztf-depth-quantile-table",
                str(args.ztf_depth_quantile_table),
            ])
    return command


def read_build_result(
    args: argparse.Namespace,
    build_output: Path,
    requested_size: int,
    repetition: int,
) -> dict[str, object]:
    summary_path = build_output / "stage_summary.csv"
    config_path = build_output / "four_stage_envelope_configuration.json"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete envelope output below {build_output}")
    summary = pd.read_csv(summary_path)
    selected = summary.loc[summary["stage"].astype(str) == args.stage]
    if len(selected) != 1:
        raise ValueError(f"Expected one {args.stage} row in {summary_path}")
    row = selected.iloc[0]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rules_path = (
        build_output
        / stage_directory_name(args.stage, args.m5_scenario)
        / "dynamic_color_cut_rules.csv"
    )
    low_percentile = np.nan
    high_percentile = np.nan
    n_train_last = np.nan
    if rules_path.is_file():
        rules = pd.read_csv(rules_path)
        if not rules.empty:
            last = rules.sort_values("t_max_days").iloc[-1]
            low_percentile = float(last.get("low_percentile", np.nan))
            high_percentile = float(last.get("high_percentile", np.nan))
            n_train_last = float(last.get("N_train_events", np.nan))
    return {
        "repetition": repetition,
        "subsample_seed": args.subsample_seed + repetition,
        "N_requested": requested_size,
        "N_input_events": int(config["n_input_events"]),
        "N_train_last_bin": n_train_last,
        "stage": args.stage,
        "last_valid_time_days": float(
            row["last_initial_consecutive_time_days"]
        ),
        "N_consecutive_valid_bins": int(row["N_initial_consecutive_valid_bins"]),
        "last_low_percentile": low_percentile,
        "last_high_percentile": high_percentile,
        "build_output": str(build_output),
    }








def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.interactive or len(sys.argv) == 1:
        args = interactive(args)
    run_root, sizes = validate(args)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root
        / "analysis"
        / f"envelope_population_convergence_{args.color_pair.replace('-', '_')}_{args.photometric_system}_{args.stage}"
    )
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    total = args.repetitions * len(sizes)
    counter = 0
    for repetition in range(args.repetitions):
        for sample_size in sizes:
            counter += 1
            build_output = output / "builds" / f"rep_{repetition:03d}" / f"N_{sample_size:09d}"
            summary_path = build_output / "stage_summary.csv"
            config_path = build_output / "four_stage_envelope_configuration.json"
            if not (args.resume and summary_path.is_file() and config_path.is_file()):
                build_output.mkdir(parents=True, exist_ok=True)
                command = builder_command(
                    args, run_root, sample_size, repetition, build_output
                )
                print(
                    f"[{counter}/{total}] repetition={repetition + 1}/{args.repetitions}, "
                    f"N={sample_size:,}"
                )
                subprocess.run(command, check=True)
            else:
                print(
                    f"[{counter}/{total}] reusing repetition={repetition + 1}, "
                    f"N={sample_size:,}"
                )
            records.append(read_build_result(
                args, build_output, sample_size, repetition
            ))

    raw = pd.DataFrame(records)
    raw, summary, reference, threshold = aggregate_results(
        raw, args.tolerance_days, args.required_fraction
    )
    raw_path = output / "envelope_population_convergence_raw.csv"
    summary_path = output / "envelope_population_convergence_summary.csv"
    plot_path = output / "envelope_population_convergence.png"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_plot(raw, summary, reference, threshold, args, plot_path)
    configuration = {
        "run_dir": str(run_root),
        "photometric_system": args.photometric_system,
        "color_pair": args.color_pair,
        "stage": args.stage,
        "sample_sizes": sizes,
        "repetitions": args.repetitions,
        "nested_subsets_within_each_repetition": True,
        "subsample_seed_base": args.subsample_seed,
        "last_bin_definition": "last initial consecutive valid envelope-bin upper edge",
        "reference": "last-bin time at the largest tested population, paired by repetition",
        "tolerance_days": args.tolerance_days,
        "required_fraction": args.required_fraction,
        "stability_rule": (
            "first tested N for which the median and the required fraction of "
            "repetitions are within tolerance, and all larger tested N also pass"
        ),
        "stable_population_threshold": threshold,
        "largest_N_reference_median_days": reference,
        "exponential_fit_is_visual_only": bool(args.fit_exponential),
    }
    (output / "envelope_population_convergence_configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Reference last edge: {reference:g} days")
    print(
        "Stable population threshold: "
        + (f"{threshold:,} KNe" if threshold is not None else "not determined")
    )
    print("Plot:", plot_path)
    print("Summary:", summary_path)
    print("Raw repetitions:", raw_path)


if __name__ == "__main__":
    main()
