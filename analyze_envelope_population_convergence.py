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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kne_pipeline_common import prompt_choice, prompt_text


STAGES = (
    "intrinsic_no_m5_no_mw",
    "no_m5_with_mw",
    "m5_no_mw",
    "m5_with_mw",
)
M5_STAGES = frozenset({"m5_no_mw", "m5_with_mw"})
M5_SCENARIOS = ("worst_p25", "median_p50", "best_p75")


def parse_sample_sizes(text: str) -> list[int]:
    values: list[int] = []
    for piece in text.split(","):
        piece = piece.strip().replace("_", "")
        if not piece:
            continue
        value = int(float(piece))
        if value <= 0:
            raise ValueError("Every population size must be positive")
        values.append(value)
    values = sorted(set(values))
    if len(values) < 2:
        raise ValueError("At least two distinct population sizes are required")
    return values


def split_colour_pair(pair: str) -> tuple[str, str]:
    pieces = [piece.strip().lower() for piece in pair.split("-")]
    if len(pieces) != 2 or not all(pieces) or pieces[0] == pieces[1]:
        raise ValueError("Colour pair must look like g-r, g-i or r-i")
    return pieces[0], pieces[1]


def resolve_run_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "summary.csv").is_file():
        return path
    if path.name in {"parquet", "csv"} and (path.parent / "summary.csv").is_file():
        return path.parent
    raise FileNotFoundError(
        f"Could not find summary.csv below {path}. Supply the run root, not an "
        "individual Parquet shard."
    )


def count_summary_events(run_root: Path) -> int:
    with (run_root / "summary.csv").open("r", encoding="utf-8", errors="replace") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def stage_directory_name(stage: str, scenario: str) -> str:
    return f"{stage}_{scenario}" if stage in M5_STAGES else stage


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


def aggregate_results(
    raw: pd.DataFrame,
    tolerance: float,
    required_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float, int | None]:
    raw = raw.copy()
    maximum_n = int(raw["N_requested"].max())
    references = (
        raw.loc[raw["N_requested"] == maximum_n, ["repetition", "last_valid_time_days"]]
        .rename(columns={"last_valid_time_days": "reference_time_days"})
    )
    raw = raw.merge(references, on="repetition", how="left", validate="many_to_one")
    raw["absolute_difference_from_reference_days"] = (
        raw["last_valid_time_days"] - raw["reference_time_days"]
    ).abs()
    raw["within_reference_tolerance"] = (
        raw["absolute_difference_from_reference_days"] <= tolerance + 1e-12
    )
    def finite_percentile(values: pd.Series, percentile: float) -> float:
        finite = pd.to_numeric(values, errors="coerce").to_numpy(float)
        finite = finite[np.isfinite(finite)]
        return float(np.percentile(finite, percentile)) if finite.size else np.nan

    grouped = raw.groupby("N_requested", sort=True)
    summary = grouped.agg(
        N_effective_median=("N_input_events", "median"),
        repetitions=("repetition", "count"),
        last_time_median=("last_valid_time_days", "median"),
        last_time_p16=("last_valid_time_days", lambda x: finite_percentile(x, 16)),
        last_time_p84=("last_valid_time_days", lambda x: finite_percentile(x, 84)),
        fraction_within_reference=("within_reference_tolerance", "mean"),
        median_absolute_difference_days=(
            "absolute_difference_from_reference_days", "median"
        ),
        last_low_percentile_median=("last_low_percentile", "median"),
        N_train_last_bin_median=("N_train_last_bin", "median"),
    ).reset_index()
    reference_median = float(
        raw.loc[raw["N_requested"] == maximum_n, "last_valid_time_days"].median()
    )
    summary["median_within_reference_tolerance"] = (
        (summary["last_time_median"] - reference_median).abs()
        <= tolerance + 1e-12
    )
    summary["point_passes_stability"] = (
        summary["median_within_reference_tolerance"]
        & (summary["fraction_within_reference"] >= required_fraction)
    )
    passes = summary["point_passes_stability"].to_numpy(bool)
    stable_from_here = np.logical_and.accumulate(passes[::-1])[::-1]
    summary["all_larger_sizes_also_stable"] = stable_from_here
    candidates = summary.loc[summary["all_larger_sizes_also_stable"]]
    threshold = int(candidates.iloc[0]["N_requested"]) if not candidates.empty else None
    return raw, summary, reference_median, threshold


def exponential_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(x) < 4 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    def model(n: np.ndarray, asymptote: float, amplitude: float, scale: float) -> np.ndarray:
        return asymptote - amplitude * np.exp(-n / scale)

    spread = max(float(np.nanmax(y) - np.nanmin(y)), 0.1)
    initial = [float(np.nanmax(y)), spread, float(np.median(x))]
    lower = [float(np.nanmin(y) - spread), 0.0, max(float(np.nanmin(x)) / 100.0, 1e-6)]
    upper = [float(np.nanmax(y) + 2.0 * spread), 10.0 * spread, float(np.nanmax(x)) * 100.0]
    try:
        parameters, _ = curve_fit(
            model,
            x.astype(float),
            y.astype(float),
            p0=initial,
            bounds=(lower, upper),
            maxfev=50000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    fit_x = np.geomspace(float(np.min(x)), float(np.max(x)), 500)
    return fit_x, model(fit_x, *parameters)


def save_plot(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    reference: float,
    threshold: int | None,
    args: argparse.Namespace,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
    for _, group in raw.groupby("repetition"):
        group = group.sort_values("N_requested")
        ax.plot(
            group["N_requested"],
            group["last_valid_time_days"],
            color="0.72",
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )
    x = summary["N_requested"].to_numpy(float)
    median = summary["last_time_median"].to_numpy(float)
    p16 = summary["last_time_p16"].to_numpy(float)
    p84 = summary["last_time_p84"].to_numpy(float)
    ax.fill_between(x, p16, p84, color="#3b78b5", alpha=0.22, label="16--84% subsamples")
    ax.plot(x, median, "o-", color="#155a96", linewidth=2.0, markersize=5.5, label="Median last valid edge")
    if args.fit_exponential:
        fitted = exponential_fit(x, median)
        if fitted is not None:
            ax.plot(
                fitted[0], fitted[1], linestyle="--", color="#d17a22",
                linewidth=1.8, label="Exponential saturation guide",
            )
    ax.axhspan(
        reference - args.tolerance_days,
        reference + args.tolerance_days,
        color="#4f9d69",
        alpha=0.12,
        label=f"Largest-N reference +/- {args.tolerance_days:g} d",
    )
    ax.axhline(reference, color="#327a4b", linewidth=1.2)
    if threshold is not None:
        ax.axvline(threshold, color="#9b2f2f", linestyle=":", linewidth=1.8)
        ax.annotate(
            f"N* = {threshold:,}",
            xy=(threshold, reference),
            xytext=(8, 12),
            textcoords="offset points",
            color="#8a2525",
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Number of injected KNe")
    ax.set_ylabel("Last consecutive valid bin edge [days]")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", axis="x", alpha=0.10)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output, dpi=300)
    plt.close(fig)


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
