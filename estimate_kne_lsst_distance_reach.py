#!/usr/bin/env python3
"""Estimate LSST colour-measurement completeness as a function of distance.

The input must be a variable-distance synthetic KNe population (normally a
run drawn uniformly in comoving volume) saved without a generator-level m5
cut.  For every event, the script asks whether at least one temporal-bin centre
contains finite magnitudes in both requested bands and whether both are
brighter than the selected Rubin/LSST m5 thresholds.

Distance-binned binomial fractions are reported with Wilson intervals.  A
weighted monotonically decreasing isotonic curve is used only to interpolate
D50, D30, D20, D10 and D1; the raw binned measurements remain in the output.
This measures synthetic colour availability under global m5 thresholds, not
the probability that the actual Rubin cadence observes both bands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_color_envelopes import (
    SyntheticPopulationSource,
    discover_m5_table,
    event_extinction_mag,
    load_m5,
)
from dynamic_color_cut_common import (
    canonical_event_id,
    interpolate_band,
    load_summary,
    run_root,
)
from kne_pipeline_common import prompt_choice, prompt_text


def parse_colour_pair(value: str) -> tuple[str, str]:
    pieces = tuple(piece.strip().lower() for piece in value.split("-"))
    if len(pieces) != 2 or pieces[0] == pieces[1] or any(
        band not in "ugrizy" for band in pieces
    ):
        raise ValueError("Colour pair must look like g-r, r-i or i-z")
    return pieces


def parse_target_percentages(value: str) -> list[float]:
    targets = sorted(
        {
            float(piece.strip())
            for piece in value.split(",")
            if piece.strip()
        },
        reverse=True,
    )
    if not targets or any(target <= 0 or target >= 100 for target in targets):
        raise ValueError("Completeness targets must lie strictly between 0 and 100")
    return targets


def wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float,
) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    fraction = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (fraction + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * np.sqrt(
            fraction * (1.0 - fraction) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def decreasing_isotonic(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted pool-adjacent-violators fit constrained to be decreasing."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("values and weights must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(values)) or not np.all(weights > 0):
        raise ValueError("isotonic inputs must be finite with positive weights")
    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append({
            "start": index,
            "end": index,
            "weight": float(weight),
            "weighted_sum": float(value * weight),
        })
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = float(previous["weighted_sum"]) / float(previous["weight"])
            current_mean = float(current["weighted_sum"]) / float(current["weight"])
            if previous_mean >= current_mean:
                break
            merged = {
                "start": int(previous["start"]),
                "end": int(current["end"]),
                "weight": float(previous["weight"]) + float(current["weight"]),
                "weighted_sum": (
                    float(previous["weighted_sum"])
                    + float(current["weighted_sum"])
                ),
            }
            blocks[-2:] = [merged]
    fitted = np.empty_like(values)
    for block in blocks:
        mean = float(block["weighted_sum"]) / float(block["weight"])
        fitted[int(block["start"]): int(block["end"]) + 1] = mean
    return fitted


def crossing_distance(
    distances: np.ndarray,
    completeness: np.ndarray,
    target: float,
) -> tuple[float, str]:
    distances = np.asarray(distances, dtype=float)
    completeness = np.asarray(completeness, dtype=float)
    finite = np.isfinite(distances) & np.isfinite(completeness)
    distances = distances[finite]
    completeness = completeness[finite]
    order = np.argsort(distances)
    distances = distances[order]
    completeness = completeness[order]
    if len(distances) < 2:
        return np.nan, "insufficient_distance_bins"
    if completeness[0] < target:
        return np.nan, "below_target_at_minimum_distance"
    if completeness[-1] > target:
        return np.nan, "above_target_at_maximum_distance"
    indices = np.flatnonzero(completeness <= target)
    if len(indices) == 0:
        return np.nan, "target_not_bracketed"
    upper = int(indices[0])
    if upper == 0 or completeness[upper] == target:
        return float(distances[upper]), "interpolated"
    lower = upper - 1
    y0, y1 = completeness[lower], completeness[upper]
    if np.isclose(y0, y1):
        return float(0.5 * (distances[lower] + distances[upper])), "plateau_at_target"
    fraction = (target - y0) / (y1 - y0)
    distance = distances[lower] + fraction * (
        distances[upper] - distances[lower]
    )
    return float(distance), "interpolated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate D50/D30/D20/D10/D1 for an LSST colour measurement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    parser.add_argument("--color-pair", default="g-r")
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--t-max", type=float, default=16.0)
    parser.add_argument("--bin-width", type=float, default=0.4)
    parser.add_argument(
        "--m5-scenario",
        choices=["worst_p25", "median_p50", "best_p75"],
        default="median_p50",
    )
    parser.add_argument(
        "--threshold-scope",
        choices=["common_global", "latitude_conditioned"],
        default="common_global",
    )
    parser.add_argument("--m5-quantile-table", type=Path)
    parser.add_argument(
        "--mw-treatment",
        choices=["present", "corrected"],
        default="present",
        help="Use saved MW-extincted magnitudes or subtract A_band before m5 selection.",
    )
    parser.add_argument("--distance-bin-width-mpc", type=float, default=25.0)
    parser.add_argument("--distance-min-mpc", type=float)
    parser.add_argument("--distance-max-mpc", type=float)
    parser.add_argument("--minimum-events-per-bin", type=int, default=200)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--target-percentages",
        default="50,30,20,10,1",
        help="Completeness levels whose crossing distances are reported.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interactive", action="store_true")
    return parser


def interactive(args: argparse.Namespace) -> argparse.Namespace:
    args.run_dir = Path(prompt_text("Variable-distance parent run", str(args.run_dir)))
    args.color_pair = prompt_text("Colour pair", args.color_pair)
    args.t_min = float(prompt_text("First temporal-bin lower edge [days]", str(args.t_min)))
    args.t_max = float(prompt_text("Last temporal-bin upper edge [days]", str(args.t_max)))
    args.bin_width = float(prompt_text("Fixed temporal-bin width [days]", str(args.bin_width)))
    args.m5_scenario = prompt_choice(
        "Rubin/LSST m5 threshold",
        [
            ("worst_p25", "worst p25"),
            ("median_p50", "median p50"),
            ("best_p75", "best p75"),
        ],
        args.m5_scenario,
    )
    args.threshold_scope = prompt_choice(
        "m5 latitude scope",
        [
            ("common_global", "Same all-sky thresholds"),
            ("latitude_conditioned", "Thresholds measured at high latitude"),
        ],
        args.threshold_scope,
    )
    args.mw_treatment = prompt_choice(
        "Milky-Way extinction in the detectability calculation",
        [
            ("present", "Include MW extinction [recommended for observed reach]"),
            ("corrected", "Subtract MW extinction before applying m5"),
        ],
        args.mw_treatment,
    )
    args.distance_bin_width_mpc = float(prompt_text(
        "Luminosity-distance bin width [Mpc]", str(args.distance_bin_width_mpc)
    ))
    args.minimum_events_per_bin = int(prompt_text(
        "Minimum injected events per distance bin",
        str(args.minimum_events_per_bin),
    ))
    args.target_percentages = prompt_text(
        "Reported completeness levels [%]", args.target_percentages
    )
    default_output = (
        run_root(args.run_dir)
        / "analysis"
        / f"lsst_distance_reach_{args.color_pair.replace('-', '_')}_{args.m5_scenario}"
    )
    args.output_dir = Path(prompt_text(
        "Output directory", str(args.output_dir or default_output)
    ))
    return args


def validate(args: argparse.Namespace) -> tuple[Path, tuple[str, str], list[float]]:
    root = run_root(args.run_dir)
    if not (root / "summary.csv").is_file():
        raise FileNotFoundError(f"Missing {root / 'summary.csv'}")
    bands = parse_colour_pair(args.color_pair)
    targets = parse_target_percentages(args.target_percentages)
    if args.t_max <= args.t_min or args.bin_width <= 0:
        raise ValueError("Invalid temporal interval or bin width")
    if not np.isclose(
        (args.t_max - args.t_min) / args.bin_width,
        round((args.t_max - args.t_min) / args.bin_width),
        atol=1e-8,
    ):
        raise ValueError("(t-max - t-min) must be an integer multiple of bin-width")
    if args.distance_bin_width_mpc <= 0:
        raise ValueError("--distance-bin-width-mpc must be positive")
    if args.minimum_events_per_bin < 1:
        raise ValueError("--minimum-events-per-bin must be at least one")
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must lie in (0, 1)")
    return root, bands, targets


def event_detectability(
    root: Path,
    metadata: pd.DataFrame,
    bands: tuple[str, str],
    centres: np.ndarray,
    thresholds: dict[str, float],
    mw_treatment: str,
) -> pd.Series:
    source = SyntheticPopulationSource(root)
    if source.kind == "missing":
        raise FileNotFoundError(
            f"No parquet/synthetic_part_*.parquet or legacy synthetic CSV below {root}"
        )
    event_ids = metadata["event_id"].tolist()
    detected = pd.Series(False, index=event_ids, dtype=bool)
    metadata_by_event = metadata.set_index("event_id", drop=False)
    processed: set[str] = set()
    progress_interval = max(1000, len(event_ids) // 20)
    for event_id, lightcurve in source.iter_events(
        event_ids,
        bands,
        ("lsst",),
    ):
        event_id = canonical_event_id(event_id)
        if event_id in processed:
            raise RuntimeError(f"Duplicate synthetic event {event_id}")
        processed.add(event_id)
        curve = pd.DataFrame({
            "t_days": pd.to_numeric(lightcurve["t_days"], errors="coerce"),
            "band": lightcurve["band"].astype(str).str.strip().str.lower(),
            "magnitude": pd.to_numeric(lightcurve["mag"], errors="coerce"),
            "photometric_system": (
                lightcurve["photometric_system"].astype(str).str.strip().str.lower()
                if "photometric_system" in lightcurve
                else "lsst"
            ),
        })
        curve = curve.loc[
            np.isfinite(curve["t_days"])
            & np.isfinite(curve["magnitude"])
            & curve["band"].isin(bands)
            & curve["photometric_system"].eq("lsst")
        ].copy()
        row = metadata_by_event.loc[event_id]
        magnitude1 = interpolate_band(curve, bands[0], centres)
        magnitude2 = interpolate_band(curve, bands[1], centres)
        if mw_treatment == "corrected":
            magnitude1 = magnitude1 - event_extinction_mag(row, "lsst", bands[0])
            magnitude2 = magnitude2 - event_extinction_mag(row, "lsst", bands[1])
        available = (
            np.isfinite(magnitude1)
            & np.isfinite(magnitude2)
            & (magnitude1 <= thresholds[bands[0]])
            & (magnitude2 <= thresholds[bands[1]])
        )
        detected.loc[event_id] = bool(np.any(available))
        if len(processed) % progress_interval == 0:
            print(f"  read {len(processed):,}/{len(event_ids):,} events")
    print(
        f"Synthetic rows found for {len(processed):,}/{len(event_ids):,} events; "
        "events without usable rows remain colour-unavailable."
    )
    return detected


def bin_completeness(
    metadata: pd.DataFrame,
    detected: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    distances = metadata["luminosity_distance"].to_numpy(float)
    observed_min = float(np.nanmin(distances))
    observed_max = float(np.nanmax(distances))
    lower = (
        float(args.distance_min_mpc)
        if args.distance_min_mpc is not None
        else np.floor(observed_min / args.distance_bin_width_mpc)
        * args.distance_bin_width_mpc
    )
    upper = (
        float(args.distance_max_mpc)
        if args.distance_max_mpc is not None
        else np.ceil(observed_max / args.distance_bin_width_mpc)
        * args.distance_bin_width_mpc
    )
    if upper <= lower:
        raise ValueError("Distance range must have positive width")
    edges = np.arange(
        lower,
        upper + args.distance_bin_width_mpc * 0.5,
        args.distance_bin_width_mpc,
    )
    if edges[-1] < upper:
        edges = np.append(edges, upper)
    edges[-1] = np.nextafter(max(edges[-1], observed_max), np.inf)
    work = metadata[["event_id", "luminosity_distance"]].copy()
    work["colour_measurable"] = work["event_id"].map(detected).fillna(False).astype(bool)
    work["distance_bin"] = pd.cut(
        work["luminosity_distance"], edges, right=False, include_lowest=True
    )
    rows = []
    for interval, group in work.groupby("distance_bin", observed=False, sort=True):
        n_events = len(group)
        n_detected = int(group["colour_measurable"].sum())
        fraction = n_detected / n_events if n_events else np.nan
        low, high = wilson_interval(
            n_detected, n_events, args.confidence_level
        )
        rows.append({
            "distance_min_mpc": float(interval.left),
            "distance_max_mpc": float(interval.right),
            "distance_centre_mpc": 0.5 * (float(interval.left) + float(interval.right)),
            "distance_median_mpc": (
                float(group["luminosity_distance"].median()) if n_events else np.nan
            ),
            "N_injected": n_events,
            "N_colour_measurable": n_detected,
            "colour_completeness": fraction,
            "wilson_low": low,
            "wilson_high": high,
            "sufficient_bin_population": n_events >= args.minimum_events_per_bin,
        })
    result = pd.DataFrame(rows)
    valid = result["sufficient_bin_population"] & np.isfinite(
        result["colour_completeness"]
    )
    result["isotonic_completeness"] = np.nan
    if valid.sum() >= 2:
        result.loc[valid, "isotonic_completeness"] = decreasing_isotonic(
            result.loc[valid, "colour_completeness"].to_numpy(float),
            result.loc[valid, "N_injected"].to_numpy(float),
        )
    return result


def reach_table(
    binned: pd.DataFrame,
    targets: list[float],
) -> pd.DataFrame:
    valid = (
        binned["sufficient_bin_population"]
        & np.isfinite(binned["isotonic_completeness"])
    )
    distances = binned.loc[valid, "distance_median_mpc"].to_numpy(float)
    fractions = binned.loc[valid, "isotonic_completeness"].to_numpy(float)
    rows = []
    for percentage in targets:
        distance, status = crossing_distance(
            distances, fractions, percentage / 100.0
        )
        rows.append({
            "completeness_percent": percentage,
            "distance_label": f"D{percentage:g}",
            "luminosity_distance_mpc": distance,
            "status": status,
        })
    return pd.DataFrame(rows)


def save_plot(
    binned: pd.DataFrame,
    reaches: pd.DataFrame,
    pair: str,
    args: argparse.Namespace,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    valid = binned["sufficient_bin_population"].astype(bool)
    x = binned.loc[valid, "distance_median_mpc"].to_numpy(float)
    y = binned.loc[valid, "colour_completeness"].to_numpy(float)
    low = binned.loc[valid, "wilson_low"].to_numpy(float)
    high = binned.loc[valid, "wilson_high"].to_numpy(float)
    ax.errorbar(
        x,
        100.0 * y,
        yerr=np.vstack((100.0 * (y - low), 100.0 * (high - y))),
        fmt="o",
        markersize=4.5,
        color="#155a96",
        ecolor="#7aa6cb",
        capsize=2,
        label=f"Distance bins ({args.confidence_level * 100:g}% Wilson interval)",
    )
    fitted = binned.loc[valid, "isotonic_completeness"].to_numpy(float)
    ax.step(
        x,
        100.0 * fitted,
        where="mid",
        color="#0b3c5d",
        linewidth=2.0,
        label="Weighted monotonic fit",
    )
    colours = ["#8b1e3f", "#ad5d1e", "#9a7d0a", "#3f7d20", "#6252a3"]
    for colour, row in zip(colours, reaches.itertuples()):
        target = float(row.completeness_percent)
        ax.axhline(target, color=colour, linewidth=0.8, alpha=0.38)
        if np.isfinite(row.luminosity_distance_mpc):
            ax.axvline(
                row.luminosity_distance_mpc,
                color=colour,
                linestyle="--",
                linewidth=1.2,
                label=f"{row.distance_label} = {row.luminosity_distance_mpc:.0f} Mpc",
            )
    ax.set_xlabel("Luminosity distance [Mpc]")
    ax.set_ylabel(f"KNe with at least one measurable {pair} colour [%]")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.interactive or len(sys.argv) == 1:
        args = interactive(args)
    root, bands, targets = validate(args)
    metadata = load_summary(root).drop_duplicates("event_id").copy()
    required = {"event_id", "luminosity_distance", "ebv_mw"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"summary.csv is missing {sorted(missing)}")
    metadata["event_id"] = metadata["event_id"].map(canonical_event_id)
    metadata["luminosity_distance"] = pd.to_numeric(
        metadata["luminosity_distance"], errors="coerce"
    )
    metadata = metadata.loc[
        np.isfinite(metadata["luminosity_distance"])
        & (metadata["luminosity_distance"] > 0)
    ].reset_index(drop=True)
    if metadata["luminosity_distance"].nunique() < 3:
        raise ValueError(
            "D50/D10/D1 require a variable-distance population. This run is "
            "fixed-distance; generate a comoving-volume run spanning the desired range."
        )
    centres = args.t_min + (np.arange(
        round((args.t_max - args.t_min) / args.bin_width)
    ) + 0.5) * args.bin_width
    m5_table = discover_m5_table(root, args.m5_quantile_table)
    thresholds = load_m5(
        m5_table, args.threshold_scope, args.m5_scenario, bands
    )
    print(
        "LSST thresholds: "
        + ", ".join(f"{band}={thresholds[band]:.4f}" for band in bands)
    )
    detected = event_detectability(
        root,
        metadata,
        bands,
        centres,
        thresholds,
        args.mw_treatment,
    )
    binned = bin_completeness(metadata, detected, args)
    reaches = reach_table(binned, targets)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root
        / "analysis"
        / f"lsst_distance_reach_{bands[0]}_{bands[1]}_{args.m5_scenario}"
    )
    output.mkdir(parents=True, exist_ok=True)
    binned_path = output / "distance_completeness_by_bin.csv"
    reaches_path = output / "distance_reach_levels.csv"
    plot_path = output / "lsst_colour_distance_completeness.png"
    binned.to_csv(binned_path, index=False)
    reaches.to_csv(reaches_path, index=False)
    save_plot(binned, reaches, f"{bands[0]}-{bands[1]}", args, plot_path)
    configuration = {
        "run_dir": str(root),
        "colour_pair": f"{bands[0]}-{bands[1]}",
        "time_window_days": [args.t_min, args.t_max],
        "temporal_bin_width_days": args.bin_width,
        "colour_measurable_definition": (
            "both interpolated band magnitudes finite and <= their global m5 "
            "threshold in at least one temporal bin"
        ),
        "m5_quantile_table": str(m5_table),
        "m5_scenario": args.m5_scenario,
        "m5_threshold_scope": args.threshold_scope,
        "m5_thresholds": thresholds,
        "mw_treatment": args.mw_treatment,
        "distance_bin_width_mpc": args.distance_bin_width_mpc,
        "minimum_events_per_bin": args.minimum_events_per_bin,
        "confidence_level": args.confidence_level,
        "interpolation": "weighted decreasing isotonic regression",
        "important_limitation": (
            "global synthetic m5 availability; Rubin cadence and photometric "
            "noise are not simulated in this reach statistic"
        ),
    }
    (output / "distance_reach_configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    for row in reaches.itertuples():
        value = (
            f"{row.luminosity_distance_mpc:.1f} Mpc"
            if np.isfinite(row.luminosity_distance_mpc)
            else f"not measured ({row.status})"
        )
        print(f"{row.distance_label}: {value}")
    print("Plot:", plot_path)
    print("Binned values:", binned_path)
    print("Reach levels:", reaches_path)


if __name__ == "__main__":
    main()
