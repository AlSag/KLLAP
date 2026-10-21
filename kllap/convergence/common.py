"""Shared parsing, aggregation, and plotting for convergence commands.

This module contains no command-line entry point and performs no analysis when
imported. Both convergence commands depend on these definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kllap.envelope.core import M5_STAGES, STAGES
from kllap.m5 import M5_SCENARIOS


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
