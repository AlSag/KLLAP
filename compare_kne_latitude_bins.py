#!/usr/bin/env python3
"""Compare KNe losses in disjoint Galactic-latitude bins.

Each input is an ``event_window_availability.csv`` produced for one disjoint
interval in |b|.  The lowest-latitude interval is kept as a fixed reference:

    R_k = f_loss(bin k) / f_loss(reference bin)

The primary quantity is the additional loss caused by Milky-Way extinction
among events that still had at least one m5-supported colour bin before dust.
The m5-only and combined losses are plotted as controls.

The operational plateau is the first bin for which *all* subsequent MW ratios
remain within a configurable relative tolerance of their common median.  A
paired event bootstrap supplies confidence bands, a distribution for the
plateau boundary and the fraction of bootstrap samples in which a plateau is
found.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "event_id",
    "abs_b_deg",
    "N_bins_baseline",
    "N_bins_m5_no_mw",
    "N_bins_m5_with_mw",
}
COMPONENTS = ("m5_only", "mw_additional", "combined")
COLORS = {
    "m5_only": "#3568a8",
    "mw_additional": "#d97904",
    "combined": "#6b2e8c",
}
LABELS = {
    "m5_only": "m5 only (no MW; control)",
    "mw_additional": "Additional MW loss with m5",
    "combined": "m5 + MW (total)",
}


@dataclass(frozen=True)
class LatitudeBin:
    lower: float
    upper: float
    path: Path
    table: pd.DataFrame

    @property
    def centre(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def label(self) -> str:
        closing = "]" if np.isclose(self.upper, 90.0) else "["
        return f"[{self.lower:g}, {self.upper:g}{closing}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare relative KNe losses in disjoint |b| bins using the first "
            "bin as a fixed reference. Exactly one plot and one PNG table are written."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bin",
        dest="bins",
        action="append",
        nargs=3,
        metavar=("BMIN", "BMAX", "EVENT_TABLE"),
        help="Repeat for every disjoint latitude interval",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--confidence-level", type=float, choices=[0.68, 0.95], default=0.95
    )
    parser.add_argument(
        "--plateau-relative-tolerance",
        type=float,
        default=0.10,
        help="Maximum |R/median(R_tail)-1| allowed for every bin in the tail",
    )
    parser.add_argument(
        "--minimum-plateau-bins",
        type=int,
        default=3,
        help="Minimum number of bins, including the candidate bin, in the plateau tail",
    )
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--linear-y", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_bin(lower_text: str, upper_text: str, path_text: str) -> LatitudeBin:
    lower = float(lower_text)
    upper = float(upper_text)
    if not 0.0 <= lower < upper <= 90.0:
        raise ValueError(f"Invalid latitude interval [{lower}, {upper})")
    path = Path(path_text).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.rglob("event_window_availability.csv"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one event_window_availability.csv below {path}; "
                f"found {len(candidates)}"
            )
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, dtype={"event_id": str})
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if table["event_id"].isna().any() or table["event_id"].duplicated().any():
        raise ValueError(f"event_id must be unique and non-empty in {path}")
    numeric = [
        "abs_b_deg",
        "N_bins_baseline",
        "N_bins_m5_no_mw",
        "N_bins_m5_with_mw",
    ]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if not np.isfinite(table[numeric].to_numpy(float)).all():
        raise ValueError(f"Non-finite required values in {path}")
    if not (
        (table["N_bins_m5_with_mw"] <= table["N_bins_m5_no_mw"])
        & (table["N_bins_m5_no_mw"] <= table["N_bins_baseline"])
    ).all():
        raise ValueError(
            f"Expected N_m5_with_mw <= N_m5_no_mw <= N_baseline in {path}"
        )
    eps = 1e-8
    latitude = table["abs_b_deg"].to_numpy(float)
    inside = latitude >= lower - eps
    if np.isclose(upper, 90.0):
        inside &= latitude <= upper + eps
    else:
        inside &= latitude < upper + eps
    if not inside.all():
        bad = latitude[~inside]
        raise ValueError(
            f"{len(bad)} objects in {path} lie outside [{lower:g}, {upper:g}); "
            f"observed bad range={bad.min():g}..{bad.max():g} deg"
        )
    valid = table["N_bins_baseline"].gt(0)
    outcomes = pd.DataFrame(
        {
            "event_id": table["event_id"],
            "valid_baseline": valid,
            "loss_m5_only": valid & table["N_bins_m5_no_mw"].eq(0),
            "loss_mw_additional": (
                valid
                & table["N_bins_m5_no_mw"].gt(0)
                & table["N_bins_m5_with_mw"].eq(0)
            ),
            "loss_combined": valid & table["N_bins_m5_with_mw"].eq(0),
        }
    )
    decomposition = (
        outcomes["loss_m5_only"].astype(int)
        + outcomes["loss_mw_additional"].astype(int)
    )
    if not decomposition.eq(outcomes["loss_combined"].astype(int)).all():
        raise AssertionError(f"Loss decomposition is inconsistent in {path}")
    return LatitudeBin(lower, upper, path, outcomes)


def parse_bins(raw: list[list[str]] | None) -> list[LatitudeBin]:
    if not raw or len(raw) < 4:
        raise ValueError("Provide at least four disjoint --bin inputs")
    bins = sorted((load_bin(*item) for item in raw), key=lambda item: item.lower)
    for previous, current in zip(bins[:-1], bins[1:]):
        if not np.isclose(previous.upper, current.lower):
            raise ValueError(
                "Latitude bins must be contiguous and disjoint: "
                f"{previous.label} followed by {current.label}"
            )
    return bins


def align_outcomes(bins: list[LatitudeBin]) -> np.ndarray:
    reference_ids = bins[0].table["event_id"].tolist()
    reference_set = set(reference_ids)
    aligned = []
    columns = [
        "valid_baseline",
        "loss_m5_only",
        "loss_mw_additional",
        "loss_combined",
    ]
    for item in bins:
        ids = set(item.table["event_id"])
        if ids != reference_set:
            raise ValueError(
                f"Paired analysis requires identical event_id sets; mismatch in {item.path}"
            )
        indexed = item.table.set_index("event_id").loc[reference_ids]
        aligned.append(indexed[columns].to_numpy(bool))
    return np.stack(aligned, axis=0)  # bin, event, outcome


def counts_from_outcomes(outcomes: np.ndarray) -> np.ndarray:
    return outcomes.sum(axis=1).astype(int)


def rates_and_ratios(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denominators = counts[:, :1]
    rates = np.divide(
        counts[:, 1:],
        denominators,
        out=np.full(counts[:, 1:].shape, np.nan, dtype=float),
        where=denominators > 0,
    )
    reference = rates[0]
    ratios = np.divide(
        rates,
        reference[None, :],
        out=np.full_like(rates, np.nan),
        where=reference[None, :] > 0,
    )
    return rates, ratios


def plateau_threshold(
    lower_edges: np.ndarray,
    ratios: np.ndarray,
    tolerance: float,
    minimum_bins: int,
) -> tuple[float, float, float]:
    """Return first tail stable around its median, its level and max deviation."""
    n_bins = len(ratios)
    # Index 0 is the fixed low-latitude reference and is not a plateau candidate.
    for start in range(1, n_bins - minimum_bins + 1):
        tail = ratios[start:]
        if len(tail) < minimum_bins or not np.isfinite(tail).all() or np.any(tail <= 0):
            continue
        level = float(np.median(tail))
        deviation = float(np.max(np.abs(tail / level - 1.0)))
        if deviation <= tolerance:
            return float(lower_edges[start]), level, deviation
    return np.nan, np.nan, np.nan


def paired_bootstrap(
    outcomes: np.ndarray,
    repetitions: int,
    seed: int,
    lower_edges: np.ndarray,
    tolerance: float,
    minimum_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, n_events, _ = outcomes.shape
    rng = np.random.default_rng(seed)
    ratios = np.full((repetitions, outcomes.shape[0], 3), np.nan, dtype=float)
    thresholds = np.full(repetitions, np.nan, dtype=float)
    for repetition in range(repetitions):
        indices = rng.integers(0, n_events, size=n_events)
        counts = outcomes[:, indices, :].sum(axis=1)
        _, repetition_ratios = rates_and_ratios(counts)
        ratios[repetition] = repetition_ratios
        thresholds[repetition] = plateau_threshold(
            lower_edges,
            repetition_ratios[:, 1],
            tolerance,
            minimum_bins,
        )[0]
    return ratios, thresholds


def confidence_bounds(samples: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - confidence) / 2.0
    with np.errstate(all="ignore"):
        return (
            np.nanquantile(samples, alpha, axis=0),
            np.nanquantile(samples, 1.0 - alpha, axis=0),
        )


def threshold_summary(samples: np.ndarray, confidence: float) -> dict[str, float]:
    finite = samples[np.isfinite(samples)]
    success = len(finite) / len(samples)
    if len(finite) == 0:
        return {"median": np.nan, "low": np.nan, "high": np.nan, "success": success}
    alpha = (1.0 - confidence) / 2.0
    return {
        "median": float(np.median(finite)),
        "low": float(np.quantile(finite, alpha)),
        "high": float(np.quantile(finite, 1.0 - alpha)),
        "success": float(success),
    }


def interval_x_errors(bins: list[LatitudeBin]) -> np.ndarray:
    centres = np.array([item.centre for item in bins], dtype=float)
    lower = centres - np.array([item.lower for item in bins], dtype=float)
    upper = np.array([item.upper for item in bins], dtype=float) - centres
    return np.vstack([lower, upper])


def save_plot(
    bins: list[LatitudeBin],
    ratios: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    observed_plateau: tuple[float, float, float],
    bootstrap_plateau: dict[str, float],
    confidence: float,
    tolerance: float,
    linear_y: bool,
    output: Path,
) -> None:
    centres = np.array([item.centre for item in bins], dtype=float)
    xerr = interval_x_errors(bins)
    fig, ax = plt.subplots(figsize=(10.8, 6.5), constrained_layout=True)
    for component_index, component in enumerate(COMPONENTS):
        y = ratios[:, component_index]
        lower = ci_low[:, component_index]
        upper = ci_high[:, component_index]
        if not linear_y:
            y = np.where(y > 0, y, np.nan)
            lower = np.where(lower > 0, lower, np.nan)
            upper = np.where(upper > 0, upper, np.nan)
        ax.fill_between(centres, lower, upper, color=COLORS[component], alpha=0.14)
        ax.errorbar(
            centres,
            y,
            xerr=xerr,
            fmt="o-",
            ms=4.8,
            lw=1.8,
            capsize=2.5,
            color=COLORS[component],
            label=LABELS[component],
        )
    ax.axhline(1.0, color="#555555", ls=":", lw=1.3, label="Reference-bin loss rate")
    threshold, level, _ = observed_plateau
    if np.isfinite(threshold):
        if np.isfinite(bootstrap_plateau["low"]):
            ax.axvspan(
                bootstrap_plateau["low"],
                bootstrap_plateau["high"],
                color="#16866b",
                alpha=0.09,
            )
        ax.axvline(
            threshold,
            color="#16866b",
            ls="--",
            lw=1.7,
            label=f"Observed MW plateau starts at {threshold:g} deg",
        )
        ax.hlines(
            level,
            threshold,
            bins[-1].upper,
            color="#16866b",
            lw=2.0,
            label="Observed MW plateau level",
        )
    if not linear_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"Disjoint absolute Galactic-latitude bin $|b|$ [deg]")
    reference_text = f"[{bins[0].lower:g}°, {bins[0].upper:g}°["
    ax.set_ylabel(
        r"Relative loss rate $R=f_{\rm loss}(\mathrm{bin})/f_{\rm loss}"
        f"(reference {reference_text})$"
    )
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False, fontsize=8.3, ncol=1, loc="center right")
    confidence_text = f"{confidence:.0%} paired-event bootstrap"
    if np.isfinite(bootstrap_plateau["median"]):
        plateau_text = (
            f"bootstrap b*={bootstrap_plateau['median']:.1f} "
            f"[{bootstrap_plateau['low']:.1f}, {bootstrap_plateau['high']:.1f}] deg; "
            f"detected in {100 * bootstrap_plateau['success']:.1f}%"
        )
    else:
        plateau_text = f"plateau not detected; {100 * bootstrap_plateau['success']:.1f}% bootstrap"
    ax.text(
        0.995,
        0.012,
        f"Bands: {confidence_text}. Tail tolerance: {100*tolerance:.0f}%. {plateau_text}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#555555",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def ratio_text(value: float, low: float, high: float) -> str:
    if not np.isfinite(value):
        return "undefined"
    return f"{100*value:.2f} [{100*low:.2f}, {100*high:.2f}]"


def save_table(
    bins: list[LatitudeBin],
    counts: np.ndarray,
    rates: np.ndarray,
    ratios: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    observed_plateau: tuple[float, float, float],
    bootstrap_plateau: dict[str, float],
    confidence: float,
    tolerance: float,
    output: Path,
) -> None:
    threshold = observed_plateau[0]
    rows = []
    colours = []
    for index, item in enumerate(bins):
        label = item.label
        if np.isfinite(threshold) and np.isclose(item.lower, threshold):
            label += " *"
        mw_ratio = ratios[index, 1]
        reduction = 100 * (1.0 - mw_ratio) if np.isfinite(mw_ratio) else np.nan
        rows.append(
            [
                label,
                f"{counts[index, 0]:,}",
                f"{counts[index, 2]:,}",
                f"{100*rates[index, 1]:.2f}",
                ratio_text(mw_ratio, ci_low[index, 1], ci_high[index, 1]),
                f"{reduction:.2f}" if np.isfinite(reduction) else "undefined",
                ratio_text(ratios[index, 0], ci_low[index, 0], ci_high[index, 0]),
                ratio_text(ratios[index, 2], ci_low[index, 2], ci_high[index, 2]),
            ]
        )
        colours.append("#d8eee8" if "*" in label else ("#f4f6f8" if index % 2 else "#ffffff"))
    columns = [
        r"$|b|$ bin [deg]",
        "N valid",
        "MW losses",
        "MW loss rate [%]",
        f"MW R [{confidence:.0%} CI] %",
        "MW reduction [%]",
        f"m5 R [{confidence:.0%} CI] %",
        f"Total R [{confidence:.0%} CI] %",
    ]
    height = max(4.0, 0.48 * len(rows) + 2.1)
    fig, ax = plt.subplots(figsize=(15.2, height), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        colWidths=[0.10, 0.08, 0.09, 0.12, 0.18, 0.13, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    table.scale(1.0, 1.65)
    for column in range(len(columns)):
        cell = table[(0, column)]
        cell.set_facecolor("#30445e")
        cell.set_text_props(color="white", weight="bold")
    for row_index, colour in enumerate(colours, start=1):
        for column in range(len(columns)):
            table[(row_index, column)].set_facecolor(colour)
            table[(row_index, column)].set_edgecolor("#cbd2da")
            table[(row_index, column)].set_linewidth(0.5)

    if np.isfinite(threshold):
        observed_text = (
            f"Observed operational plateau: b*={threshold:g} deg; "
            f"all subsequent central MW ratios lie within {100*tolerance:.0f}% "
            "of their median."
        )
    else:
        observed_text = (
            f"No observed operational plateau satisfies the {100*tolerance:.0f}% "
            "all-subsequent-bins criterion."
        )
    if np.isfinite(bootstrap_plateau["median"]):
        bootstrap_text = (
            f"Bootstrap b*: {bootstrap_plateau['median']:.2f} "
            f"[{bootstrap_plateau['low']:.2f}, {bootstrap_plateau['high']:.2f}] deg "
            f"({confidence:.0%}); detected in {100*bootstrap_plateau['success']:.1f}% "
            "of paired-event resamples."
        )
    else:
        bootstrap_text = (
            f"Bootstrap plateau not robustly detected "
            f"({100*bootstrap_plateau['success']:.1f}% of resamples)."
        )
    fig.text(
        0.015,
        0.105,
        "R = bin loss rate / fixed reference-bin loss rate. MW losses are events "
        "with m5 support before dust and no surviving colour bin after dust.",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.text(0.015, 0.064, observed_text, ha="left", va="bottom", fontsize=8.5)
    fig.text(0.015, 0.023, bootstrap_text, ha="left", va="bottom", fontsize=8.5)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    if not 0 < args.plateau_relative_tolerance < 1:
        raise ValueError("--plateau-relative-tolerance must lie in (0, 1)")
    if args.minimum_plateau_bins < 2:
        raise ValueError("--minimum-plateau-bins must be at least 2")
    bins = parse_bins(args.bins)
    if args.minimum_plateau_bins >= len(bins):
        raise ValueError("Not enough non-reference bins for the plateau criterion")
    output_dir = args.output_dir.expanduser().resolve()
    plot_path = output_dir / "kne_latitude_bin_relative_losses.png"
    table_path = output_dir / "kne_latitude_bin_relative_losses_values_table.png"
    existing = [path for path in (plot_path, table_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; use --overwrite: " + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes = align_outcomes(bins)
    counts = counts_from_outcomes(outcomes)
    rates, ratios = rates_and_ratios(counts)
    if not np.isfinite(ratios[:, 1]).all():
        raise ValueError("MW reference loss rate is zero; relative MW ratios are undefined")
    bootstrap_ratios, bootstrap_thresholds = paired_bootstrap(
        outcomes,
        args.bootstrap_repetitions,
        args.seed,
        np.array([item.lower for item in bins]),
        args.plateau_relative_tolerance,
        args.minimum_plateau_bins,
    )
    ci_low, ci_high = confidence_bounds(bootstrap_ratios, args.confidence_level)
    observed_plateau = plateau_threshold(
        np.array([item.lower for item in bins]),
        ratios[:, 1],
        args.plateau_relative_tolerance,
        args.minimum_plateau_bins,
    )
    bootstrap_plateau = threshold_summary(
        bootstrap_thresholds, args.confidence_level
    )
    save_plot(
        bins,
        ratios,
        ci_low,
        ci_high,
        observed_plateau,
        bootstrap_plateau,
        args.confidence_level,
        args.plateau_relative_tolerance,
        args.linear_y,
        plot_path,
    )
    save_table(
        bins,
        counts,
        rates,
        ratios,
        ci_low,
        ci_high,
        observed_plateau,
        bootstrap_plateau,
        args.confidence_level,
        args.plateau_relative_tolerance,
        table_path,
    )
    print(f"Reference bin: {bins[0].label} deg")
    if np.isfinite(observed_plateau[0]):
        print(f"Observed operational plateau starts at: {observed_plateau[0]:g} deg")
    else:
        print("Observed operational plateau: not detected")
    if np.isfinite(bootstrap_plateau["median"]):
        print(
            f"Bootstrap plateau: {bootstrap_plateau['median']:.2f} "
            f"[{bootstrap_plateau['low']:.2f}, {bootstrap_plateau['high']:.2f}] deg; "
            f"success={100*bootstrap_plateau['success']:.1f}%"
        )
    else:
        print(
            "Bootstrap plateau: not robustly detected; "
            f"success={100*bootstrap_plateau['success']:.1f}%"
        )
    print(f"Plot: {plot_path}")
    print(f"Table: {table_path}")


if __name__ == "__main__":
    main()
