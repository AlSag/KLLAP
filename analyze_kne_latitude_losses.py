#!/usr/bin/env python3
"""Measure KNe colour-availability losses against Galactic latitude.

Read summary.csv and saved synthetic light curves from a generated run.  At
each fixed positive time-bin centre, interpolate the two bands in log10(t),
without extrapolation, as in build_color_envelopes.py.  Only
numerically supported two-band colours are counted in the denominator.

Four scenarios are evaluated on the same event/bin opportunities:
  baseline:        intrinsic, no m5
  MW alone:        extincted, no m5 (same availability as baseline)
  m5 alone:        intrinsic band 1 AND band 2 <= their Rubin m5 thresholds
  m5 plus MW:      extincted band 1 AND band 2 <= their Rubin m5 thresholds

The additional MW loss is m5-alone survivors minus m5-plus-MW survivors,
not a separate standalone detection cut.  The two loss components add
exactly to the total loss (provided Galactic extinction is non-negative).
Figures aggregate across the entire selected temporal window.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kne_pipeline_common import discover_m5_quantile_table


LSST_R = {
    "u": 4.757217815396922, "g": 3.6605664439892616,
    "r": 2.70136780871597, "i": 2.0536599130965882,
    "z": 1.5900964472616756, "y": 1.3077049588254708,
}
PS1_R = {"g": 3.172, "r": 2.271, "i": 1.682, "z": 1.322, "y": 1.087}
LATITUDE_COLUMNS = (
    "galactic_b_deg", "galactic_latitude_deg", "b_gal_deg", "b_deg"
)
COMPONENTS = ("m5_only", "additional_mw_given_m5", "combined")
COLORS = {
    "m5_only": "#395f9b",
    "additional_mw_given_m5": "#d28219",
    "combined": "#652f81",
}
LABELS = {
    "m5_only": "m5 only (no MW)",
    "additional_mw_given_m5": "Additional MW loss with m5",
    "combined": "m5 + MW (total)",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, help="A new output directory")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow writing into a non-empty output directory")
    p.add_argument("--color-pair", default="g-r")
    p.add_argument("--photometric-system", choices=["lsst", "ps1"], default="lsst")
    p.add_argument("--t-min", type=float, default=0.4)
    p.add_argument("--t-max", type=float, default=16.0)
    p.add_argument("--bin-width", type=float, default=0.4)
    p.add_argument("--m5-quantile-table", type=Path)
    p.add_argument("--m5-scenario", choices=["worst_p25", "median_p50", "best_p75"],
                   default="median_p50")
    p.add_argument("--threshold-scope", choices=["common_global", "latitude_conditioned"],
                   default="common_global",
                   help="Which *single* m5 threshold table to use for every latitude")
    p.add_argument("--latitude-bin-width", type=float, default=5.0)
    p.add_argument("--threshold-min", type=float, default=5.0)
    p.add_argument("--threshold-max", type=float, default=70.0)
    p.add_argument("--threshold-step", type=float, default=1.0)
    p.add_argument("--min-events-per-side", type=int, default=100,
                   help="Minimum baseline-supported objects on EACH side of a cut")
    p.add_argument("--plateau-tolerance", type=float, default=0.02,
                   help="Max absolute high-latitude rate difference (0.02 = 2 percentage points)")
    p.add_argument("--plateau-ratio-tolerance", type=float, default=0.10,
                   help="Maximum absolute change of low/high loss-rate ratio")
    p.add_argument("--plateau-min-thresholds", type=int, default=3,
                   help="At least this many valid cut thresholds to define a plateau")
    return p


def canonical_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text))).zfill(6)
    return text


def time_centres(t_min: float, t_max: float, width: float) -> np.ndarray:
    if t_min <= 0 or width <= 0 or t_max <= t_min:
        raise ValueError("Require 0 < t-min < t-max and bin-width > 0")
    n = (t_max - t_min) / width
    if not np.isclose(n, round(n), atol=1e-8):
        raise ValueError("(t-max - t-min) must be an integer multiple of bin-width")
    return t_min + (np.arange(round(n)) + 0.5) * width


def find_m5_table(root: Path, explicit: Path | None) -> Path:
    return discover_m5_quantile_table(root, explicit=explicit)


def read_m5(path: Path, scope: str, scenario: str, bands: tuple[str, str]) -> dict[str, float]:
    table = pd.read_csv(path)
    required = {"threshold_scope", "threshold_latitude_group", "band", "scenario", "m5"}
    if missing := required - set(table.columns):
        raise ValueError(f"Missing columns in m5 table: {sorted(missing)}")
    group = "both" if scope == "common_global" else "high"
    selected = table.loc[
        table["threshold_scope"].eq(scope)
        & table["threshold_latitude_group"].eq(group)
        & table["scenario"].eq(scenario)
        & table["band"].astype(str).str.lower().isin(bands)
    ]
    if selected["band"].duplicated().any():
        raise ValueError(f"Ambiguous repeated m5 thresholds in {path}")
    result = {str(row.band).lower(): float(row.m5) for row in selected.itertuples()}
    if set(bands) != set(result) or not all(np.isfinite(v) for v in result.values()):
        raise ValueError(f"Missing/non-finite {scenario} m5 for {bands} in {path}")
    return result


def load_events(root: Path, system: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    summary_path = root / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path, dtype={"event_id": str})
    if "event_id" not in summary or "ebv_mw" not in summary:
        raise ValueError("summary.csv requires event_id and ebv_mw")
    latitude = next((column for column in LATITUDE_COLUMNS if column in summary), None)
    if latitude is None:
        raise ValueError(
            "summary.csv has no Galactic latitude column; an all-latitude run "
            "with saved galactic_latitude_deg (or alias) is required"
        )
    summary["event_id"] = summary["event_id"].map(canonical_id)
    if summary["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id in summary.csv")
    summary["abs_b_deg"] = pd.to_numeric(summary[latitude], errors="coerce").abs()
    summary["ebv_mw"] = pd.to_numeric(summary["ebv_mw"], errors="coerce")
    good = summary["abs_b_deg"].between(0, 90) & np.isfinite(summary["ebv_mw"])
    if (summary.loc[good, "ebv_mw"] < 0).any():
        raise ValueError("Negative E(B-V) in summary.csv")
    removed = len(summary) - int(good.sum())
    if removed:
        print(f"Warning: {removed:,} injected events with missing b/E(B-V) excluded")
    summary = summary.loc[good].copy()
    if summary.empty:
        raise ValueError("No event has usable Galactic latitude and E(B-V)")
    paths = sorted((root / "parquet").glob("synthetic_part_*.parquet"))
    if paths:
        return summary.set_index("event_id", drop=False), tuple(map(str, paths))
    csv_paths = sorted((root / "csv").glob("lightcurve_*.csv"))
    if csv_paths:
        return summary.set_index("event_id", drop=False), tuple(map(str, csv_paths))
    raise FileNotFoundError(
        "No saved synthetic curves: expected parquet/synthetic_part_*.parquet "
        "or csv/lightcurve_*.csv"
    )


def iter_curves(paths: tuple[str, ...], bands: tuple[str, str], system: str):
    parquet = paths[0].endswith(".parquet")
    for filename in paths:
        path = Path(filename)
        if parquet:
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("Install pyarrow to read synthetic Parquet shards") from exc
            schema = set(pq.read_schema(path).names)
            has_system = "photometric_system" in schema
            if system != "lsst" and not has_system:
                raise ValueError(f"{path} has no PS1 photometry")
            columns = ["event_id", "t_days", "band", "mag"]
            filters: list[tuple[str, str, object]] = [("band", "in", list(bands))]
            if has_system:
                columns.append("photometric_system")
                filters.append(
                    ("photometric_system", "in", ["lsst", "ps1"] if system == "ps1" else ["lsst"])
                )
            shard = pd.read_parquet(path, columns=columns, filters=filters)
        else:
            shard = pd.read_csv(path)
            if "event_id" not in shard:
                match = re.search(r"(\d+)$", path.stem)
                if not match:
                    raise ValueError(f"Cannot infer event ID from {path}")
                shard["event_id"] = match.group(1)
            has_system = "photometric_system" in shard
            if system != "lsst" and not has_system:
                raise ValueError(f"{path} has no PS1 photometry")
        for column in ("event_id", "t_days", "band", "mag"):
            if column not in shard:
                raise ValueError(f"{path} missing required column {column}")
        if has_system:
            wanted = ("lsst", "ps1") if system == "ps1" else ("lsst",)
            shard = shard.loc[
                shard["photometric_system"].astype(str).str.strip().str.lower().isin(wanted)
            ]
        else:
            shard["photometric_system"] = "lsst"
        shard = shard.loc[shard["band"].astype(str).str.strip().str.lower().isin(bands)].copy()
        if shard.empty:
            continue
        shard["event_id"] = shard["event_id"].map(canonical_id)
        shard["band"] = shard["band"].astype(str).str.strip().str.lower()
        shard["photometric_system"] = (
            shard["photometric_system"].astype(str).str.strip().str.lower()
        )
        shard["t_days"] = pd.to_numeric(shard["t_days"], errors="coerce")
        shard["mag"] = pd.to_numeric(shard["mag"], errors="coerce")
        for event_id, curve in shard.groupby("event_id", sort=False):
            yield event_id, curve


def interpolate(curve: pd.DataFrame, band: str, centres: np.ndarray) -> np.ndarray:
    sub = curve.loc[curve["band"].eq(band), ["t_days", "mag"]]
    sub = sub.loc[
        np.isfinite(sub["t_days"]) & (sub["t_days"] > 0)
        & np.isfinite(sub["mag"])
    ]
    sub = sub.groupby("t_days", as_index=False)["mag"].median().sort_values("t_days")
    result = np.full(len(centres), np.nan, dtype=float)
    if len(sub) < 2:
        return result
    x = sub["t_days"].to_numpy(float)
    supported = (centres >= x[0]) & (centres <= x[-1])
    result[supported] = np.interp(
        np.log10(centres[supported]), np.log10(x), sub["mag"].to_numpy(float)
    )
    return result


def extinction(row: pd.Series, system: str, band: str) -> float:
    column = f"A_{system}_{band}_mw"
    if column in row and np.isfinite(pd.to_numeric(row[column], errors="coerce")):
        a = float(row[column])
    else:
        a = float(row["ebv_mw"]) * (
            LSST_R if system == "lsst" else PS1_R
        )[band]
    if a < 0 or not np.isfinite(a):
        raise ValueError(f"Invalid Galactic extinction for {row['event_id']} {band}")
    return a


def count_event_bins(
    metadata: pd.DataFrame, paths: tuple[str, ...],
    bands: tuple[str, str], system: str, centres: np.ndarray, m5: dict[str, float],
) -> pd.DataFrame:
    records = {event_id: {"N_bins_baseline": 0, "N_bins_mw_only": 0,
                          "N_bins_m5_no_mw": 0, "N_bins_m5_with_mw": 0}
               for event_id in metadata.index}
    seen: set[str] = set()
    seen_lsst_rows = False
    for event_id, curve in iter_curves(paths, bands, system):
        if event_id not in records:
            continue
        if event_id in seen:
            raise ValueError(f"Duplicate saved rows for event {event_id} in multiple files")
        seen.add(event_id)
        row = metadata.loc[event_id]
        colour_curve = curve.loc[curve["photometric_system"].eq(system)]
        selection_curve = curve.loc[curve["photometric_system"].eq("lsst")]
        seen_lsst_rows |= not selection_curve.empty
        colour_saved = [interpolate(colour_curve, band, centres) for band in bands]
        selection_saved = [interpolate(selection_curve, band, centres) for band in bands]
        selection_intrinsic = [
            mag - extinction(row, "lsst", band)
            for mag, band in zip(selection_saved, bands)
        ]
        supported = np.isfinite(colour_saved[0]) & np.isfinite(colour_saved[1])
        no_mw = (
            supported & np.isfinite(selection_intrinsic[0])
            & np.isfinite(selection_intrinsic[1])
            & (selection_intrinsic[0] <= m5[bands[0]])
            & (selection_intrinsic[1] <= m5[bands[1]])
        )
        with_mw = (
            supported & np.isfinite(selection_saved[0]) & np.isfinite(selection_saved[1])
            & (selection_saved[0] <= m5[bands[0]])
            & (selection_saved[1] <= m5[bands[1]])
        )
        if np.any(with_mw & ~no_mw):
            raise AssertionError(f"MW unexpectedly increased m5 availability for {event_id}")
        records[event_id] = {
            "N_bins_baseline": int(supported.sum()),
            "N_bins_mw_only": int(supported.sum()),
            "N_bins_m5_no_mw": int(no_mw.sum()),
            "N_bins_m5_with_mw": int(with_mw.sum()),
        }
        if len(seen) % 5000 == 0:
            print(f"  Read {len(seen):,}/{len(metadata):,} selected events", flush=True)
    if system == "ps1" and not seen_lsst_rows:
        raise ValueError(
            "PS1 colours require paired LSST rows to evaluate Rubin m5; "
            "none were found for the selected events"
        )
    counts = pd.DataFrame.from_dict(records, orient="index")
    counts.index.name = "event_id"
    result = metadata[["abs_b_deg", "ebv_mw"]].join(counts)
    result["has_saved_rows_in_requested_bands"] = result.index.isin(seen)
    for suffix in ("baseline", "mw_only", "m5_no_mw", "m5_with_mw"):
        result[f"has_any_{suffix}"] = result[f"N_bins_{suffix}"] > 0
    result["N_bins_lost_m5_only"] = result["N_bins_baseline"] - result["N_bins_m5_no_mw"]
    result["N_bins_lost_additional_mw_given_m5"] = (
        result["N_bins_m5_no_mw"] - result["N_bins_m5_with_mw"]
    )
    result["N_bins_lost_combined"] = result["N_bins_baseline"] - result["N_bins_m5_with_mw"]
    result["N_bins_lost_mw_only"] = result["N_bins_baseline"] - result["N_bins_mw_only"]
    if not (result["N_bins_lost_m5_only"] + result["N_bins_lost_additional_mw_given_m5"]
            == result["N_bins_lost_combined"]).all():
        raise AssertionError("Loss decomposition does not sum to the combined loss")
    print(f"Saved two-band rows found for {len(seen):,}/{len(metadata):,} injected events")
    return result.reset_index()


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def wilson(success: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = success / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z / denom * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2))
    return float(centre - spread), float(centre + spread)


def rate_ratio_ci95(lost_low: int, total_low: int, lost_high: int,
                    total_high: int) -> tuple[float, float]:
    """Approximate independent-event log-rate-ratio CI; undefined at zero losses."""
    if min(lost_low, total_low, lost_high, total_high) <= 0:
        return float("nan"), float("nan")
    rr = (lost_low / total_low) / (lost_high / total_high)
    se = np.sqrt(
        1 / lost_low - 1 / total_low + 1 / lost_high - 1 / total_high
    )
    return float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))


def metrics(group: pd.DataFrame) -> dict[str, float | int]:
    n = len(group)
    supported = int(group["has_any_baseline"].sum())
    survivors_m5 = int(group["has_any_m5_no_mw"].sum())
    survivors_both = int(group["has_any_m5_with_mw"].sum())
    supported_bins = int(group["N_bins_baseline"].sum())
    surviving_bins_m5 = int(group["N_bins_m5_no_mw"].sum())
    surviving_bins_both = int(group["N_bins_m5_with_mw"].sum())
    event_lost = {
        "m5_only": supported - survivors_m5,
        "additional_mw_given_m5": survivors_m5 - survivors_both,
        "combined": supported - survivors_both,
    }
    bin_lost = {
        "m5_only": supported_bins - surviving_bins_m5,
        "additional_mw_given_m5": surviving_bins_m5 - surviving_bins_both,
        "combined": supported_bins - surviving_bins_both,
    }
    out: dict[str, float | int] = {
        "N_injected": n, "N_events_with_baseline_color": supported,
        "N_events_mw_only": supported, "N_events_m5_no_mw": survivors_m5,
        "N_events_m5_with_mw": survivors_both,
        "N_event_bin_opportunities": supported_bins,
        "N_bins_mw_only": supported_bins,
        "N_bins_m5_no_mw": surviving_bins_m5,
        "N_bins_m5_with_mw": surviving_bins_both,
        "event_loss_mw_only": 0, "event_loss_rate_mw_only": ratio(0, supported),
        "bin_loss_mw_only": 0, "bin_loss_rate_mw_only": ratio(0, supported_bins),
        "baseline_color_fraction_all_injected": ratio(supported, n),
    }
    for key in COMPONENTS:
        out[f"event_loss_{key}"] = event_lost[key]
        out[f"event_loss_rate_{key}"] = ratio(event_lost[key], supported)
        lower, upper = wilson(event_lost[key], supported)
        out[f"event_loss_rate_{key}_ci95_low"] = lower
        out[f"event_loss_rate_{key}_ci95_high"] = upper
        out[f"event_loss_fraction_all_injected_{key}"] = ratio(event_lost[key], n)
        out[f"bin_loss_{key}"] = bin_lost[key]
        out[f"bin_loss_rate_{key}"] = ratio(bin_lost[key], supported_bins)
    out["event_mw_penalty_conditional_on_m5_survivors"] = ratio(
        event_lost["additional_mw_given_m5"], survivors_m5
    )
    out["bin_mw_penalty_conditional_on_m5_survivors"] = ratio(
        bin_lost["additional_mw_given_m5"], surviving_bins_m5
    )
    return out


def latitude_bins(events: pd.DataFrame, width: float) -> pd.DataFrame:
    if not 0 < width <= 90:
        raise ValueError("latitude-bin-width must lie in (0, 90]")
    edges = np.arange(0, 90 + width, width)
    edges[-1] = 90
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        subset = events.loc[
            (events["abs_b_deg"] >= lo) &
            (events["abs_b_deg"] <= hi if hi == 90 else events["abs_b_deg"] < hi)
        ]
        rows.append({
            "b_min_deg": float(lo), "b_max_deg": float(hi),
            "b_center_deg": float((lo + hi) / 2),
            **metrics(subset),
        })
    return pd.DataFrame(rows)


def threshold_scan(events: pd.DataFrame, thresholds: np.ndarray, minimum: int) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        low = metrics(events.loc[events["abs_b_deg"] < threshold])
        high = metrics(events.loc[events["abs_b_deg"] >= threshold])
        eligible = (low["N_events_with_baseline_color"] >= minimum
                    and high["N_events_with_baseline_color"] >= minimum)
        row: dict[str, float | int | bool] = {
            "threshold_deg": float(threshold),
            "N_low_injected": low["N_injected"],
            "N_high_injected": high["N_injected"],
            "N_low_baseline_color": low["N_events_with_baseline_color"],
            "N_high_baseline_color": high["N_events_with_baseline_color"],
            "eligible_min_events": bool(eligible),
        }
        for key in COMPONENTS:
            for level in ("event", "bin"):
                field = f"{level}_loss_rate_{key}"
                row[f"{level}_low_{key}"] = low[field]
                row[f"{level}_high_{key}"] = high[field]
                row[f"{level}_low_over_high_{key}"] = (
                    ratio(low[field], high[field]) if eligible else np.nan
                )
                row[f"{level}_low_loss_count_{key}"] = low[f"{level}_loss_{key}"]
                row[f"{level}_high_loss_count_{key}"] = high[f"{level}_loss_{key}"]
            for side, data in (("low", low), ("high", high)):
                for bound in ("low", "high"):
                    row[f"event_{side}_{key}_ci95_{bound}"] = data[
                        f"event_loss_rate_{key}_ci95_{bound}"
                    ]
            ci_low, ci_high = (
                rate_ratio_ci95(
                    low[f"event_loss_{key}"], low["N_events_with_baseline_color"],
                    high[f"event_loss_{key}"], high["N_events_with_baseline_color"],
                ) if eligible else (np.nan, np.nan)
            )
            row[f"event_low_over_high_{key}_ci95_low"] = ci_low
            row[f"event_low_over_high_{key}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def find_plateau(scan: pd.DataFrame, tolerance: float, min_thresholds: int,
                 column: str) -> dict[str, float | int | str | None]:
    valid = scan.loc[scan["eligible_min_events"]].dropna(
        subset=[column]
    ).reset_index(drop=True)
    if len(valid) < min_thresholds:
        return {"status": "insufficient_valid_thresholds", "candidate_deg": None}
    if (valid[column] == 0).all():
        return {
            "status": "no_measurable_signal",
            "candidate_deg": None,
            "reference_value": 0.0,
        }
    reference = float(valid[column].iloc[-min_thresholds:].median())
    differences = (valid[column] - reference).abs().to_numpy(float)
    for i in range(len(valid) - min_thresholds + 1):
        if np.all(differences[i:] <= tolerance):
            return {
                "status": "heuristic_candidate",
                "candidate_deg": float(valid.loc[i, "threshold_deg"]),
                "reference_value": reference,
                "absolute_tolerance": tolerance,
                "N_remaining_thresholds": int(len(valid) - i),
            }
    return {
        "status": "no_stable_plateau",
        "candidate_deg": None,
        "reference_value": reference,
        "absolute_tolerance": tolerance,
    }


def save_plots(bins: pd.DataFrame, scan: pd.DataFrame,
               plateaux: dict[str, dict[str, dict]], output: Path) -> None:
    eligible = scan.loc[scan["eligible_min_events"]]
    for level, level_label, suffix in (
        ("event", "Events losing every colour in the window", "events"),
        ("bin", "Lost event–time-bin opportunities", "event_bins"),
    ):
        fig, axes = plt.subplots(
            2, 1, figsize=(10.0, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True,
        )
        for key in COMPONENTS:
            axes[0].plot(
                bins["b_center_deg"], 100 * bins[f"{level}_loss_rate_{key}"],
                marker="o", lw=1.6, color=COLORS[key], label=LABELS[key],
            )
        axes[0].set_ylabel(level_label + " [% of baseline]")
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].grid(alpha=0.2)
        axes[1].bar(
            bins["b_center_deg"], bins["N_events_with_baseline_color"],
            width=bins["b_max_deg"] - bins["b_min_deg"],
            color="#b6bac2", edgecolor="white",
        )
        axes[1].set_ylabel("N events")
        axes[1].set_xlabel(r"Absolute Galactic latitude $|b|$ [deg]")
        axes[1].grid(axis="y", alpha=0.2)
        fig.savefig(output / f"latitude_loss_rates_in_bins_{suffix}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.0, 5.6), constrained_layout=True)
        for key in COMPONENTS:
            ax.plot(eligible["threshold_deg"],
                    eligible[f"{level}_low_over_high_{key}"],
                    marker=".", lw=1.5, color=COLORS[key], label=LABELS[key])
        ax.axhline(1, color="#666", ls=":", lw=1.0)
        ratio_plateau = plateaux[level]["ratio"]
        if ratio_plateau["status"] == "heuristic_candidate":
            ax.axvline(
                ratio_plateau["candidate_deg"], ls="--", color="#222",
                label=f"Candidate ratio plateau: {ratio_plateau['candidate_deg']:g}°",
            )
        ax.set_xlabel(r"Latitude cut $b_{\rm cut}$ [deg]: below vs above")
        ax.set_ylabel(f"{level_label}: loss rate below / above")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(output / f"latitude_low_over_high_loss_ratio_{suffix}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.0, 5.6), constrained_layout=True)
        for key in COMPONENTS:
            ax.plot(eligible["threshold_deg"], 100 * eligible[f"{level}_high_{key}"],
                    marker=".", lw=1.5, color=COLORS[key], label=LABELS[key])
            if level == "event":
                ax.fill_between(
                    eligible["threshold_deg"].to_numpy(float),
                    100 * eligible[f"event_high_{key}_ci95_low"].to_numpy(float),
                    100 * eligible[f"event_high_{key}_ci95_high"].to_numpy(float),
                    color=COLORS[key], alpha=0.10,
                )
        plateau = plateaux[level]["high_side"]
        if plateau["status"] == "heuristic_candidate":
            ax.axvline(plateau["candidate_deg"], ls="--", color="#222",
                       label=f"Candidate plateau: {plateau['candidate_deg']:g}°")
        ax.set_xlabel(r"Lower boundary $b_{\rm cut}$ of $|b|\geq b_{\rm cut}$ [deg]")
        ax.set_ylabel(f"High-latitude {level_label.lower()} [%]")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(output / f"latitude_high_side_loss_and_plateau_{suffix}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    root = args.run_dir.expanduser().resolve()
    bands = tuple(args.color_pair.lower().replace("_", "-").split("-"))
    if len(bands) != 2 or bands[0] == bands[1]:
        raise ValueError("Use a pair like g-r")
    if any(b not in (LSST_R if args.photometric_system == "lsst" else PS1_R) for b in bands):
        raise ValueError("Unknown band for selected photometric system")
    centres = time_centres(args.t_min, args.t_max, args.bin_width)
    if not 0 < args.threshold_min < args.threshold_max < 90 or args.threshold_step <= 0:
        raise ValueError("Require 0 < threshold-min < threshold-max < 90 and step > 0")
    if (args.min_events_per_side < 1 or args.plateau_min_thresholds < 1
            or args.plateau_tolerance < 0 or args.plateau_ratio_tolerance < 0):
        raise ValueError("Minimum event/threshold counts must be positive and tolerance >= 0")
    m5_path = find_m5_table(root, args.m5_quantile_table)
    m5 = read_m5(m5_path, args.threshold_scope, args.m5_scenario, bands)
    metadata, paths = load_events(root, args.photometric_system)
    name = (f"latitude_losses_{bands[0]}_{bands[1]}_{args.photometric_system}_"
            f"{args.m5_scenario}_{args.threshold_scope}_"
            f"{args.t_min:g}to{args.t_max:g}d".replace(".", "p"))
    output = (args.output_dir.expanduser().resolve() if args.output_dir
              else root / "analysis" / name)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; choose a new path or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Analysing {len(metadata):,} events across {len(centres)} bins; m5={m5}")
    events = count_event_bins(metadata, paths, bands, args.photometric_system,
                              centres, m5)
    overall = metrics(events)
    bins = latitude_bins(events, args.latitude_bin_width)
    thresholds = np.arange(args.threshold_min,
                           args.threshold_max + args.threshold_step * 1e-9,
                           args.threshold_step)
    scan = threshold_scan(events, thresholds, args.min_events_per_side)
    plateaux = {
        level: {
            "high_side": find_plateau(
                scan, args.plateau_tolerance, args.plateau_min_thresholds,
                f"{level}_high_additional_mw_given_m5",
            ),
            "ratio": find_plateau(
                scan, args.plateau_ratio_tolerance, args.plateau_min_thresholds,
                f"{level}_low_over_high_additional_mw_given_m5",
            ),
        }
        for level in ("event", "bin")
    }
    events.to_csv(output / "event_window_availability.csv", index=False)
    pd.DataFrame([overall]).to_csv(output / "whole_population_loss_summary.csv", index=False)
    bins.to_csv(output / "latitude_loss_rates_by_bin.csv", index=False)
    scan.to_csv(output / "latitude_low_high_threshold_scan.csv", index=False)
    pd.DataFrame([
        {"level": level, "metric": metric, **info}
        for level, estimates in plateaux.items()
        for metric, info in estimates.items()
    ]).to_csv(output / "latitude_plateau_candidates.csv", index=False)
    save_plots(bins, scan, plateaux, output)
    config = {
        "run_dir": str(root), "source": "saved synthetic light curves",
        "photometric_system": args.photometric_system, "color_pair": args.color_pair,
        "time_reference": "days since merger, observer frame",
        "t_min_days": args.t_min, "t_max_days": args.t_max,
        "bin_width_days": args.bin_width, "N_bin_centres": len(centres),
        "interpolation": "linear in log10 positive time, no extrapolation",
        "m5_table": str(m5_path), "m5_scenario": args.m5_scenario,
        "m5_threshold_scope": args.threshold_scope, "m5_by_band": m5,
        "thresholds_constant_across_latitude": True,
        "m5_selection_photometric_system": "lsst",
        "ps1_note": (
            "PS1 two-band colour availability is used for the denominator; "
            "the m5 cut uses the corresponding LSST bands, never PS1 magnitudes"
            if args.photometric_system == "ps1" else None
        ),
        "loss_rate_denominator_events": "events with >=1 baseline two-band colour in the whole window",
        "loss_rate_denominator_bins": "supported event-bin two-band opportunities in the whole window",
        "event_loss": "baseline-supported objects with zero surviving bins over the full window",
        "extinction_only_loss": "zero: adding A without m5 changes colour, not pair availability",
        "additional_mw_given_m5": "m5-no-MW survivors lost when A is restored",
        "plateaux": plateaux,
        "plateau_rule": (
            "For additional MW loss given m5, first cut for which the rate "
            "among |b| >= cut (or low/high ratio) and all subsequent eligible "
            "values lie within the configured tolerance of the median of the "
            "last eligible values; descriptive only, not a physical dust boundary"
        ),
        "ratio_ci95": "approximate event-level independent-group log-rate-ratio interval; undefined for zero losses",
    }
    (output / "analysis_configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Whole-window event loss: m5 only={overall['event_loss_rate_m5_only']:.3f}; "
        f"additional MW with m5={overall['event_loss_rate_additional_mw_given_m5']:.3f}; "
        f"combined={overall['event_loss_rate_combined']:.3f}"
    )
    for level in ("event", "bin"):
        for metric in ("ratio", "high_side"):
            estimate = plateaux[level][metric]
            print(f"Heuristic MW {level} {metric} plateau:",
                  estimate["candidate_deg"], estimate["status"])
    print("Output:", output)


if __name__ == "__main__":
    main()
