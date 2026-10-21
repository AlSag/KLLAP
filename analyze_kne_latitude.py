#!/usr/bin/env python3
"""Interactive, self-contained Galactic-latitude analyses for KNe populations.

The script exposes three complementary scientific products: losses inside one
all-sky or latitude-selected population, paired comparisons of disjoint
latitude bins, and stability of a moving low/high latitude cut.  The terminal
interface, calculations and figures are grouped in explicit sections below.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from kne_pipeline_common import discover_m5_quantile_table
from kne_m5 import M5_SCENARIOS

# =============================================================================
# USER-FRIENDLY TERMINAL INTERFACE AND INPUT VALIDATION
# =============================================================================

def ask(label: str, default="", cast=str):
    value = input(f"{label} [{default}]: ").strip()
    return cast(value or default)


def choose(title: str, choices, default):
    print(f"\n{title}")
    selected_default = next(i for i, (value, _) in enumerate(choices, 1) if value == default)
    for index, (value, label) in enumerate(choices, 1):
        suffix = " [défaut]" if value == default else ""
        print(f"  {index}. {label}{suffix}")
    entered = input(f"Choix [{selected_default}]: ").strip()
    if not entered:
        return default
    index = int(entered)
    if not 1 <= index <= len(choices):
        raise ValueError(f"Le choix doit être compris entre 1 et {len(choices)}")
    return choices[index - 1][0]


def yes_no(label: str, default=False) -> bool:
    marker = "O/n" if default else "o/N"
    value = input(f"{label} [{marker}]: ").strip().lower()
    if not value:
        return default
    if value not in {"o", "oui", "y", "yes", "n", "non", "no"}:
        raise ValueError("Répondre oui ou non")
    return value in {"o", "oui", "y", "yes"}


def population_default_configuration() -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=Path("."), output_dir=None, overwrite=False,
        color_pair="g-r", photometric_system="lsst", t_min=0.4,
        t_max=16.0, bin_width=0.4, m5_quantile_table=None,
        m5_scenario="median_p50", threshold_scope="common_global",
        latitude_bin_width=5.0, threshold_min=5.0, threshold_max=70.0,
        threshold_step=1.0, min_events_per_side=100,
        plateau_tolerance=0.02, plateau_ratio_tolerance=0.10,
        plateau_min_thresholds=3,
    )


def population_interactive(args: argparse.Namespace) -> argparse.Namespace:
    print("Pertes de disponibilité en couleur selon la latitude galactique")
    args.run_dir = Path(ask("Dossier du run (contenant summary.csv)", args.run_dir))
    args.color_pair = ask("Paire de couleurs", args.color_pair)
    args.photometric_system = choose(
        "Système photométrique", [("lsst", "LSST"), ("ps1", "PS1")],
        args.photometric_system,
    )
    args.t_min = ask("Début de la fenêtre [jours]", args.t_min, float)
    args.t_max = ask("Fin de la fenêtre [jours]", args.t_max, float)
    args.bin_width = ask("Largeur des bins [jours]", args.bin_width, float)
    args.m5_scenario = choose(
        "Seuil de profondeur m5",
        [("worst_p25", "p25"), ("median_p50", "p50"), ("best_p75", "p75")],
        args.m5_scenario,
    )
    explicit = ask("Table m5 (vide = recherche automatique)", "")
    args.m5_quantile_table = Path(explicit) if explicit else None
    args.threshold_scope = choose(
        "Portée du seuil m5",
        [("common_global", "Même seuil global à toutes les latitudes"),
         ("latitude_conditioned", "Seuil conditionné par la latitude")],
        args.threshold_scope,
    )
    args.latitude_bin_width = ask("Largeur des bins de latitude [deg]", args.latitude_bin_width, float)
    args.threshold_min = ask("Premier seuil low/high [deg]", args.threshold_min, float)
    args.threshold_max = ask("Dernier seuil low/high [deg]", args.threshold_max, float)
    args.threshold_step = ask("Pas des seuils [deg]", args.threshold_step, float)
    args.min_events_per_side = ask("Événements minimum de chaque côté", args.min_events_per_side, int)
    output = ask("Dossier de sortie (vide = automatique)", "")
    args.output_dir = Path(output) if output else None
    args.overwrite = yes_no("Écraser un dossier de sortie non vide", False)
    return args


def comparison_default_configuration() -> argparse.Namespace:
    return argparse.Namespace(
        bins=None, output_dir=Path("analysis/latitude_bin_comparison"),
        bootstrap_repetitions=2000, confidence_level=0.95,
        plateau_relative_tolerance=0.10, minimum_plateau_bins=3,
        seed=20260930, linear_y=False, overwrite=False,
    )


def comparison_interactive(args: argparse.Namespace) -> argparse.Namespace:
    print("Comparaison de populations appariées par tranches de latitude")
    edges = [float(value.strip()) for value in ask(
        "Bornes des tranches |b| [deg], séparées par des virgules",
        "0,5,10,15,20,25,30,40,50,90",
    ).split(",") if value.strip()]
    if len(edges) < 3:
        raise ValueError("Il faut au moins deux tranches")
    args.bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        table = ask(
            f"CSV event_window_availability pour [{low:g},{high:g}{']' if high == 90 else '['}",
            "event_window_availability.csv",
        )
        args.bins.append([str(low), str(high), table])
    args.bootstrap_repetitions = ask("Répétitions bootstrap", args.bootstrap_repetitions, int)
    args.confidence_level = float(choose(
        "Intervalle de confiance", [(0.68, "68 %"), (0.95, "95 %")],
        args.confidence_level,
    ))
    args.plateau_relative_tolerance = ask(
        "Tolérance relative du plateau", args.plateau_relative_tolerance, float
    )
    args.minimum_plateau_bins = ask(
        "Nombre minimum de tranches dans le plateau", args.minimum_plateau_bins, int
    )
    args.output_dir = Path(ask("Dossier de sortie", args.output_dir))
    args.overwrite = yes_no("Remplacer les sorties existantes", False)
    return args


def plateau_default_configuration() -> argparse.Namespace:
    return argparse.Namespace(
        event_table=None, pair=None, thresholds="5,10,15,20,25,30,40,50",
        bootstrap_mode="paired", bootstrap_repetitions=2000,
        confidence=0.95, tolerance=0.10, minimum_plateau_points=3,
        minimum_plateau_span=10.0, min_events=100, min_mw_losses=20,
        proposed_cut=20.0, seed=20261007,
        output_dir=Path("analysis/low_high_plateau"), overwrite=False,
    )


def plateau_interactive(args: argparse.Namespace) -> argparse.Namespace:
    print("Pertes MW relatives high / low et plateau en latitude")
    mode = choose(
        "Source des événements",
        [("allsky", "Une table all-sky"), ("pairs", "Runs low/high séparés")],
        "allsky",
    )
    if mode == "allsky":
        args.event_table = Path(ask(
            "event_window_availability.csv all-sky", "event_window_availability.csv"
        ))
        args.pair = None
        args.thresholds = ask("Seuils |b| [deg], séparés par des virgules", args.thresholds)
    else:
        args.event_table = None
        args.pair = []
        cuts = plateau_parse_cuts(ask("Seuils |b| [deg], séparés par des virgules", args.thresholds))
        for cut in cuts:
            low = ask(f"CSV ou dossier low |b| < {cut:g}", "event_window_availability.csv")
            high = ask(f"CSV ou dossier high |b| >= {cut:g}", "event_window_availability.csv")
            args.pair.append([str(cut), low, high])
        args.bootstrap_mode = "paired" if yes_no(
            "Mêmes KNe et mêmes event_id dans tous les runs", True
        ) else "independent"
    args.bootstrap_repetitions = ask("Répétitions bootstrap", args.bootstrap_repetitions, int)
    args.tolerance = ask("Tolérance relative du plateau", args.tolerance, float)
    args.minimum_plateau_span = ask(
        "Étendue minimale après le début du plateau [deg]", args.minimum_plateau_span, float
    )
    args.output_dir = Path(ask("Dossier de sortie", args.output_dir))
    args.overwrite = yes_no("Remplacer les deux PNG existants", False)
    return args

# =============================================================================
# POPULATION LOSS CALCULATIONS AND FIGURES
# =============================================================================

#!/usr/bin/env python3




matplotlib.use("Agg")



population_LSST_R = {
    "u": 4.757217815396922, "g": 3.6605664439892616,
    "r": 2.70136780871597, "i": 2.0536599130965882,
    "z": 1.5900964472616756, "y": 1.3077049588254708,
}
population_PS1_R = {"g": 3.172, "r": 2.271, "i": 1.682, "z": 1.322, "y": 1.087}
population_LATITUDE_COLUMNS = (
    "galactic_b_deg", "galactic_latitude_deg", "b_gal_deg", "b_deg"
)
population_COMPONENTS = ("m5_only", "additional_mw_given_m5", "combined")
population_COLORS = {
    "m5_only": "#395f9b",
    "additional_mw_given_m5": "#d28219",
    "combined": "#652f81",
}
population_LABELS = {
    "m5_only": "m5 only (no MW)",
    "additional_mw_given_m5": "Additional MW loss with m5",
    "combined": "m5 + MW (total)",
}




def population_canonical_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text))).zfill(6)
    return text


def population_time_centres(t_min: float, t_max: float, width: float) -> np.ndarray:
    if t_min <= 0 or width <= 0 or t_max <= t_min:
        raise ValueError("Require 0 < t-min < t-max and bin-width > 0")
    n = (t_max - t_min) / width
    if not np.isclose(n, round(n), atol=1e-8):
        raise ValueError("(t-max - t-min) must be an integer multiple of bin-width")
    return t_min + (np.arange(round(n)) + 0.5) * width


def population_find_m5_table(root: Path, explicit: Path | None) -> Path:
    return discover_m5_quantile_table(root, explicit=explicit)


def population_read_m5(path: Path, scope: str, scenario: str, bands: tuple[str, str]) -> dict[str, float]:
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


def population_load_events(root: Path, system: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    summary_path = root / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path, dtype={"event_id": str})
    if "event_id" not in summary or "ebv_mw" not in summary:
        raise ValueError("summary.csv requires event_id and ebv_mw")
    latitude = next((column for column in population_LATITUDE_COLUMNS if column in summary), None)
    if latitude is None:
        raise ValueError(
            "summary.csv has no Galactic latitude column; an all-latitude run "
            "with saved galactic_latitude_deg (or alias) is required"
        )
    summary["event_id"] = summary["event_id"].map(population_canonical_id)
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


def population_iter_curves(paths: tuple[str, ...], bands: tuple[str, str], system: str):
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
        shard["event_id"] = shard["event_id"].map(population_canonical_id)
        shard["band"] = shard["band"].astype(str).str.strip().str.lower()
        shard["photometric_system"] = (
            shard["photometric_system"].astype(str).str.strip().str.lower()
        )
        shard["t_days"] = pd.to_numeric(shard["t_days"], errors="coerce")
        shard["mag"] = pd.to_numeric(shard["mag"], errors="coerce")
        for event_id, curve in shard.groupby("event_id", sort=False):
            yield event_id, curve


def population_interpolate(curve: pd.DataFrame, band: str, centres: np.ndarray) -> np.ndarray:
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


def population_extinction(row: pd.Series, system: str, band: str) -> float:
    column = f"A_{system}_{band}_mw"
    if column in row and np.isfinite(pd.to_numeric(row[column], errors="coerce")):
        a = float(row[column])
    else:
        a = float(row["ebv_mw"]) * (
            population_LSST_R if system == "lsst" else population_PS1_R
        )[band]
    if a < 0 or not np.isfinite(a):
        raise ValueError(f"Invalid Galactic extinction for {row['event_id']} {band}")
    return a


def population_count_event_bins(
    metadata: pd.DataFrame, paths: tuple[str, ...],
    bands: tuple[str, str], system: str, centres: np.ndarray, m5: dict[str, float],
) -> pd.DataFrame:
    records = {event_id: {"N_bins_baseline": 0, "N_bins_mw_only": 0,
                          "N_bins_m5_no_mw": 0, "N_bins_m5_with_mw": 0}
               for event_id in metadata.index}
    seen: set[str] = set()
    seen_lsst_rows = False
    for event_id, curve in population_iter_curves(paths, bands, system):
        if event_id not in records:
            continue
        if event_id in seen:
            raise ValueError(f"Duplicate saved rows for event {event_id} in multiple files")
        seen.add(event_id)
        row = metadata.loc[event_id]
        colour_curve = curve.loc[curve["photometric_system"].eq(system)]
        selection_curve = curve.loc[curve["photometric_system"].eq("lsst")]
        seen_lsst_rows |= not selection_curve.empty
        colour_saved = [population_interpolate(colour_curve, band, centres) for band in bands]
        selection_saved = [population_interpolate(selection_curve, band, centres) for band in bands]
        selection_intrinsic = [
            mag - population_extinction(row, "lsst", band)
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


def population_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def population_wilson(success: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = success / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z / denom * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2))
    return float(centre - spread), float(centre + spread)


def population_rate_ratio_ci95(lost_low: int, total_low: int, lost_high: int,
                    total_high: int) -> tuple[float, float]:
    """Approximate independent-event log-rate-ratio CI; undefined at zero losses."""
    if min(lost_low, total_low, lost_high, total_high) <= 0:
        return float("nan"), float("nan")
    rr = (lost_low / total_low) / (lost_high / total_high)
    se = np.sqrt(
        1 / lost_low - 1 / total_low + 1 / lost_high - 1 / total_high
    )
    return float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))


def population_metrics(group: pd.DataFrame) -> dict[str, float | int]:
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
        "event_loss_mw_only": 0, "event_loss_rate_mw_only": population_ratio(0, supported),
        "bin_loss_mw_only": 0, "bin_loss_rate_mw_only": population_ratio(0, supported_bins),
        "baseline_color_fraction_all_injected": population_ratio(supported, n),
    }
    for key in population_COMPONENTS:
        out[f"event_loss_{key}"] = event_lost[key]
        out[f"event_loss_rate_{key}"] = population_ratio(event_lost[key], supported)
        lower, upper = population_wilson(event_lost[key], supported)
        out[f"event_loss_rate_{key}_ci95_low"] = lower
        out[f"event_loss_rate_{key}_ci95_high"] = upper
        out[f"event_loss_fraction_all_injected_{key}"] = population_ratio(event_lost[key], n)
        out[f"bin_loss_{key}"] = bin_lost[key]
        out[f"bin_loss_rate_{key}"] = population_ratio(bin_lost[key], supported_bins)
    out["event_mw_penalty_conditional_on_m5_survivors"] = population_ratio(
        event_lost["additional_mw_given_m5"], survivors_m5
    )
    out["bin_mw_penalty_conditional_on_m5_survivors"] = population_ratio(
        bin_lost["additional_mw_given_m5"], surviving_bins_m5
    )
    return out


def population_latitude_bins(events: pd.DataFrame, width: float) -> pd.DataFrame:
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
            **population_metrics(subset),
        })
    return pd.DataFrame(rows)


def population_threshold_scan(events: pd.DataFrame, thresholds: np.ndarray, minimum: int) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        low = population_metrics(events.loc[events["abs_b_deg"] < threshold])
        high = population_metrics(events.loc[events["abs_b_deg"] >= threshold])
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
        for key in population_COMPONENTS:
            for level in ("event", "bin"):
                field = f"{level}_loss_rate_{key}"
                row[f"{level}_low_{key}"] = low[field]
                row[f"{level}_high_{key}"] = high[field]
                row[f"{level}_low_over_high_{key}"] = (
                    population_ratio(low[field], high[field]) if eligible else np.nan
                )
                row[f"{level}_low_loss_count_{key}"] = low[f"{level}_loss_{key}"]
                row[f"{level}_high_loss_count_{key}"] = high[f"{level}_loss_{key}"]
            for side, data in (("low", low), ("high", high)):
                for bound in ("low", "high"):
                    row[f"event_{side}_{key}_ci95_{bound}"] = data[
                        f"event_loss_rate_{key}_ci95_{bound}"
                    ]
            ci_low, ci_high = (
                population_rate_ratio_ci95(
                    low[f"event_loss_{key}"], low["N_events_with_baseline_color"],
                    high[f"event_loss_{key}"], high["N_events_with_baseline_color"],
                ) if eligible else (np.nan, np.nan)
            )
            row[f"event_low_over_high_{key}_ci95_low"] = ci_low
            row[f"event_low_over_high_{key}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def population_find_plateau(scan: pd.DataFrame, tolerance: float, min_thresholds: int,
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


def population_save_plots(bins: pd.DataFrame, scan: pd.DataFrame,
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
        for key in population_COMPONENTS:
            axes[0].plot(
                bins["b_center_deg"], 100 * bins[f"{level}_loss_rate_{key}"],
                marker="o", lw=1.6, color=population_COLORS[key], label=population_LABELS[key],
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
        for key in population_COMPONENTS:
            ax.plot(eligible["threshold_deg"],
                    eligible[f"{level}_low_over_high_{key}"],
                    marker=".", lw=1.5, color=population_COLORS[key], label=population_LABELS[key])
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
        for key in population_COMPONENTS:
            ax.plot(eligible["threshold_deg"], 100 * eligible[f"{level}_high_{key}"],
                    marker=".", lw=1.5, color=population_COLORS[key], label=population_LABELS[key])
            if level == "event":
                ax.fill_between(
                    eligible["threshold_deg"].to_numpy(float),
                    100 * eligible[f"event_high_{key}_ci95_low"].to_numpy(float),
                    100 * eligible[f"event_high_{key}_ci95_high"].to_numpy(float),
                    color=population_COLORS[key], alpha=0.10,
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

def run_population_analysis(args: argparse.Namespace) -> Path:
    root = args.run_dir.expanduser().resolve()
    bands = tuple(args.color_pair.lower().replace("_", "-").split("-"))
    if len(bands) != 2 or bands[0] == bands[1]:
        raise ValueError("Use a pair like g-r")
    if any(b not in (population_LSST_R if args.photometric_system == "lsst" else population_PS1_R) for b in bands):
        raise ValueError("Unknown band for selected photometric system")
    centres = population_time_centres(args.t_min, args.t_max, args.bin_width)
    if not 0 < args.threshold_min < args.threshold_max < 90 or args.threshold_step <= 0:
        raise ValueError("Require 0 < threshold-min < threshold-max < 90 and step > 0")
    if (args.min_events_per_side < 1 or args.plateau_min_thresholds < 1
            or args.plateau_tolerance < 0 or args.plateau_ratio_tolerance < 0):
        raise ValueError("Minimum event/threshold counts must be positive and tolerance >= 0")
    m5_path = population_find_m5_table(root, args.m5_quantile_table)
    m5 = population_read_m5(m5_path, args.threshold_scope, args.m5_scenario, bands)
    metadata, paths = population_load_events(root, args.photometric_system)
    name = (f"latitude_losses_{bands[0]}_{bands[1]}_{args.photometric_system}_"
            f"{args.m5_scenario}_{args.threshold_scope}_"
            f"{args.t_min:g}to{args.t_max:g}d".replace(".", "p"))
    output = (args.output_dir.expanduser().resolve() if args.output_dir
              else root / "analysis" / name)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; choose a new path or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Analysing {len(metadata):,} events across {len(centres)} bins; m5={m5}")
    events = population_count_event_bins(metadata, paths, bands, args.photometric_system,
                              centres, m5)
    overall = population_metrics(events)
    bins = population_latitude_bins(events, args.latitude_bin_width)
    thresholds = np.arange(args.threshold_min,
                           args.threshold_max + args.threshold_step * 1e-9,
                           args.threshold_step)
    scan = population_threshold_scan(events, thresholds, args.min_events_per_side)
    plateaux = {
        level: {
            "high_side": population_find_plateau(
                scan, args.plateau_tolerance, args.plateau_min_thresholds,
                f"{level}_high_additional_mw_given_m5",
            ),
            "ratio": population_find_plateau(
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
    population_save_plots(bins, scan, plateaux, output)
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
    return output

# =============================================================================
# PAIRED DISJOINT-BIN CALCULATIONS AND FIGURES
# =============================================================================

#!/usr/bin/env python3





comparison_REQUIRED_COLUMNS = {
    "event_id",
    "abs_b_deg",
    "N_bins_baseline",
    "N_bins_m5_no_mw",
    "N_bins_m5_with_mw",
}
comparison_COMPONENTS = ("m5_only", "mw_additional", "combined")
comparison_COLORS = {
    "m5_only": "#3568a8",
    "mw_additional": "#d97904",
    "combined": "#6b2e8c",
}
comparison_LABELS = {
    "m5_only": "m5 only (no MW; control)",
    "mw_additional": "Additional MW loss with m5",
    "combined": "m5 + MW (total)",
}


@dataclass(frozen=True)
class comparison_LatitudeBin:
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




def comparison_load_bin(lower_text: str, upper_text: str, path_text: str) -> comparison_LatitudeBin:
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
    missing = comparison_REQUIRED_COLUMNS - set(table.columns)
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
    return comparison_LatitudeBin(lower, upper, path, outcomes)


def comparison_parse_bins(raw: list[list[str]] | None) -> list[comparison_LatitudeBin]:
    if not raw or len(raw) < 4:
        raise ValueError("Provide at least four disjoint --bin inputs")
    bins = sorted((comparison_load_bin(*item) for item in raw), key=lambda item: item.lower)
    for previous, current in zip(bins[:-1], bins[1:]):
        if not np.isclose(previous.upper, current.lower):
            raise ValueError(
                "Latitude bins must be contiguous and disjoint: "
                f"{previous.label} followed by {current.label}"
            )
    return bins


def comparison_align_outcomes(bins: list[comparison_LatitudeBin]) -> np.ndarray:
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


def comparison_counts_from_outcomes(outcomes: np.ndarray) -> np.ndarray:
    return outcomes.sum(axis=1).astype(int)


def comparison_rates_and_ratios(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def comparison_plateau_threshold(
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


def comparison_paired_bootstrap(
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
        _, repetition_ratios = comparison_rates_and_ratios(counts)
        ratios[repetition] = repetition_ratios
        thresholds[repetition] = comparison_plateau_threshold(
            lower_edges,
            repetition_ratios[:, 1],
            tolerance,
            minimum_bins,
        )[0]
    return ratios, thresholds


def comparison_confidence_bounds(samples: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - confidence) / 2.0
    with np.errstate(all="ignore"):
        return (
            np.nanquantile(samples, alpha, axis=0),
            np.nanquantile(samples, 1.0 - alpha, axis=0),
        )


def comparison_threshold_summary(samples: np.ndarray, confidence: float) -> dict[str, float]:
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


def comparison_interval_x_errors(bins: list[comparison_LatitudeBin]) -> np.ndarray:
    centres = np.array([item.centre for item in bins], dtype=float)
    lower = centres - np.array([item.lower for item in bins], dtype=float)
    upper = np.array([item.upper for item in bins], dtype=float) - centres
    return np.vstack([lower, upper])


def comparison_save_plot(
    bins: list[comparison_LatitudeBin],
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
    xerr = comparison_interval_x_errors(bins)
    fig, ax = plt.subplots(figsize=(10.8, 6.5), constrained_layout=True)
    for component_index, component in enumerate(comparison_COMPONENTS):
        y = ratios[:, component_index]
        lower = ci_low[:, component_index]
        upper = ci_high[:, component_index]
        if not linear_y:
            y = np.where(y > 0, y, np.nan)
            lower = np.where(lower > 0, lower, np.nan)
            upper = np.where(upper > 0, upper, np.nan)
        ax.fill_between(centres, lower, upper, color=comparison_COLORS[component], alpha=0.14)
        ax.errorbar(
            centres,
            y,
            xerr=xerr,
            fmt="o-",
            ms=4.8,
            lw=1.8,
            capsize=2.5,
            color=comparison_COLORS[component],
            label=comparison_LABELS[component],
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


def comparison_ratio_text(value: float, low: float, high: float) -> str:
    if not np.isfinite(value):
        return "undefined"
    return f"{100*value:.2f} [{100*low:.2f}, {100*high:.2f}]"


def comparison_save_table(
    bins: list[comparison_LatitudeBin],
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
                comparison_ratio_text(mw_ratio, ci_low[index, 1], ci_high[index, 1]),
                f"{reduction:.2f}" if np.isfinite(reduction) else "undefined",
                comparison_ratio_text(ratios[index, 0], ci_low[index, 0], ci_high[index, 0]),
                comparison_ratio_text(ratios[index, 2], ci_low[index, 2], ci_high[index, 2]),
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

def run_bin_comparison(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    if not 0 < args.plateau_relative_tolerance < 1:
        raise ValueError("--plateau-relative-tolerance must lie in (0, 1)")
    if args.minimum_plateau_bins < 2:
        raise ValueError("--minimum-plateau-bins must be at least 2")
    bins = comparison_parse_bins(args.bins)
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

    outcomes = comparison_align_outcomes(bins)
    counts = comparison_counts_from_outcomes(outcomes)
    rates, ratios = comparison_rates_and_ratios(counts)
    if not np.isfinite(ratios[:, 1]).all():
        raise ValueError("MW reference loss rate is zero; relative MW ratios are undefined")
    bootstrap_ratios, bootstrap_thresholds = comparison_paired_bootstrap(
        outcomes,
        args.bootstrap_repetitions,
        args.seed,
        np.array([item.lower for item in bins]),
        args.plateau_relative_tolerance,
        args.minimum_plateau_bins,
    )
    ci_low, ci_high = comparison_confidence_bounds(bootstrap_ratios, args.confidence_level)
    observed_plateau = comparison_plateau_threshold(
        np.array([item.lower for item in bins]),
        ratios[:, 1],
        args.plateau_relative_tolerance,
        args.minimum_plateau_bins,
    )
    bootstrap_plateau = comparison_threshold_summary(
        bootstrap_thresholds, args.confidence_level
    )
    comparison_save_plot(
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
    comparison_save_table(
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
    return plot_path, table_path

# =============================================================================
# MOVING LOW/HIGH CUT AND PLATEAU CALCULATIONS
# =============================================================================

#!/usr/bin/env python3


matplotlib.use("Agg")


plateau_REQUIRED = ["event_id", "abs_b_deg", "N_bins_baseline",
            "N_bins_m5_no_mw", "N_bins_m5_with_mw"]
plateau_CONFIG_KEYS = ["photometric_system", "color_pair", "t_min_days", "t_max_days",
               "bin_width_days", "m5_scenario", "m5_threshold_scope", "m5_by_band"]


def plateau_resolve_table(text):
    path = Path(text).expanduser().resolve()
    if path.is_file():
        return path
    candidates = sorted(path.rglob("event_window_availability.csv")) if path.is_dir() else []
    if len(candidates) != 1:
        raise ValueError(f"Indiquer un CSV exact : {len(candidates)} fichiers trouves sous {path}")
    return candidates[0]


def plateau_load_table(text):
    path = plateau_resolve_table(text)
    df = pd.read_csv(path, dtype={"event_id": str})
    missing = set(plateau_REQUIRED) - set(df)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing)}")
    if df.event_id.isna().any() or df.event_id.duplicated().any():
        raise ValueError(f"event_id doit etre unique et non vide dans {path}")
    for col in plateau_REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    values = df[plateau_REQUIRED[1:]].to_numpy(float)
    if not np.isfinite(values).all() or not df.abs_b_deg.between(0, 90).all():
        raise ValueError(f"Latitudes ou comptages invalides dans {path}")
    bins = df[plateau_REQUIRED[2:]].to_numpy(float)
    if (bins < 0).any() or not np.equal(bins, np.floor(bins)).all():
        raise ValueError(f"Les nombres de bins doivent etre des entiers positifs ou nuls : {path}")
    if not ((bins[:, 2] <= bins[:, 1]) & (bins[:, 1] <= bins[:, 0])).all():
        raise ValueError(f"Attendu N_with_MW <= N_m5 <= N_baseline : {path}")
    baseline = bins[:, 0] > 0
    m5_loss = baseline & (bins[:, 1] == 0)
    mw_loss = baseline & (bins[:, 1] > 0) & (bins[:, 2] == 0)
    # Columns: baseline denominator, losses from m5, additional MW losses.
    outcomes = np.column_stack([baseline, m5_loss, mw_loss]).astype(int)
    config_path = path.parent / "analysis_configuration.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return df, outcomes, config


def plateau_consistent_configs(configs):
    for key in plateau_CONFIG_KEYS:
        values = [json.dumps(c[key], sort_keys=True) for c in configs if key in c]
        if len(set(values)) > 1:
            raise ValueError(f"Analyses incompatibles : le parametre {key} differe entre les fichiers")


def plateau_high_over_low(counts):
    """Counts shape (..., thresholds, low/high, [baseline, m5 loss, MW loss])."""
    n = counts[..., 0].astype(float)
    losses = counts[..., 2].astype(float)
    rates = np.divide(losses, n, out=np.full_like(losses, np.nan), where=n > 0)
    return np.divide(rates[..., 1], rates[..., 0],
                     out=np.full_like(rates[..., 0], np.nan), where=rates[..., 0] > 0)


def plateau_allsky_counts(text, cuts, repetitions, rng):
    df, outcomes, config = plateau_load_table(text)
    valid = outcomes[:, 0].astype(bool)
    removed = int((~valid).sum())
    latitude = df.abs_b_deg.to_numpy()[valid]
    outcomes = outcomes[valid]
    if len(outcomes) == 0:
        raise ValueError("Aucune KNe avec couleur de reference")
    # Aggregating into latitude x outcome categories is exactly an event
    # bootstrap for these statistics and preserves overlap across all cuts.
    latitude_cell = np.searchsorted(cuts, latitude, side="right")
    outcome_cell = np.where(outcomes[:, 1] > 0, 0,
                            np.where(outcomes[:, 2] > 0, 1, 2))
    cells = np.bincount(latitude_cell * 3 + outcome_cell,
                        minlength=(len(cuts) + 1) * 3)

    def convert(draws):
        grouped = draws.reshape(*draws.shape[:-1], len(cuts) + 1, 3)
        lower = np.cumsum(grouped, axis=-2)[..., :-1, :]
        higher = grouped.sum(axis=-2)[..., None, :] - lower
        sides = np.stack([lower, higher], axis=-2)
        return np.stack([sides.sum(axis=-1), sides[..., 0], sides[..., 1]], axis=-1)

    observed = convert(cells)
    draws = rng.multinomial(int(cells.sum()), cells / cells.sum(), size=repetitions)
    return observed, convert(draws), [config], f"Bootstrap evenements commun a tous les seuils; {removed} sans couleur exclus"


def plateau_pair_counts(raw_pairs, repetitions, rng, mode):
    pairs = sorted(raw_pairs, key=lambda row: float(row[0]))
    cuts = np.array([float(row[0]) for row in pairs])
    runs, configs = [], []
    for cut, low_path, high_path in pairs:
        sides = []
        for side, text in enumerate([low_path, high_path]):
            df, out, config = plateau_load_table(text)
            mask = df.abs_b_deg < float(cut) if side == 0 else df.abs_b_deg >= float(cut)
            if not mask.all():
                raise ValueError(f"Des evenements sont du mauvais cote du seuil {cut} : {text}")
            sides.append((df.event_id.tolist(), out))
            configs.append(config)
        runs.append(sides)
    plateau_consistent_configs(configs)
    observed = np.array([[out.sum(axis=0) for _, out in pair] for pair in runs])
    boot = np.zeros((repetitions, len(cuts), 2, 3), dtype=int)
    if mode == "paired":
        ids = runs[0][0][0]
        reference = set(ids)
        aligned = []
        for pair in runs:
            aligned_pair = []
            for side_ids, out in pair:
                if set(side_ids) != reference:
                    raise ValueError("Mode apparie : memes event_id requis dans tous les runs")
                order = {identifier: i for i, identifier in enumerate(side_ids)}
                aligned_pair.append(out[[order[identifier] for identifier in ids]])
            aligned.append(aligned_pair)
        for start in range(0, repetitions, 50):
            stop = min(repetitions, start + 50)
            indices = rng.integers(len(ids), size=(stop - start, len(ids)))
            for j, pair in enumerate(aligned):
                for side, out in enumerate(pair):
                    boot[start:stop, j, side] = out[indices].sum(axis=1)
        note = "Bootstrap apparie; event_id doit identifier la meme KNe synthetique dans tous les runs"
    else:
        for j, pair in enumerate(runs):
            for side, (_, out) in enumerate(pair):
                patterns, freq = np.unique(out, axis=0, return_counts=True)
                draws = rng.multinomial(len(out), freq / freq.sum(), size=repetitions)
                boot[:, j, side] = draws @ patterns
        note = "Bootstrap independant; les runs doivent etre des simulations independantes"
    return cuts, observed, boot, configs, note


def plateau_tail_deviation(q):
    """Maximum relative distance to the median of a whole candidate tail."""
    q = np.asarray(q, float)
    median = np.median(q, axis=-1)
    valid = np.isfinite(q).all(axis=-1) & (q >= 0).all(axis=-1) & (median > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.max(np.abs(q / median[..., None] - 1), axis=-1)
    return np.where(valid, deviation, np.inf)


def plateau_assess_plateau(cuts, q, q_boot, counts, args):
    reliable = ((counts[..., 0] >= args.min_events).all(axis=1)
                & (counts[..., 2] >= args.min_mw_losses).all(axis=1)
                & np.isfinite(q))
    rows = []
    selected = None
    for i, cut in enumerate(cuts):
        enough = (len(cuts) - i >= args.minimum_plateau_points
                  and cuts[-1] - cut >= args.minimum_plateau_span)
        if not enough:
            rows.append(("Etendue insuffisante", np.nan, np.nan))
            continue
        if not reliable[i:].all():
            bad = i + int(np.flatnonzero(~reliable[i:])[0])
            if (counts[bad, :, 0] < args.min_events).any():
                reason = f"Effectif faible a {cuts[bad]:g} deg"
            elif (counts[bad, :, 2] < args.min_mw_losses).any():
                reason = f"Pertes rares a {cuts[bad]:g} deg"
            else:
                reason = f"Rapport indefini a {cuts[bad]:g} deg"
            rows.append((reason, np.nan, np.nan))
            continue
        observed = float(plateau_tail_deviation(q[i:]))
        deviations = plateau_tail_deviation(q_boot[:, i:])
        # A one-sided percentile bound on this maximum uses the entire tail,
        # rather than declaring stability from overlapping pointwise intervals.
        ordered = np.sort(deviations)
        upper = float(ordered[max(0, int(np.ceil(args.confidence * len(ordered))) - 1)])
        accepted = observed <= args.tolerance and upper <= args.tolerance
        rows.append(("Plateau compatible" if accepted else "Non confirme", observed, upper))
        if accepted and selected is None:
            selected = i
    return selected, rows


def plateau_bootstrap_ci(q_boot, confidence):
    alpha = (1 - confidence) / 2
    lower, upper = [], []
    for column in q_boot.T:
        finite = column[np.isfinite(column)]
        bounds = np.quantile(finite, [alpha, 1-alpha]) if len(finite) else [np.nan, np.nan]
        lower.append(bounds[0]); upper.append(bounds[1])
    return np.array(lower), np.array(upper)


def plateau_summary_text(cuts, selected, args):
    if selected is None:
        return f"Plateau non demontre sur {cuts[0]:g}-{cuts[-1]:g} deg avec ce critere"
    return (f"Premier plateau operationnel compatible : {cuts[selected]:g} deg "
            f"(teste jusqu'a {cuts[-1]:g} deg)")


def plateau_save_outputs(cuts, counts, q, lo, hi, selected, status, configs, note, args):
    output = args.output_dir.expanduser().resolve()
    plot = output / "kne_low_high_mw_plateau.png"
    table_path = output / "kne_low_high_mw_values.png"
    if not args.overwrite and any(p.exists() for p in [plot, table_path]):
        raise FileExistsError("Les sorties existent. Choisir un autre dossier ou ajouter --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    config = next((c for c in configs if c), {})
    details = " | ".join(str(config.get(k, "")) for k in
                          ["color_pair", "photometric_system", "m5_scenario"] if config.get(k))
    if "t_min_days" in config and "t_max_days" in config:
        details += f" | {config['t_min_days']:g}-{config['t_max_days']:g} j"
    summary = plateau_summary_text(cuts, selected, args)
    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.25, top=0.96)
    ax.fill_between(cuts, 100*lo, 100*hi, color="#e08616", alpha=.20,
                    label=f"Intervalle bootstrap {args.confidence:.0%} (point par point)")
    ax.plot(cuts, 100*q, "o-", color="#b96007", lw=2.5,
            label="Pertes MW : taux high / taux low")
    if cuts[0] <= args.proposed_cut <= cuts[-1]:
        ax.axvline(args.proposed_cut, color="#4e5966", ls="--", lw=1.6,
                   label=f"Coupure envisagee : {args.proposed_cut:g} deg")
    if selected is not None:
        plateau = np.median(q[selected:]) * 100
        ax.fill_between([cuts[selected], cuts[-1]],
                        plateau*(1-args.tolerance), plateau*(1+args.tolerance),
                        color="#168066", alpha=.16, label="Zone de tolerance du plateau")
        ax.axvline(cuts[selected], color="#168066", lw=2,
                   label=f"Plateau operationnel : {cuts[selected]:g} deg")
    if np.isfinite(hi).any() and np.nanmax(hi) >= .8:
        ax.axhline(100, color="gray", lw=1, ls=":")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Seuil $b_{\rm cut}$ [deg] : low $|b|<b_{\rm cut}$ ; high $|b|\geq b_{\rm cut}$")
    ax.set_ylabel("Taux de perte MW high / low [%]")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=9)
    fig.text(.12, .16, summary, fontsize=12, weight="bold", color="#168066" if selected is not None else "#4e5966")
    fig.text(.12, .115, "Q = 10 % : le taux de perte high vaut 10 % du taux low (90 % de moins).", fontsize=9)
    fig.text(.12, .075, f"Plateau : tous les seuils suivants ; tolerance {args.tolerance:.0%} ; borne bootstrap {args.confidence:.0%}.", fontsize=9)
    fig.text(.12, .035, details or "Perte MW supplementaire apres m5 ; denominateur = KNe avec couleur sans coupure.", fontsize=9, color="#555555")
    fig.savefig(plot, dpi=200)
    plt.close(fig)

    def ratio(value):
        return f"{100*value:.2f}" if np.isfinite(value) else "indefini"

    rows = []
    for i, cut in enumerate(cuts):
        nl, nh = counts[i, :, 0]
        ll, lh = counts[i, :, 2]
        fl = ll/nl if nl else np.nan
        fh = lh/nh if nh else np.nan
        rows.append([f"{cut:g}", f"{ll:,}/{nl:,}", ratio(fl),
                     f"{lh:,}/{nh:,}", ratio(fh),
                     f"{ratio(q[i])} [{ratio(lo[i])}, {ratio(hi[i])}]",
                     ratio(1-q[i]), status[i][0]])
    fig, ax = plt.subplots(figsize=(16, max(5.4, 2.6 + .43*len(rows))))
    ax.axis("off")
    cols = ["Seuil [deg]", "Pertes MW low / N low", "Taux low [%]",
            "Pertes MW high / N high", "Taux high [%]",
            f"Q high/low [%], IC {args.confidence:.0%}", "Baisse 1-Q [%]", "Plateau a partir d'ici"]
    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   colWidths=[.065,.15,.09,.15,.09,.18,.095,.18], bbox=[0,.28,1,.70])
    tab.auto_set_font_size(False); tab.set_fontsize(8.5)
    for (r, _), cell in tab.get_celld().items():
        cell.set_edgecolor("#d5dce5"); cell.set_linewidth(.5)
        if r == 0:
            cell.set_facecolor("#2e425b"); cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#dcefe7" if selected == r-1 else ("#f2f5f8" if r%2 == 0 else "white"))
    fig.text(.06, .21, summary, fontsize=12, weight="bold")
    fig.text(.06, .16, "Une perte MW = couleur possible apres m5 sans poussiere, aucun bin couleur restant apres ajout de MW.", fontsize=9)
    fig.text(.06, .12, "N = KNe avec au moins un bin couleur avant m5 et MW. Les pertes deja dues a m5 ne sont pas comptees comme pertes MW.", fontsize=9)
    fig.text(.06, .08, "Q = taux high / taux low. Les intervalles sont point par point ; le test de plateau porte sur toute la suite des seuils.", fontsize=9)
    fig.text(.06, .04, details or note, fontsize=8.5, color="#555555")
    fig.savefig(table_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return plot, table_path


def plateau_parse_cuts(text):
    cuts = np.array(sorted(set(float(x.strip()) for x in text.split(","))))
    if len(cuts) < 3 or not np.isfinite(cuts).all() or not ((cuts > 0) & (cuts < 90)).all():
        raise ValueError("Donner au moins 3 seuils distincts dans ]0,90[ separes par des virgules")
    return cuts

def run_plateau_analysis(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.event_table is None and not args.pair:
        raise ValueError("Indiquer une table all-sky ou plusieurs paires low/high")
    if args.bootstrap_repetitions < 100 or not 0 < args.tolerance < 1:
        raise ValueError("Au moins 100 bootstraps et une tolérance dans ]0,1[ sont requis")
    if args.min_events < 1 or args.min_mw_losses < 1 or args.minimum_plateau_points < 3 or args.minimum_plateau_span <= 0:
        raise ValueError("Effectifs minimum positifs, au moins 3 points et une étendue positive requis")
    rng = np.random.default_rng(args.seed)
    if args.event_table is not None:
        cuts = plateau_parse_cuts(args.thresholds)
        counts, boot, configs, note = plateau_allsky_counts(args.event_table, cuts, args.bootstrap_repetitions, rng)
    else:
        cuts, counts, boot, configs, note = plateau_pair_counts(args.pair, args.bootstrap_repetitions, rng, args.bootstrap_mode)
        if len(cuts) < 3 or len(set(cuts)) != len(cuts) or not ((cuts > 0) & (cuts < 90)).all():
            raise ValueError("Donner au moins 3 seuils distincts dans ]0,90[")
    q = plateau_high_over_low(counts); q_boot = plateau_high_over_low(boot)
    lo, hi = plateau_bootstrap_ci(q_boot, args.confidence)
    selected, status = plateau_assess_plateau(cuts, q, q_boot, counts, args)
    plot, table = plateau_save_outputs(cuts, counts, q, lo, hi, selected, status, configs, note, args)
    print(plateau_summary_text(cuts, selected, args)); print(note)
    for i, cut in enumerate(cuts):
        print(f"{cut:g} deg : low {counts[i,0,2]}/{counts[i,0,0]}, high {counts[i,1,2]}/{counts[i,1,0]}, Q={q[i]:.4g} ; {status[i][0]}")
    print(f"Plot : {plot}"); print(f"Tableau : {table}")
    return plot, table

# Compatibility exports for tests and notebooks that import the scientific
# functions directly.  They are implemented above in this same public script.
allsky_counts = plateau_allsky_counts
assess_plateau = plateau_assess_plateau
high_over_low = plateau_high_over_low
load_table = plateau_load_table
pair_counts = plateau_pair_counts


def main() -> None:
    mode = choose(
        "Analyse de latitude galactique",
        [
            ("population", "Pertes dans une population"),
            ("compare", "Comparaison de tranches appariées"),
            ("plateau", "Stabilité d'une coupure low/high"),
        ],
        "population",
    )
    if mode == "population":
        run_population_analysis(population_interactive(population_default_configuration()))
    elif mode == "compare":
        run_bin_comparison(comparison_interactive(comparison_default_configuration()))
    else:
        run_plateau_analysis(plateau_interactive(plateau_default_configuration()))


if __name__ == "__main__":
    main()
