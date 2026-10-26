#!/usr/bin/env python3
"""Single-load convergence analysis for KNe colour envelopes.

This self-contained workflow reads the saved population once, constructs
reproducible nested subsets and optionally evaluates bootstrap robustness for
every requested population size.  It contains three explicitly labelled scientific
paths: fixed-percentile shape/extent convergence, the former adaptive-envelope
comparison, and the legacy last-valid-bin analysis.  Calculation, aggregation,
figure generation, terminal questions and orchestration are kept in this file.

Run ``python analyze_envelope_population_convergence_fast.py`` and answer the
terminal questions.  Programmatic workflows can start from
``default_configuration()`` and call the numerical functions directly; no
hidden command-line options or subprocess configuration files are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kne_pipeline_common import prompt_choice, prompt_text

import build_color_envelopes as builder
import kne_envelope_core as envelope_core
from kne_population_io import SyntheticPopulationSource, prepare_metadata
from dynamic_color_cut_common import (
    assign_event_splits,
    canonical_event_id,
    interpolate_band,
    parse_floats,
)


# ============================================================================
# SHARED CONVERGENCE PARSING, AGGREGATION AND PLOTTING
# ============================================================================

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kne_envelope_core import M5_STAGES, STAGES
from kne_m5 import M5_SCENARIOS


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


def percentile_diagnostics(
    values: np.ndarray,
    lower_grid: list[float],
    bootstrap: bool,
    n_boot: int,
    bootstrap_threshold: float,
    minimum_tail_events: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    """Validate percentiles with optional event bootstrap.

    With bootstrap enabled, this delegates to the canonical envelope helper.
    Without bootstrap, no resampling is performed: percentile validity is
    determined only from the expected support in the nearest tail.  The NaN
    bootstrap columns are retained so output tables keep the same schema.
    """
    if bootstrap:
        diagnostics, selected = envelope_core.percentile_diagnostics(
            values,
            lower_grid,
            n_boot,
            bootstrap_threshold,
            minimum_tail_events,
            rng,
        )
        diagnostics = diagnostics.copy()
        diagnostics["bootstrap_enabled"] = True
        return diagnostics, selected

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lower = sorted(set(float(q) for q in lower_grid if 0.0 < q <= 50.0))
    percentiles = sorted(set(lower + [100.0 - q for q in lower] + [50.0]))
    estimates = np.percentile(values, percentiles)
    n_values = len(values)
    rows = []
    for q, value in zip(percentiles, estimates):
        expected_tail = n_values * min(q, 100.0 - q) / 100.0
        tail_valid = bool(expected_tail >= minimum_tail_events)
        rows.append({
            "percentile": q,
            "value": float(value),
            "bootstrap_p16": np.nan,
            "bootstrap_p84": np.nan,
            "bootstrap_half_width_68": np.nan,
            "expected_nearest_tail_events": expected_tail,
            # True means that this disabled criterion cannot reject the row.
            "bootstrap_width_valid": True,
            "tail_support_valid": tail_valid,
            "percentile_valid": tail_valid,
            "bootstrap_enabled": False,
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

    q_low = min(valid_pairs)
    q_high = 100.0 - q_low
    selected = {
        "low_percentile": q_low,
        "high_percentile": q_high,
        "color_min": float(by_q.loc[q_low, "value"]),
        "color_median": float(by_q.loc[50.0, "value"]),
        "color_max": float(by_q.loc[q_high, "value"]),
        "low_bootstrap_half_width_68": np.nan,
        "median_bootstrap_half_width_68": np.nan,
        "high_bootstrap_half_width_68": np.nan,
        "expected_lower_tail_events": float(n_values * q_low / 100.0),
        "expected_upper_tail_events": float(n_values * q_low / 100.0),
        "nominal_coverage_percent": float(q_high - q_low),
    }
    return diagnostics, selected


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

# ============================================================================
# LEGACY ADAPTIVE-ENVELOPE CONVERGENCE
# ============================================================================

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kne_envelope_core as envelope_core

COMPONENTS = ("color_min", "color_median", "color_max")
PROFILE_VERSION = 2


def colour_errors(
    value,
    reference,
    tolerance_mode="absolute",
    relative_tolerance=0.10,
    absolute_tolerance_mag=0.10,
):
    """Return absolute/relative errors and convergence flags.

    ``absolute`` tests ``abs(value-reference) <= absolute_tolerance_mag``.
    ``relative`` tests ``abs(value-reference) <= relative_tolerance*abs(reference)``.
    Relative errors are still recorded for diagnostics in absolute mode, but
    they do not decide convergence. Missing or non-finite values always fail.
    """
    value, reference = np.broadcast_arrays(
        np.asarray(value, dtype=float), np.asarray(reference, dtype=float)
    )
    finite = np.isfinite(value) & np.isfinite(reference)
    difference = np.abs(value - reference)
    relative = np.full(value.shape, np.inf)
    np.divide(difference, np.abs(reference), out=relative,
              where=finite & (reference != 0))
    relative[finite & (reference == 0) & (difference == 0)] = 0.0
    if tolerance_mode == "absolute":
        tolerance = np.full(value.shape, float(absolute_tolerance_mag))
    elif tolerance_mode == "relative":
        tolerance = relative_tolerance * np.abs(reference)
    else:
        raise ValueError("tolerance_mode must be 'absolute' or 'relative'")
    # Roundoff protection only: this does not create a physical mag floor.
    slack = 32 * np.finfo(float).eps * np.maximum.reduce(
        [np.ones(value.shape), np.abs(value), np.abs(reference)]
    )
    passes = finite & (difference <= tolerance + slack)
    return difference, relative, passes


def envelope_profile(a, sample, arrays, indices, depths, bands, edges, centres,
                     reference=None):
    """Compute one full profile, delegating percentile validity to the builder."""
    from dynamic_color_cut_common import assign_event_splits, parse_floats

    split = assign_event_splits(sample.event_id.astype(str).tolist(),
                                a.train_fraction, a.validation_fraction, a.builder_seed)
    train_indices = indices[split.split.to_numpy() == "train"]
    key = (
        "colour_extincted"
        if a.stage in envelope_core.MW_EXTINCTED_STAGES
        else "colour_intrinsic"
    )
    colours = arrays[key][train_indices]
    available = (np.isfinite(arrays["colour_intrinsic"][train_indices])
                 & np.isfinite(arrays["colour_extincted"][train_indices]))
    if a.stage in envelope_core.M5_STAGES:
        keys = ("mag_sel_int_1", "mag_sel_int_2") if a.stage == "m5_no_mw" else (
            "mag_sel_ext_1", "mag_sel_ext_2")
        for name, band in zip(keys, bands):
            mags = arrays[name][train_indices]
            available &= np.isfinite(mags) & (mags <= depths[band])
    tails = parse_floats(a.tail_percentiles or envelope_core.DEFAULT_TAIL_GRID)
    tails = sorted(set(tails + [25., 30., 35., 40., 42., 44., 45., 46., 47., 48., 49., 50.]))
    if reference is not None and a.bound_mode == "reference-percentiles":
        tails = sorted(set(tails + reference.loc[reference.valid, "low_percentile"].tolist()))
    rng = np.random.default_rng(
        a.builder_seed
        + 1000 * envelope_core.STAGE_BOOTSTRAP_SEED_OFFSETS[a.stage]
    )
    rows = []
    active = True
    reference_lookup = reference.set_index("bin_index") if reference is not None else None
    for b, centre in enumerate(centres):
        row = {"bin_index": b, "t_low_days": float(edges[b]),
               "t_high_days": float(edges[b + 1]), "t_center_days": float(centre),
               "valid": False, "status": "after_first_invalid_bin", "N_train": 0,
               "low_percentile": np.nan, "high_percentile": np.nan,
               "color_min": np.nan, "color_median": np.nan, "color_max": np.nan}
        if not active:
            rows.append(row)
            continue
        values = colours[available[:, b], b].astype(float)
        row["N_train"] = len(values)
        if len(values) < a.minimum_events:
            row["status"] = "insufficient_events"
            active = False
            rows.append(row)
            continue
        diagnostics, selected = percentile_diagnostics(
            values, tails, getattr(a, "bootstrap", True), a.n_boot,
            a.bootstrap_threshold_mag, a.minimum_tail_events, rng)
        if (reference_lookup is not None and a.bound_mode == "reference-percentiles"
                and bool(reference_lookup.loc[b, "valid"])):
            ref = reference_lookup.loc[b]
            qs = [float(ref.low_percentile), 50., float(ref.high_percentile)]
            by_q = diagnostics.set_index("percentile")
            if all(q in by_q.index and bool(by_q.loc[q, "percentile_valid"]) for q in qs):
                selected = {"low_percentile": qs[0], "high_percentile": qs[2],
                            **{name: float(by_q.loc[q, "value"]) for name, q in zip(COMPONENTS, qs)}}
            else:
                selected = None
        if selected is None:
            row["status"] = "no_valid_percentile_pair"
            active = False
        else:
            row.update({k: float(selected[k]) for k in (*COMPONENTS, "low_percentile", "high_percentile")})
            if selected["low_percentile"] >= selected["high_percentile"] and not a.allow_degenerate_envelope:
                row["status"] = "degenerate_p50_p50_excluded"
                active = False
            else:
                row.update(valid=True, status="valid")
        rows.append(row)
    return pd.DataFrame(rows)


def compare_profiles(
    profile,
    reference,
    tolerance_mode="absolute",
    relative_tolerance=0.10,
    absolute_tolerance_mag=0.10,
):
    """Compare every reference-valid bin; absent sample bins always fail."""
    ref = reference.loc[reference.valid].copy()
    if ref.empty:
        raise ValueError("Largest-N reference has no valid envelope bin")
    other = profile.set_index("bin_index").reindex(ref.bin_index)
    out = ref[["bin_index", "t_low_days", "t_high_days", "t_center_days"]].copy().reset_index(drop=True)
    out["reference_N_train"] = ref.N_train.to_numpy()
    out["sample_N_train"] = other.N_train.to_numpy()
    supported = other.valid.eq(True).to_numpy(dtype=bool)
    out["sample_bin_valid"] = supported
    out["reference_low_percentile"] = ref.low_percentile.to_numpy()
    out["reference_high_percentile"] = ref.high_percentile.to_numpy()
    out["sample_low_percentile"] = other.low_percentile.to_numpy()
    out["sample_high_percentile"] = other.high_percentile.to_numpy()
    joint = supported.copy()
    for name in COMPONENTS:
        values = other[name].to_numpy(float)
        targets = ref[name].to_numpy(float)
        difference, relative, passes = colour_errors(
            values,
            targets,
            tolerance_mode=tolerance_mode,
            relative_tolerance=relative_tolerance,
            absolute_tolerance_mag=absolute_tolerance_mag,
        )
        passes &= supported
        relative[~supported] = np.inf
        out[f"sample_{name}"] = values
        out[f"reference_{name}"] = targets
        out[f"absolute_error_{name}_mag"] = difference
        out[f"relative_error_{name}"] = relative
        out[f"passes_{name}"] = passes
        joint &= passes
    out["all_three_pass"] = joint
    return out


def repetition_record(comparison, rep, size, subset_seed, tolerance_mode="absolute"):
    record = {"repetition": rep, "subsample_seed": subset_seed, "N_requested": size,
              "N_reference_bins": len(comparison),
              "N_supported_reference_bins": int(comparison.sample_bin_valid.sum()),
              "N_bins_all_three_pass": int(comparison.all_three_pass.sum()),
              "all_reference_bins_pass": bool(comparison.all_three_pass.all())}
    absolute_errors = []
    relative_errors = []
    for name in COMPONENTS:
        absolute_error = float(comparison[f"absolute_error_{name}_mag"].max())
        relative_error = float(comparison[f"relative_error_{name}"].max())
        record[f"max_absolute_error_{name}_mag"] = absolute_error
        record[f"max_relative_error_{name}"] = relative_error
        record[f"all_bins_pass_{name}"] = bool(comparison[f"passes_{name}"].all())
        absolute_errors.append(absolute_error)
        relative_errors.append(relative_error)
    record["max_absolute_error_all_mag"] = max(absolute_errors)
    record["max_relative_error_all"] = max(relative_errors)
    if tolerance_mode == "absolute":
        record["max_metric_error_all"] = record["max_absolute_error_all_mag"]
    elif tolerance_mode == "relative":
        record["max_metric_error_all"] = record["max_relative_error_all"]
    else:
        raise ValueError("tolerance_mode must be 'absolute' or 'relative'")
    return record


def safe_quantile(values, q):
    """Quantile that preserves infinite errors instead of producing NaN."""
    v = np.sort(np.asarray(values, float))
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan
    p = (len(v) - 1) * q
    i, j = int(np.floor(p)), int(np.ceil(p))
    if i == j:
        return float(v[i])
    if np.isinf(v[j]):
        return np.inf
    return float(v[i] + (p - i) * (v[j] - v[i]))


def aggregate_colour_results(raw, required_fraction=0.9):
    rows = []
    for size, group in raw.groupby("N_requested", sort=True):
        fraction = float(group.all_reference_bins_pass.mean())
        row = {"N_requested": int(size), "repetitions": len(group),
               "fraction_all_bins_all_three_pass": fraction,
               "point_passes_stability": bool(fraction + 1e-12 >= required_fraction),
               "minimum_supported_reference_bins": int(group.N_supported_reference_bins.min())}
        metric_columns = [
            *(f"max_absolute_error_{name}_mag" for name in (*COMPONENTS, "all")),
            *(f"max_relative_error_{name}" for name in (*COMPONENTS, "all")),
            "max_metric_error_all",
        ]
        for column in metric_columns:
            for label, q in (("median", .5), ("p16", .16), ("p84", .84)):
                row[f"{column}_{label}"] = safe_quantile(group[column], q)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["all_larger_sizes_also_stable"] = np.logical_and.accumulate(
        summary.point_passes_stability.to_numpy()[::-1])[::-1]
    candidates = summary.loc[summary.all_larger_sizes_also_stable, "N_requested"]
    threshold = int(candidates.iloc[0]) if len(candidates) else None
    return summary, threshold


def save_colour_plot(raw, summary, a, threshold, path):
    fig, (ax, frac_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    x = summary.N_requested.to_numpy(float)
    absolute_mode = a.tolerance_mode == "absolute"
    factor = 1.0 if absolute_mode else 100.0
    finite = raw.max_metric_error_all.to_numpy(float) * factor
    finite = finite[np.isfinite(finite)]
    requested_tolerance = (
        a.absolute_tolerance_mag if absolute_mode else 100.0 * a.relative_tolerance
    )
    minimum_cap = 0.30 if absolute_mode else 30.0
    maximum_cap = 3.0 if absolute_mode else 1000.0
    cap = max(minimum_cap, requested_tolerance * 3,
              min(maximum_cap, float(np.max(finite)) * 1.05 if len(finite) else minimum_cap))
    def clip(values):
        return np.minimum(np.asarray(values, float) * factor, cap)
    for _, group in raw.groupby("repetition"):
        group = group.sort_values("N_requested")
        ax.plot(group.N_requested, clip(group.max_metric_error_all),
                color="0.65", alpha=.35, linewidth=.8)
    ax.fill_between(x, clip(summary.max_metric_error_all_p16),
                    clip(summary.max_metric_error_all_p84), color="#2471a3", alpha=.18,
                    label="16–84% des sous-échantillons")
    ax.plot(x, clip(summary.max_metric_error_all_median), "o-", color="#2471a3",
            label="Médiane de l'écart maximal")
    ax.axhline(requested_tolerance, color="#a93226", linestyle="--",
               label=(f"Tolérance absolue : {a.absolute_tolerance_mag:g} mag"
                      if absolute_mode else
                      f"Tolérance relative : {100 * a.relative_tolerance:g}%"))
    ax.set_ylim(0, cap * 1.06)
    ax.set_ylabel(("Écart absolu maximal (mag)\n"
                   if absolute_mode else "Écart relatif maximal (%)\n")
                  + "médiane et deux bornes, tous les bins")
    if np.any(~np.isfinite(raw.max_metric_error_all)) or np.any(finite > cap):
        ax.text(.02, .98, "Valeurs infinies ou hors échelle plafonnées en haut\n"
                "(bin de référence non supporté par le sous-échantillon)", transform=ax.transAxes,
                va="top", fontsize=8)
    ax.legend(loc="best", fontsize=9)
    frac_ax.plot(x, 100 * summary.fraction_all_bins_all_three_pass, "o-", color="#2471a3")
    frac_ax.axhline(100 * a.required_fraction, color="#a93226", linestyle="--")
    frac_ax.set_ylim(-3, 103)
    frac_ax.set_ylabel("Répétitions\ncompatibles (%)")
    frac_ax.set_xlabel("Nombre de KNe injectées")
    for axis in (ax, frac_ax):
        axis.set_xscale("log")
        axis.grid(alpha=.2)
        if threshold is not None:
            axis.axvline(threshold, color="#287d46", linestyle=":")
    if threshold is not None:
        frac_ax.text(.98, .07, f"N requis sur la grille : {threshold:,}".replace(",", " "),
                     transform=frac_ax.transAxes, ha="right", color="#287d46")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def atomic_csv(frame, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def run_colour_convergence(a, root, sizes, bands, edges, centres):
    import analyze_envelope_population_convergence_fast as fast
    import build_color_envelopes as builder
    import dynamic_color_cut_common as common

    if a.tolerance_mode not in {"absolute", "relative"}:
        raise ValueError("tolerance-mode must be 'absolute' or 'relative'")
    if not np.isfinite(a.relative_tolerance) or a.relative_tolerance <= 0:
        raise ValueError("relative-tolerance must be finite and positive")
    if not np.isfinite(a.absolute_tolerance_mag) or a.absolute_tolerance_mag <= 0:
        raise ValueError("absolute-tolerance-mag must be finite and positive")
    if (a.minimum_events < 1 or a.minimum_tail_events < 0
            or (getattr(a, "bootstrap", True)
                and a.bootstrap_threshold_mag < 0)):
        raise ValueError("Invalid event-count or bootstrap-validity limits")
    if not (0 < a.train_fraction < 1 and 0 <= a.validation_fraction < 1
            and a.train_fraction + a.validation_fraction < 1):
        raise ValueError("Invalid train/validation fractions")
    output = a.output_dir.expanduser().resolve() if a.output_dir else (
        root / "analysis" /
        f"convergence_colours_{a.color_pair.replace('-', '_')}_"
        f"{a.photometric_system}_{a.stage}"
        f"{'_no_bootstrap' if not getattr(a, 'bootstrap', True) else ''}")
    output.mkdir(parents=True, exist_ok=True)
    bargs = fast.make_builder_args(a, root)
    depths = (
        fast.get_depths(a, bargs, root, bands)
        if a.stage in envelope_core.M5_STAGES
        else {}
    )
    conf = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
            if k not in {"resume", "output_dir", "fit_exponential", "tolerance_days"}}
    conf.update(method="per_bin_colour_convergence", profile_version=PROFILE_VERSION,
                run_dir=str(root), sample_sizes=sizes, applied_depths=depths,
                comparison="all three quantities in every initial consecutive valid reference bin",
                tolerance_mode=a.tolerance_mode,
                tolerance=(f"absolute difference <= {a.absolute_tolerance_mag:g} mag"
                           if a.tolerance_mode == "absolute" else
                           f"relative difference <= {a.relative_tolerance:g}"),
                reference="largest tested N, paired by subsampling repetition")
    # Do not reuse profiles silently after changing the builder, input summary,
    # or any recorded photometry shard. Content hash for code/summary; stat for
    # potentially enormous photometry products.
    conf["source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path(fast.__file__), Path(builder.__file__),
        Path(envelope_core.__file__), Path(common.__file__), root / "summary.csv")}
    files = sorted((root / "parquet").glob("synthetic_part_*.parquet"))
    if not files:
        files = sorted((root / "csv").glob("lightcurve_*.csv"))
    conf["photometry_signature"] = hashlib.sha256(json.dumps([
        [str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns] for p in files
    ]).encode()).hexdigest()
    conf_path = output / "colour_configuration.json"
    if a.resume and conf_path.exists() and json.loads(conf_path.read_text()) != conf:
        raise ValueError("Colour-run configuration differs; choose a new output directory or --no-resume")
    conf_path.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")
    checkpoint_dir = output / "colour_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tasks = [(rep, size) for rep in range(a.repetitions) for size in sizes]
    missing = any(not (checkpoint_dir / f"rep_{rep:03d}_N_{size:09d}.csv").is_file()
                  for rep, size in tasks)
    population = None
    if missing or not a.resume:
        metadata, event_ids, arrays = fast.load_population_once(a, root, bands, centres, sizes)
        population = (metadata, arrays, {eid: i for i, eid in enumerate(event_ids)})
    all_profiles, comparisons, records, references = [], [], [], []
    counter = 0
    for rep in range(a.repetitions):
        reference = None
        seed = a.subsample_seed + rep
        # Compute the largest sample first, so each repetition has its own
        # reference and optional reference-percentile targets.
        for size in [sizes[-1], *sizes[:-1]]:
            counter += 1
            checkpoint = checkpoint_dir / f"rep_{rep:03d}_N_{size:09d}.csv"
            if a.resume and checkpoint.is_file():
                profile = pd.read_csv(checkpoint, float_precision="round_trip")
                print(f"[{counter}/{len(tasks)}] reusing repetition={rep + 1}, N={size:,}")
            else:
                print(f"[{counter}/{len(tasks)}] repetition={rep + 1}/{a.repetitions}, N={size:,}")
                metadata, arrays, position = population
                sample = fast.select_loaded_metadata_subset(metadata, size, seed)
                indices = np.asarray([position[eid] for eid in sample.event_id.astype(str)], dtype=int)
                profile = envelope_profile(a, sample, arrays, indices, depths, bands, edges, centres,
                                           reference=reference)
                atomic_csv(profile, checkpoint)
            if size == sizes[-1]:
                reference = profile
                if not profile.valid.any():
                    raise ValueError(f"No valid reference envelope for repetition {rep + 1}; "
                                     "check the saved filter system, depth, and validity settings")
                refs = profile.loc[profile.valid].copy()
                refs.insert(0, "repetition", rep)
                references.append(refs)
            tagged = profile.copy()
            tagged.insert(0, "N_requested", size)
            tagged.insert(0, "repetition", rep)
            all_profiles.append(tagged)
            comparison = compare_profiles(
                profile,
                reference,
                tolerance_mode=a.tolerance_mode,
                relative_tolerance=a.relative_tolerance,
                absolute_tolerance_mag=a.absolute_tolerance_mag,
            )
            record = repetition_record(
                comparison, rep, size, seed, tolerance_mode=a.tolerance_mode
            )
            records.append(record)
            comparison.insert(0, "N_requested", size)
            comparison.insert(0, "repetition", rep)
            comparisons.append(comparison)
    raw = pd.DataFrame(records).sort_values(["repetition", "N_requested"])
    summary, threshold = aggregate_colour_results(raw, a.required_fraction)
    atomic_csv(raw, output / "envelope_colour_convergence_raw.csv")
    atomic_csv(summary, output / "envelope_colour_convergence_summary.csv")
    atomic_csv(pd.concat(comparisons, ignore_index=True), output / "envelope_colour_convergence_per_bin.csv")
    atomic_csv(pd.concat(all_profiles, ignore_index=True), output / "envelope_colour_profiles.csv")
    atomic_csv(pd.concat(references, ignore_index=True), output / "envelope_colour_reference.csv")
    plot_path = output / "envelope_colour_convergence.png"
    save_colour_plot(raw, summary, a, threshold, plot_path)
    result = dict(conf, stable_population_threshold=threshold,
                  below_largest_tested_size=bool(threshold is not None and threshold < sizes[-1]),
                  stability_rule="first tested N where the required fraction of repetitions passes all three colours in all reference bins, and every larger tested N also passes",
                  scientific_scope="conditional convergence to the largest saved sample, not an asymptotic guarantee")
    (output / "envelope_colour_convergence_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    last = pd.concat(references).groupby("repetition").t_high_days.max()
    print("Reference valid envelope last edges [days]:", last.tolist())
    print("Stable colour-population threshold:", f"{threshold:,} KNe" if threshold is not None else "not determined")
    if threshold == sizes[-1]:
        print("Only the largest tested size passes: it is compared with itself; no plateau below it is established.")
    print("Plot:", plot_path)
    print("Summary:", output / "envelope_colour_convergence_summary.csv")
    print("Per-bin comparisons:", output / "envelope_colour_convergence_per_bin.csv")

# ============================================================================
# FIXED-PERCENTILE SHAPE AND TEMPORAL-EXTENT CONVERGENCE
# ============================================================================

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kne_envelope_core as envelope_core


PROFILE_VERSION = 3
FIXED_PERCENTILES = (5.0, 16.0, 25.0, 50.0, 75.0, 84.0, 95.0)


def percentile_id(q: float) -> str:
    """Stable, CSV-friendly identifier for a percentile."""
    return "p" + (f"{float(q):g}".replace(".", "p"))


FIXED_ESTIMATORS = {percentile_id(q): float(q) for q in FIXED_PERCENTILES}
FIXED_FAMILIES = {
    "central_90": ("p5", "p50", "p95"),
    "central_68": ("p16", "p50", "p84"),
    "central_50": ("p25", "p50", "p75"),
}
ANALYSIS_GROUPS = dict(FIXED_FAMILIES)
PLOTTED_GROUPS = ("central_90", "central_68", "central_50")
PRIMARY_DECLARATIONS = frozenset(PLOTTED_GROUPS)
GROUP_LABELS = {
    "central_90": "p5-p95",
    "central_68": "p16-p84",
    "central_50": "p25-p75",
}
GROUP_COLOURS = {
    "central_90": "#ca6f1e",
    "central_68": "#2471a3",
    "central_50": "#229954",
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _validity_status(row: pd.Series) -> str:
    tail = bool(row["tail_support_valid"])
    width = bool(row["bootstrap_width_valid"])
    if tail and width:
        return "valid"
    if not tail and not width:
        return "tail_and_bootstrap_invalid"
    if not tail:
        return "tail_support_invalid"
    return "bootstrap_width_invalid"


def _empty_estimator_row(base: dict, name: str, kind: str, q=np.nan,
                         status="insufficient_events") -> dict:
    return {
        **base,
        "estimator_id": name,
        "estimator_kind": kind,
        "percentile": q,
        "value": np.nan,
        "valid": False,
        "status": status,
        "bootstrap_p16": np.nan,
        "bootstrap_p84": np.nan,
        "bootstrap_half_width_68": np.nan,
        "expected_nearest_tail_events": np.nan,
        "bootstrap_width_valid": False,
        "tail_support_valid": False,
    }


def multiquantile_profile(a, sample, arrays, indices, depths, bands, edges,
                          centres) -> pd.DataFrame:
    """Calculate the seven fixed percentiles using every selected event."""
    # No train/validation/test split: all events in the selected subset enter
    # the percentile estimators.
    if len(sample) != len(indices):
        raise ValueError("sample and indices must contain the same events")
    analysis_indices = indices
    colour_key = (
        "colour_extincted"
        if a.stage in envelope_core.MW_EXTINCTED_STAGES
        else "colour_intrinsic"
    )
    colours = arrays[colour_key][analysis_indices]
    available = (
        np.isfinite(arrays["colour_intrinsic"][analysis_indices])
        & np.isfinite(arrays["colour_extincted"][analysis_indices])
    )
    if a.stage in envelope_core.M5_STAGES:
        magnitude_keys = (
            ("mag_sel_int_1", "mag_sel_int_2")
            if a.stage == "m5_no_mw"
            else ("mag_sel_ext_1", "mag_sel_ext_2")
        )
        for key, band in zip(magnitude_keys, bands):
            magnitudes = arrays[key][analysis_indices]
            available &= np.isfinite(magnitudes) & (magnitudes <= depths[band])

    # The builder accepts lower-tail percentiles and automatically evaluates
    # their symmetric upper bounds plus p50.
    diagnostic_lower = [5.0, 16.0, 25.0, 50.0]
    rng = np.random.default_rng(
        a.builder_seed + 1000 * envelope_core.STAGE_BOOTSTRAP_SEED_OFFSETS[a.stage]
    )
    rows: list[dict] = []
    for bin_index, centre in enumerate(centres):
        values = colours[available[:, bin_index], bin_index].astype(float)
        base = {
            "bin_index": int(bin_index),
            "t_low_days": float(edges[bin_index]),
            "t_high_days": float(edges[bin_index + 1]),
            "t_center_days": float(centre),
            "N_available": int(len(values)),
        }
        if len(values) < a.minimum_events:
            for name, q in FIXED_ESTIMATORS.items():
                rows.append(_empty_estimator_row(base, name, "fixed", q))
            continue

        diagnostics, _ = percentile_diagnostics(
            values,
            diagnostic_lower,
            getattr(a, "bootstrap", True),
            a.n_boot,
            a.bootstrap_threshold_mag,
            a.minimum_tail_events,
            rng,
        )
        by_q = diagnostics.set_index("percentile")
        for name, q in FIXED_ESTIMATORS.items():
            diagnostic = by_q.loc[q]
            rows.append({
                **base,
                "estimator_id": name,
                "estimator_kind": "fixed",
                "percentile": float(q),
                "value": float(diagnostic["value"]),
                "valid": bool(diagnostic["percentile_valid"]),
                "status": _validity_status(diagnostic),
                "bootstrap_p16": float(diagnostic["bootstrap_p16"]),
                "bootstrap_p84": float(diagnostic["bootstrap_p84"]),
                "bootstrap_half_width_68": float(
                    diagnostic["bootstrap_half_width_68"]
                ),
                "expected_nearest_tail_events": float(
                    diagnostic["expected_nearest_tail_events"]
                ),
                "bootstrap_width_valid": bool(
                    diagnostic["bootstrap_width_valid"]
                ),
                "tail_support_valid": bool(diagnostic["tail_support_valid"]),
            })

    return pd.DataFrame(rows)


def compare_shape_group(profile: pd.DataFrame, reference: pd.DataFrame,
                        group_id: str, members: tuple[str, ...],
                        shape_t_max: float, tolerance_mode="absolute",
                        relative_tolerance=0.10,
                        absolute_tolerance_mag=0.10) -> pd.DataFrame:
    """Compare a group on every bin in the predeclared shape window."""
    ref = reference.loc[
        reference.estimator_id.isin(members)
        & (reference.t_high_days <= shape_t_max + 1e-10)
    ].copy()
    if ref.empty:
        raise ValueError(f"No reference rows in the shape window for {group_id}")
    expected_bins = sorted(ref.bin_index.unique())
    expected = pd.MultiIndex.from_product(
        [expected_bins, members], names=["bin_index", "estimator_id"]
    )
    ref = ref.set_index(["bin_index", "estimator_id"]).reindex(expected)
    candidate = (
        profile.loc[profile.estimator_id.isin(members)]
        .set_index(["bin_index", "estimator_id"])
        .reindex(expected)
    )
    difference, relative, passes = colour_errors(
        candidate.value.to_numpy(float),
        ref.value.to_numpy(float),
        tolerance_mode=tolerance_mode,
        relative_tolerance=relative_tolerance,
        absolute_tolerance_mag=absolute_tolerance_mag,
    )
    reference_valid = ref.valid.fillna(False).to_numpy(bool)
    sample_valid = candidate.valid.fillna(False).to_numpy(bool)
    supported = reference_valid & sample_valid
    passes &= supported
    out = pd.DataFrame({
        "group_id": group_id,
        "bin_index": expected.get_level_values("bin_index").to_numpy(int),
        "estimator_id": expected.get_level_values("estimator_id"),
        "t_low_days": ref.t_low_days.to_numpy(float),
        "t_high_days": ref.t_high_days.to_numpy(float),
        "reference_percentile": ref.percentile.to_numpy(float),
        "sample_percentile": candidate.percentile.to_numpy(float),
        "reference_value": ref.value.to_numpy(float),
        "sample_value": candidate.value.to_numpy(float),
        "reference_valid": reference_valid,
        "sample_valid": sample_valid,
        "both_supported": supported,
        "absolute_error_mag": difference,
        "relative_error": relative,
        "passes_tolerance": passes,
    })
    return out


def shape_repetition_record(comparison: pd.DataFrame, repetition: int,
                            size: int, seed: int,
                            tolerance_mode="absolute") -> dict:
    """Reduce one group comparison without hiding unsupported values."""
    supported = comparison.both_supported.to_numpy(bool)
    absolute = comparison.absolute_error_mag.to_numpy(float)[supported]
    relative = comparison.relative_error.to_numpy(float)[supported]
    unique_bins = comparison[["bin_index", "reference_valid", "sample_valid"]]
    ref_bin_support = unique_bins.groupby("bin_index").reference_valid.all()
    sample_bin_support = unique_bins.groupby("bin_index").sample_valid.all()
    all_pass = bool(
        comparison.reference_valid.all()
        and comparison.sample_valid.all()
        and comparison.passes_tolerance.all()
    )
    return {
        "repetition": int(repetition),
        "subsample_seed": int(seed),
        "N_requested": int(size),
        "group_id": str(comparison.group_id.iloc[0]),
        "shape_horizon_days": float(comparison.t_high_days.max()),
        "N_shape_bins": int(comparison.bin_index.nunique()),
        "N_reference_supported_shape_bins": int(ref_bin_support.sum()),
        "N_sample_supported_shape_bins": int(sample_bin_support.sum()),
        "N_values_required": int(len(comparison)),
        "N_values_supported": int(supported.sum()),
        "N_values_passing": int(comparison.passes_tolerance.sum()),
        "reference_shape_supported": bool(comparison.reference_valid.all()),
        "sample_shape_supported": bool(comparison.sample_valid.all()),
        "max_absolute_error_supported_mag": (
            float(np.max(absolute)) if len(absolute) else np.nan
        ),
        "max_relative_error_supported": (
            float(np.max(relative)) if len(relative) else np.nan
        ),
        "max_metric_error_supported": (
            float(np.max(absolute if tolerance_mode == "absolute" else relative))
            if len(absolute) else np.nan
        ),
        "all_shape_values_pass": all_pass,
    }


def last_consecutive_edge(profile: pd.DataFrame,
                          members: tuple[str, ...]) -> tuple[float, int]:
    """Last upper edge of the initial sequence where all members are valid."""
    selected = profile.loc[profile.estimator_id.isin(members)].copy()
    count = 0
    edge = np.nan
    for bin_index in sorted(profile.bin_index.unique()):
        rows = selected.loc[selected.bin_index.eq(bin_index)]
        valid = (
            len(rows) == len(members)
            and set(rows.estimator_id) == set(members)
            and bool(rows.valid.all())
        )
        if not valid:
            break
        count += 1
        edge = float(rows.t_high_days.iloc[0])
    return edge, count


def shape_horizon(reference: pd.DataFrame, members: tuple[str, ...], a) -> float:
    """Return the fixed or family-specific percentile-comparison horizon."""
    mode = getattr(a, "shape_horizon_mode", "fixed")
    if mode == "fixed":
        return float(a.shape_t_max)
    if mode != "reference-minus-tolerance":
        raise ValueError(f"Unknown shape horizon mode: {mode}")
    reference_edge, _ = last_consecutive_edge(reference, members)
    if not np.isfinite(reference_edge):
        raise ValueError("Reference has no statistically valid temporal bin")
    horizon = float(reference_edge - a.tolerance_days)
    minimum_edge = float(reference.t_high_days.min())
    if horizon < minimum_edge - 1e-10:
        raise ValueError(
            "Reference-minus-tolerance shape horizon precedes the first bin"
        )
    return horizon


def extent_repetition_record(profile: pd.DataFrame, reference: pd.DataFrame,
                             group_id: str, members: tuple[str, ...],
                             repetition: int, size: int, seed: int,
                             tolerance_days: float) -> dict:
    edge, bins = last_consecutive_edge(profile, members)
    reference_edge, reference_bins = last_consecutive_edge(reference, members)
    finite = np.isfinite(edge) and np.isfinite(reference_edge)
    difference = abs(edge - reference_edge) if finite else np.nan
    return {
        "repetition": int(repetition),
        "subsample_seed": int(seed),
        "N_requested": int(size),
        "group_id": group_id,
        "last_valid_time_days": edge,
        "N_consecutive_valid_bins": int(bins),
        "reference_last_valid_time_days": reference_edge,
        "reference_consecutive_valid_bins": int(reference_bins),
        "absolute_extent_error_days": difference,
        "extent_within_tolerance": bool(
            finite and difference <= tolerance_days + 1e-12
        ),
    }


def _add_threshold(summary: pd.DataFrame, pass_column: str) -> tuple[pd.DataFrame, int | None]:
    summary = summary.sort_values("N_requested").reset_index(drop=True)
    passes = summary[pass_column].to_numpy(bool)
    summary["all_larger_sizes_also_stable"] = np.logical_and.accumulate(
        passes[::-1]
    )[::-1]
    candidates = summary.loc[summary.all_larger_sizes_also_stable, "N_requested"]
    threshold = int(candidates.iloc[0]) if len(candidates) else None
    return summary, threshold


def aggregate_shape(raw: pd.DataFrame, required_fraction: float):
    summaries = []
    thresholds: dict[str, int | None] = {}
    for group_id, group_rows in raw.groupby("group_id", sort=False):
        rows = []
        for size, values in group_rows.groupby("N_requested", sort=True):
            fraction = float(values.all_shape_values_pass.mean())
            row = {
                "group_id": group_id,
                "N_requested": int(size),
                "repetitions": int(len(values)),
                "fraction_all_shape_values_pass": fraction,
                "point_passes_stability": bool(
                    fraction + 1e-12 >= required_fraction
                ),
                "minimum_supported_shape_bins": int(
                    values.N_sample_supported_shape_bins.min()
                ),
                "minimum_supported_values": int(values.N_values_supported.min()),
                "shape_horizon_days": (
                    float(values.shape_horizon_days.median())
                    if "shape_horizon_days" in values else np.nan
                ),
            }
            for column in (
                "max_absolute_error_supported_mag",
                "max_relative_error_supported",
                "max_metric_error_supported",
            ):
                for label, q in (("p16", .16), ("median", .5), ("p84", .84)):
                    row[f"{column}_{label}"] = safe_quantile(values[column], q)
            rows.append(row)
        summary, threshold = _add_threshold(
            pd.DataFrame(rows), "point_passes_stability"
        )
        summaries.append(summary)
        thresholds[group_id] = threshold
    return pd.concat(summaries, ignore_index=True), thresholds


def aggregate_extent(raw: pd.DataFrame, required_fraction: float):
    summaries = []
    thresholds: dict[str, int | None] = {}
    for group_id, group_rows in raw.groupby("group_id", sort=False):
        rows = []
        for size, values in group_rows.groupby("N_requested", sort=True):
            fraction = float(values.extent_within_tolerance.mean())
            row = {
                "group_id": group_id,
                "N_requested": int(size),
                "repetitions": int(len(values)),
                "fraction_extent_within_tolerance": fraction,
                "point_passes_stability": bool(
                    fraction + 1e-12 >= required_fraction
                ),
            }
            for column in (
                "last_valid_time_days", "reference_last_valid_time_days",
                "absolute_extent_error_days",
            ):
                for label, q in (("p16", .16), ("median", .5), ("p84", .84)):
                    # Temporal edges live on the discrete bin grid.  Use an
                    # actually observed repetition value instead of linearly
                    # interpolating artificial times such as 6.22 days.
                    finite = pd.to_numeric(values[column], errors="coerce")
                    finite = finite[np.isfinite(finite)].to_numpy(float)
                    row[f"{column}_{label}"] = (
                        float(np.quantile(finite, q, method="nearest"))
                        if len(finite) else np.nan
                    )
            rows.append(row)
        summary, threshold = _add_threshold(
            pd.DataFrame(rows), "point_passes_stability"
        )
        summaries.append(summary)
        thresholds[group_id] = threshold
    return pd.concat(summaries, ignore_index=True), thresholds


def threshold_table(shape_thresholds: dict, extent_thresholds: dict,
                    largest_size: int) -> pd.DataFrame:
    rows = []
    for group_id in ANALYSIS_GROUPS:
        shape = shape_thresholds.get(group_id)
        extent = extent_thresholds.get(group_id)
        recommended = max(shape, extent) if shape is not None and extent is not None else None
        rows.append({
            "group_id": group_id,
            "members": "+".join(ANALYSIS_GROUPS[group_id]),
            "primary_scientific_declaration": group_id in PRIMARY_DECLARATIONS,
            "N_shape_stable": shape,
            "N_extent_stable": extent,
            "N_recommended_both": recommended,
            "shape_plateau_below_reference": bool(
                shape is not None and shape < largest_size
            ),
            "extent_plateau_below_reference": bool(
                extent is not None and extent < largest_size
            ),
            "both_plateau_below_reference": bool(
                recommended is not None and recommended < largest_size
            ),
        })
    return pd.DataFrame(rows)


def save_shape_plot(raw: pd.DataFrame, summary: pd.DataFrame, a, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for group_id in PLOTTED_GROUPS:
        data = summary.loc[summary.group_id.eq(group_id)].sort_values("N_requested")
        if data.empty:
            continue
        colour = GROUP_COLOURS[group_id]
        x = data.N_requested.to_numpy(float)
        y = data.max_absolute_error_supported_mag_median.to_numpy(float)
        low = data.max_absolute_error_supported_mag_p16.to_numpy(float)
        high = data.max_absolute_error_supported_mag_p84.to_numpy(float)
        ax.fill_between(x, low, high, color=colour, alpha=.10)
        ax.plot(x, y, "o-", color=colour, label=GROUP_LABELS[group_id])
    ax.axhline(a.absolute_tolerance_mag, color="black", linestyle="--",
               label=f"tolérance = {a.absolute_tolerance_mag:g} mag")
    ax.set_ylabel("Écart absolu maximal aux\npercentiles de référence (mag)")
    ax.set_xlabel("Nombre de KNe injectées")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_extent_plot(summary: pd.DataFrame, a, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for group_id in PLOTTED_GROUPS:
        data = summary.loc[summary.group_id.eq(group_id)].sort_values("N_requested")
        if data.empty:
            continue
        colour = GROUP_COLOURS[group_id]
        x = data.N_requested.to_numpy(float)
        y = data.last_valid_time_days_median.to_numpy(float)
        low = data.last_valid_time_days_p16.to_numpy(float)
        high = data.last_valid_time_days_p84.to_numpy(float)
        ax.fill_between(x, low, high, color=colour, alpha=.10)
        ax.plot(x, y, "o-", color=colour, label=GROUP_LABELS[group_id])
    ax.set_ylabel("Dernier bord temporel\nstatistiquement valide (jours)")
    ax.set_xlabel("Nombre de KNe injectées")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def extent_values_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the compact table shown in the temporal-extent PNG."""
    groups = ("central_90", "central_68")
    sizes = sorted(
        pd.to_numeric(
            summary.loc[summary.group_id.isin(groups), "N_requested"],
            errors="coerce",
        ).dropna().astype(int).unique()
    )
    rows = []
    largest_size = max(sizes) if sizes else None
    for size in sizes:
        row = {"N KNe": f"{size:,}".replace(",", " ")}
        for group_id in groups:
            prefix = GROUP_LABELS[group_id]
            selected = summary.loc[
                summary.group_id.eq(group_id)
                & summary.N_requested.eq(size)
            ]
            edge_column = f"{prefix}\nDernier bord valide (j)"
            fraction_column = f"{prefix}\nCompatible réf. (%)"
            plateau_column = f"{prefix}\nPlateau atteint"
            if len(selected) != 1:
                row.update({edge_column: "—", fraction_column: "—", plateau_column: "—"})
                continue
            value = selected.iloc[0]
            median = float(value.last_valid_time_days_median)
            low = float(value.last_valid_time_days_p16)
            high = float(value.last_valid_time_days_p84)
            row[edge_column] = (
                f"{median:.2f} [{low:.2f}, {high:.2f}]"
                if np.isfinite(median) and np.isfinite(low) and np.isfinite(high)
                else "—"
            )
            fraction = float(value.fraction_extent_within_tolerance)
            row[fraction_column] = (
                f"{100 * fraction:.0f}" if np.isfinite(fraction) else "—"
            )
            if size == largest_size:
                # The reference is a self-comparison.  It cannot, by itself,
                # demonstrate that convergence was reached below N_ref.
                row[plateau_column] = "Ref."
            else:
                row[plateau_column] = (
                    "Yes" if bool(value.all_larger_sizes_also_stable) else "No"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def save_extent_values_table(summary: pd.DataFrame, a, path: Path) -> None:
    """Save last-valid-bin values for the 90% and 68% families as a PNG."""
    table_data = extent_values_table(summary)
    if table_data.empty:
        raise ValueError("No central_90/central_68 temporal-extent rows to tabulate")
    _save_plain_table(table_data, path)


def shape_values_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact PNG-table values for percentile-value convergence."""
    groups = ("central_90", "central_68")
    sizes = sorted(
        pd.to_numeric(
            summary.loc[summary.group_id.isin(groups), "N_requested"],
            errors="coerce",
        ).dropna().astype(int).unique()
    )
    rows = []
    largest_size = max(sizes) if sizes else None
    for size in sizes:
        row = {"N KNe": f"{size:,}".replace(",", " ")}
        for group_id in groups:
            prefix = GROUP_LABELS[group_id]
            selected = summary.loc[
                summary.group_id.eq(group_id)
                & summary.N_requested.eq(size)
            ]
            error_column = f"{prefix}\nMax |ΔC| (mag)"
            fraction_column = f"{prefix}\nRépétitions stables (%)"
            plateau_column = f"{prefix}\nPlateau atteint"
            if len(selected) != 1:
                row.update({
                    error_column: "—", fraction_column: "—",
                    plateau_column: "—",
                })
                continue
            value = selected.iloc[0]
            median = float(value.max_absolute_error_supported_mag_median)
            low = float(value.max_absolute_error_supported_mag_p16)
            high = float(value.max_absolute_error_supported_mag_p84)
            row[error_column] = (
                f"{median:.3f} [{low:.3f}, {high:.3f}]"
                if np.isfinite(median) and np.isfinite(low) and np.isfinite(high)
                else "—"
            )
            fraction = float(value.fraction_all_shape_values_pass)
            row[fraction_column] = (
                f"{100 * fraction:.0f}" if np.isfinite(fraction) else "—"
            )
            if size == largest_size:
                row[plateau_column] = "Ref."
            else:
                row[plateau_column] = (
                    "Yes" if bool(value.all_larger_sizes_also_stable) else "No"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _save_plain_table(table_data: pd.DataFrame, path: Path) -> None:
    """Render a title-free table with neutral column headings."""
    if table_data.empty:
        raise ValueError("Cannot render an empty convergence table")
    nrows, ncols = table_data.shape
    fig, ax = plt.subplots(figsize=(15.0, .65 + .48 * nrows))
    ax.axis("off")
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center", colLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.35)
    for column in range(ncols):
        cell = table[(0, column)]
        cell.set_facecolor("#ffffff")
        cell.set_text_props(color="black", weight="bold")
        cell.set_edgecolor("#b7c0c8")
    for row in range(1, nrows + 1):
        base = "#f4f6f7" if row % 2 else "#eaf0f4"
        for column in range(ncols):
            table[(row, column)].set_facecolor(base)
            table[(row, column)].set_edgecolor("white")
        for column in range(ncols):
            verdict = str(table_data.iloc[row - 1, column]).lower()
            colour = {
                "yes": "#d9ead3", "no": "#f4cccc", "ref.": "#d9eaf7",
            }.get(verdict)
            if colour:
                table[(row, column)].set_facecolor(colour)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_shape_values_table(summary: pd.DataFrame, path: Path) -> None:
    """Save the main percentile-value convergence numbers as a PNG."""
    _save_plain_table(shape_values_table(summary), path)


def _configuration(a, root, sizes, depths, fast, builder, common) -> dict:
    conf = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(a).items()
        if key not in {
            "resume", "output_dir", "fit_exponential",
            # These options belong to the two legacy modes and are ignored by
            # the fixed-percentile analysis.
            "train_fraction", "validation_fraction", "tail_percentiles",
            "bound_mode", "allow_degenerate_envelope",
        }
    }
    conf.update({
        "method": "fixed_multi_percentile_shape_and_extent_convergence",
        "profile_version": PROFILE_VERSION,
        "run_dir": str(root),
        "sample_sizes": sizes,
        "applied_depths": depths,
        "fixed_percentiles": list(FIXED_PERCENTILES),
        "fixed_families": {key: list(value) for key, value in FIXED_FAMILIES.items()},
        "population_usage": "100% of each selected subset; no split",
        "shape_definition": (
            "all requested percentile values in every fixed-window bin; "
            + (
                "both reference and sample must satisfy tail and bootstrap support"
                if getattr(a, "bootstrap", True) else
                "both reference and sample must satisfy tail support; bootstrap disabled"
            )
        ),
        "extent_definition": (
            "last upper edge in the initial consecutive sequence where every "
            "member of the percentile family is statistically valid"
        ),
        "reference": (
            "largest tested N; shared across repetitions when it is the full "
            "parent population"
        ),
    })
    source_files = (
        Path(__file__), Path(fast.__file__), Path(builder.__file__),
        Path(envelope_core.__file__),
        Path(common.__file__), root / "summary.csv",
    )
    conf["source_hashes"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    files = sorted((root / "parquet").glob("synthetic_part_*.parquet"))
    if not files:
        files = sorted((root / "csv").glob("lightcurve_*.csv"))
    conf["photometry_signature"] = hashlib.sha256(json.dumps([
        [str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns]
        for path in files
    ]).encode()).hexdigest()
    return conf


def run_multi_percentile_convergence(a, root, sizes, bands, edges, centres):
    """Entry point used by the fast convergence CLI."""
    import analyze_envelope_population_convergence_fast as fast
    import build_color_envelopes as builder
    import dynamic_color_cut_common as common

    if a.tolerance_mode != "absolute":
        raise ValueError(
            "The multi-percentile primary analysis requires "
            "--tolerance-mode absolute"
        )
    if not np.isfinite(a.absolute_tolerance_mag) or a.absolute_tolerance_mag <= 0:
        raise ValueError("absolute-tolerance-mag must be finite and positive")
    if a.minimum_events < 1 or a.minimum_tail_events < 0:
        raise ValueError("Invalid event-count validity limits")
    if (getattr(a, "bootstrap", True)
            and a.bootstrap_threshold_mag < 0):
        raise ValueError("bootstrap-threshold-mag cannot be negative")
    output = a.output_dir.expanduser().resolve() if a.output_dir else (
        root / "analysis" /
        f"convergence_multiquantile_{a.color_pair.replace('-', '_')}_"
        f"{a.photometric_system}_{a.stage}"
        f"{'_no_bootstrap' if not getattr(a, 'bootstrap', True) else ''}"
    )
    output.mkdir(parents=True, exist_ok=True)
    bargs = fast.make_builder_args(a, root)
    depths = (
        fast.get_depths(a, bargs, root, bands)
        if a.stage in envelope_core.M5_STAGES else {}
    )
    conf = _configuration(a, root, sizes, depths, fast, builder, common)
    conf_path = output / "multi_percentile_configuration.json"
    if a.resume and conf_path.exists():
        previous = json.loads(conf_path.read_text(encoding="utf-8"))
        if previous != conf:
            raise ValueError(
                "Multi-percentile configuration differs; choose a new output "
                "directory or use --no-resume"
            )
    conf_path.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")

    checkpoint_dir = output / "multi_percentile_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tasks = [(rep, size) for rep in range(a.repetitions) for size in sizes]
    missing = any(
        not (checkpoint_dir / f"rep_{rep:03d}_N_{size:09d}.csv").is_file()
        for rep, size in tasks
    )
    population = None
    if missing or not a.resume:
        metadata, event_ids, arrays = fast.load_population_once(
            a, root, bands, centres, sizes
        )
        population = (
            metadata, arrays,
            {event_id: index for index, event_id in enumerate(event_ids)},
        )

    profiles = []
    references = []
    shape_comparisons = []
    shape_records = []
    extent_records = []
    counter = 0
    parent_size = fast.convergence.count_summary_events(root)
    full_parent_is_reference = sizes[-1] == parent_size
    shared_full_reference = None
    for repetition in range(a.repetitions):
        reference = None
        seed = a.subsample_seed + repetition
        for size in [sizes[-1], *sizes[:-1]]:
            counter += 1
            checkpoint = checkpoint_dir / f"rep_{repetition:03d}_N_{size:09d}.csv"
            if (
                size == sizes[-1]
                and full_parent_is_reference
                and shared_full_reference is not None
            ):
                profile = shared_full_reference.copy()
                if not checkpoint.is_file():
                    atomic_csv(profile, checkpoint)
                print(
                    f"[{counter}/{len(tasks)}] reusing shared full-population "
                    f"reference, repetition={repetition + 1}, N={size:,}"
                )
            elif a.resume and checkpoint.is_file():
                profile = pd.read_csv(checkpoint, float_precision="round_trip")
                print(
                    f"[{counter}/{len(tasks)}] reusing repetition={repetition + 1}, "
                    f"N={size:,}"
                )
            else:
                print(
                    f"[{counter}/{len(tasks)}] repetition={repetition + 1}/"
                    f"{a.repetitions}, N={size:,}"
                )
                metadata, arrays, position = population
                sample = fast.select_loaded_metadata_subset(metadata, size, seed)
                indices = np.asarray([
                    position[event_id]
                    for event_id in sample.event_id.astype(str)
                ], dtype=int)
                profile = multiquantile_profile(
                    a, sample, arrays, indices, depths, bands, edges, centres
                )
                atomic_csv(profile, checkpoint)

            if size == sizes[-1]:
                reference = profile
                if full_parent_is_reference and shared_full_reference is None:
                    shared_full_reference = profile.copy()
                tagged_reference = profile.copy()
                tagged_reference.insert(0, "repetition", repetition)
                references.append(tagged_reference)
            tagged = profile.copy()
            tagged.insert(0, "N_requested", size)
            tagged.insert(0, "repetition", repetition)
            profiles.append(tagged)

            for group_id, members in ANALYSIS_GROUPS.items():
                comparison_horizon = shape_horizon(reference, members, a)
                comparison = compare_shape_group(
                    profile,
                    reference,
                    group_id,
                    members,
                    comparison_horizon,
                    tolerance_mode=a.tolerance_mode,
                    relative_tolerance=a.relative_tolerance,
                    absolute_tolerance_mag=a.absolute_tolerance_mag,
                )
                comparison.insert(0, "N_requested", size)
                comparison.insert(0, "repetition", repetition)
                shape_comparisons.append(comparison)
                shape_records.append(shape_repetition_record(
                    comparison,
                    repetition,
                    size,
                    seed,
                    tolerance_mode=a.tolerance_mode,
                ))
                extent_records.append(extent_repetition_record(
                    profile,
                    reference,
                    group_id,
                    members,
                    repetition,
                    size,
                    seed,
                    a.tolerance_days,
                ))

    profile_table = pd.concat(profiles, ignore_index=True)
    reference_table = pd.concat(references, ignore_index=True)
    comparison_table = pd.concat(shape_comparisons, ignore_index=True)
    shape_raw = pd.DataFrame(shape_records).sort_values(
        ["group_id", "repetition", "N_requested"]
    )
    extent_raw = pd.DataFrame(extent_records).sort_values(
        ["group_id", "repetition", "N_requested"]
    )
    shape_summary, shape_thresholds = aggregate_shape(
        shape_raw, a.required_fraction
    )
    extent_summary, extent_thresholds = aggregate_extent(
        extent_raw, a.required_fraction
    )
    thresholds = threshold_table(
        shape_thresholds, extent_thresholds, sizes[-1]
    )

    outputs = {
        "profiles": output / "percentile_profiles.csv",
        "references": output / "percentile_reference_profiles.csv",
        "shape_per_bin": output / "percentile_value_convergence_per_bin.csv",
        "shape_raw": output / "percentile_value_convergence_raw.csv",
        "shape_summary": output / "percentile_value_convergence_summary.csv",
        "extent_raw": output / "temporal_extent_convergence_raw.csv",
        "extent_summary": output / "temporal_extent_convergence_summary.csv",
        "thresholds": output / "convergence_thresholds.csv",
    }
    for frame, key in (
        (profile_table, "profiles"),
        (reference_table, "references"),
        (comparison_table, "shape_per_bin"),
        (shape_raw, "shape_raw"),
        (shape_summary, "shape_summary"),
        (extent_raw, "extent_raw"),
        (extent_summary, "extent_summary"),
        (thresholds, "thresholds"),
    ):
        atomic_csv(frame, outputs[key])

    shape_plot = output / "percentile_value_convergence.png"
    extent_plot = output / "temporal_extent_convergence.png"
    extent_table_plot = output / "temporal_extent_values_table.png"
    shape_table_plot = output / "percentile_value_convergence_values_table.png"
    save_shape_plot(shape_raw, shape_summary, a, shape_plot)
    save_extent_plot(extent_summary, a, extent_plot)
    save_extent_values_table(extent_summary, a, extent_table_plot)
    save_shape_values_table(shape_summary, shape_table_plot)

    results = {
        **conf,
        "shape_thresholds": shape_thresholds,
        "extent_thresholds": extent_thresholds,
        "primary_declarations": sorted(PRIMARY_DECLARATIONS),
        "scientific_scope": (
            "conditional convergence to the largest saved population; not an "
            "asymptotic proof"
        ),
        "largest_reference_warning": (
            "a threshold equal to the largest tested N is only a self-"
            "comparison and does not establish a plateau below the reference"
        ),
    }
    (output / "multi_percentile_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    for group_id in PLOTTED_GROUPS:
        shape = shape_thresholds.get(group_id)
        extent = extent_thresholds.get(group_id)
        print(
            f"{GROUP_LABELS[group_id]}: shape N="
            f"{shape if shape is not None else 'not determined'}; extent N="
            f"{extent if extent is not None else 'not determined'}"
        )
    print("Shape plot:", shape_plot)
    print("Extent plot:", extent_plot)
    print("Extent values table:", extent_table_plot)
    print("Percentile values table:", shape_table_plot)
    print("Threshold table:", outputs["thresholds"])

# ============================================================================
# USER-FRIENDLY TERMINAL INTERFACE AND WORKFLOW ORCHESTRATION
# ============================================================================

def default_configuration() -> argparse.Namespace:
    """Return documented defaults before the interactive questions."""
    return argparse.Namespace(
        run_dir=Path("."), color_pair="g-r", photometric_system="lsst",
        stage="m5_with_mw",
        sample_sizes="1000,3000,10000,30000,50000,75000,100000,150000,200000",
        repetitions=10, subsample_seed=20261002, builder_seed=12345,
        t_min=0.4, t_max=16.0, bin_width=0.4,
        bootstrap=True, n_boot=200,
        minimum_events=30, minimum_tail_events=20.0,
        bootstrap_threshold_mag=0.1, train_fraction=0.70,
        validation_fraction=0.15, tail_percentiles=None,
        m5_scenario="median_p50", threshold_scope="common_global",
        m5_quantile_table=None, ztf_depth_source="manual", ztf_m5_g=None,
        ztf_m5_r=None, ztf_m5_i=None, ztf_depth_quantile_table=None,
        tolerance_days=0.4, required_fraction=0.9,
        stability_metric="colour", shape_t_max=4.0,
        shape_horizon_mode="fixed", tolerance_mode="absolute",
        absolute_tolerance_mag=0.10, relative_tolerance=0.10,
        bound_mode="adaptive", allow_degenerate_envelope=False,
        output_dir=None, resume=True, fit_exponential=True,
    )


def interactive_configuration(args: argparse.Namespace) -> argparse.Namespace:
    """Collect one convergence analysis through terminal questions."""
    args.run_dir = Path(prompt_text("Parent run directory", str(args.run_dir)))
    args.color_pair = prompt_text("Colour pair", args.color_pair)
    args.photometric_system = prompt_choice(
        "Photometric system",
        [("lsst", "LSST"), ("ps1", "PS1"), ("ztf", "ZTF")],
        args.photometric_system,
    )
    args.stage = prompt_choice(
        "Envelope stage", [(stage, stage) for stage in STAGES], args.stage
    )
    args.sample_sizes = prompt_text("Nested population sizes", args.sample_sizes)
    args.repetitions = int(prompt_text("Independent subset repetitions", str(args.repetitions)))
    args.subsample_seed = int(prompt_text("Subset master seed", str(args.subsample_seed)))
    args.builder_seed = int(prompt_text("Builder master seed", str(args.builder_seed)))
    args.t_min = float(prompt_text("First temporal-bin lower edge [days]", str(args.t_min)))
    args.t_max = float(prompt_text("Last temporal-bin upper edge [days]", str(args.t_max)))
    args.bin_width = float(prompt_text("Temporal-bin width [days]", str(args.bin_width)))
    args.minimum_events = int(prompt_text("Minimum finite colours per bin", str(args.minimum_events)))
    args.minimum_tail_events = float(prompt_text(
        "Minimum expected events in each percentile tail",
        str(args.minimum_tail_events),
    ))
    bootstrap_choice = prompt_choice(
        "Bootstrap validation of percentiles",
        [("yes", "Enable bootstrap"), ("no", "Do not bootstrap")],
        "yes" if args.bootstrap else "no",
    )
    args.bootstrap = bootstrap_choice == "yes"
    if args.bootstrap:
        args.n_boot = int(prompt_text(
            "Bootstrap repetitions per bin", str(max(args.n_boot, 1))
        ))
        args.bootstrap_threshold_mag = float(prompt_text(
            "Maximum 68% bootstrap half-width [mag]",
            str(args.bootstrap_threshold_mag),
        ))
    else:
        args.n_boot = 0
    args.stability_metric = prompt_choice(
        "Convergence analysis",
        [("colour", "Fixed percentiles: shape and temporal extent"),
         ("legacy-colour", "Adaptive envelope values"),
         ("last-edge", "Legacy last-valid-bin extent")],
        args.stability_metric,
    )
    if args.stability_metric == "colour":
        args.shape_t_max = float(prompt_text(
            "Fixed shape-comparison horizon [days]", str(args.shape_t_max)
        ))
        args.shape_horizon_mode = prompt_choice(
            "Shape horizon",
            [("fixed", "Use the fixed horizon entered above"),
             ("reference-minus-tolerance", "Reference last edge minus tolerance")],
            args.shape_horizon_mode,
        )
    args.tolerance_mode = prompt_choice(
        "Value-convergence tolerance",
        [("absolute", "Absolute colour difference [mag]"),
         ("relative", "Fractional difference")],
        args.tolerance_mode,
    )
    if args.tolerance_mode == "absolute":
        args.absolute_tolerance_mag = float(prompt_text(
            "Maximum absolute colour difference [mag]",
            str(args.absolute_tolerance_mag),
        ))
    else:
        args.relative_tolerance = float(prompt_text(
            "Maximum fractional colour difference", str(args.relative_tolerance)
        ))
    args.tolerance_days = float(prompt_text(
        "Temporal-extent tolerance [days]", str(args.tolerance_days)
    ))
    args.required_fraction = float(prompt_text(
        "Required fraction of repetitions", str(args.required_fraction)
    ))
    args.m5_scenario = prompt_choice(
        "m5 threshold scenario",
        [(scenario, scenario) for scenario in M5_SCENARIOS],
        args.m5_scenario,
    )
    args.threshold_scope = prompt_choice(
        "m5 latitude scope",
        [("common_global", "Common all-sky thresholds"),
         ("latitude_conditioned", "High-latitude thresholds")],
        args.threshold_scope,
    )
    if args.photometric_system == "ztf":
        for band in sorted(set(split_colour_pair(args.color_pair))):
            value = prompt_text(
                f"Fixed ZTF {band}-band 5-sigma limiting magnitude [AB mag]", ""
            )
            if not value.strip():
                raise ValueError(f"A fixed ZTF {band}-band depth is required")
            setattr(args, f"ztf_m5_{band}", float(value))
    output = prompt_text("Output directory (empty = automatic)", "").strip()
    args.output_dir = Path(output) if output else None
    return args




def validate(a: argparse.Namespace) -> tuple[Path, list[int], tuple[str, str], np.ndarray, np.ndarray]:
    root = resolve_run_root(a.run_dir)
    sizes = parse_sample_sizes(a.sample_sizes)
    bands = split_colour_pair(a.color_pair)
    if a.photometric_system == "ztf" and not set(bands) <= {"g", "r", "i"}:
        raise ValueError("ZTF supports g, r and i only")
    if sizes[-1] > count_summary_events(root):
        raise ValueError("Largest sample size exceeds the events in summary.csv")
    if a.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if getattr(a, "bootstrap", True) and a.n_boot < 1:
        raise ValueError("n-boot must be positive when bootstrap is enabled")
    if a.bin_width <= 0 or a.t_max <= a.t_min or a.tolerance_days < 0:
        raise ValueError("Invalid time grid or tolerance")
    if not a.t_min < a.shape_t_max <= a.t_max:
        raise ValueError("--shape-t-max must lie in (t-min, t-max]")
    shape_steps = (a.shape_t_max - a.t_min) / a.bin_width
    if not np.isclose(shape_steps, round(shape_steps), atol=1e-8):
        raise ValueError("(shape-t-max - t-min) must be an integer multiple of bin-width")
    if not 0 < a.required_fraction <= 1:
        raise ValueError("required-fraction must lie in (0, 1]")
    edges = np.arange(a.t_min, a.t_max + 0.5 * a.bin_width, a.bin_width)
    if not np.isclose(edges[-1], a.t_max, atol=1e-8):
        raise ValueError("(t-max - t-min) must be an integer multiple of bin-width")
    return root, sizes, bands, edges, (edges[:-1] + edges[1:]) / 2


def make_builder_args(a: argparse.Namespace, root: Path) -> argparse.Namespace:
    """Create the builder's default metadata-selection namespace."""
    args = builder.default_configuration()
    args.run_dir = root
    args.color_pair = a.color_pair
    args.photometric_system = a.photometric_system
    args.t_min = a.t_min
    args.t_max = a.t_max
    args.bin_width = a.bin_width
    args.seed = a.builder_seed
    args.max_events = 0
    args.threshold_scope = a.threshold_scope
    args.m5_scenario = a.m5_scenario
    args.m5_quantile_table = a.m5_quantile_table
    # The convergence CLI intentionally uses the all-latitude parent sample,
    # matching the original script's defaults.
    return args


def get_depths(a: argparse.Namespace, bargs: argparse.Namespace, root: Path, bands: tuple[str, str]) -> dict[str, float]:
    if a.photometric_system == "ztf":
        if a.ztf_depth_source == "manual":
            manual = {"g": a.ztf_m5_g, "r": a.ztf_m5_r, "i": a.ztf_m5_i}
            missing = [band for band in set(bands) if manual[band] is None]
            if missing:
                raise ValueError("Missing manual ZTF depth(s): " + ", ".join(missing))
            return {band: float(manual[band]) for band in bands}
        if a.ztf_depth_quantile_table is None:
            raise ValueError("Table mode needs --ztf-depth-quantile-table")
        depths, _ = builder.load_ztf_depth_thresholds(
            directory=None, quantile_table=a.ztf_depth_quantile_table,
            scenario=a.m5_scenario, bands=bands, n_samples=0, seed=0,
            depth_source="table", manual_thresholds=None,
        )
        return depths
    table = builder.discover_m5_table(root, a.m5_quantile_table)
    return builder.load_m5(table, a.threshold_scope, a.m5_scenario, bands)


def load_population_once(a: argparse.Namespace, root: Path, bands: tuple[str, str], centres: np.ndarray, sizes: list[int]):
    """Read relevant event photometry once and return compact numerical arrays."""
    source = SyntheticPopulationSource(root)
    if source.kind == "missing":
        raise FileNotFoundError(f"No saved synthetic population under {root}")
    metadata = prepare_metadata(make_builder_args(a, root), source.available_event_ids())
    maximum = sizes[-1]
    # Form the union of event IDs that can occur in any requested repetition.
    ids_needed: set[str] = set()
    rank_maps: dict[int, dict[str, int]] = {}
    for rep in range(a.repetitions):
        seed = a.subsample_seed + rep
        permutation = np.random.default_rng(seed).permutation(len(metadata))
        selected = permutation[:min(maximum, len(metadata))]
        selected_ids = metadata.iloc[selected].event_id.astype(str).tolist()
        rank_maps[seed] = {event_id: rank for rank, event_id in enumerate(selected_ids)}
        ids_needed.update(selected_ids)
    metadata = metadata.loc[metadata.event_id.astype(str).isin(ids_needed)].copy().reset_index(drop=True)
    # Preserve each permutation's original rank.  Re-drawing from the union
    # would create different, non-nested subsets when repetitions overlap.
    for seed, ranks in rank_maps.items():
        metadata[f"_subsample_rank_{seed}"] = (
            metadata.event_id.astype(str).map(ranks).fillna(np.inf)
        )
    event_ids = metadata.event_id.astype(str).tolist()
    pos = {canonical_event_id(eid): i for i, eid in enumerate(event_ids)}
    n, nt = len(metadata), len(centres)
    band1, band2 = bands
    system = a.photometric_system
    selection_system = "lsst" if system == "ps1" else system
    systems = tuple(dict.fromkeys((system, selection_system)))
    arrays = {
        "colour_intrinsic": np.full((n, nt), np.nan, np.float32),
        "colour_extincted": np.full((n, nt), np.nan, np.float32),
        "mag_sel_ext_1": np.full((n, nt), np.nan, np.float32),
        "mag_sel_ext_2": np.full((n, nt), np.nan, np.float32),
        "mag_sel_int_1": np.full((n, nt), np.nan, np.float32),
        "mag_sel_int_2": np.full((n, nt), np.nan, np.float32),
    }
    meta_by_id = metadata.set_index("event_id", drop=False)
    seen: set[str] = set()
    interval = max(500, n // 20)
    print(f"Reading {n:,} events once for all sample sizes ({source.kind})")
    for eid, curve in source.iter_events(event_ids, bands, systems):
        eid = canonical_event_id(eid)
        if eid in seen:
            raise RuntimeError(f"Duplicate saved event rows for {eid}")
        seen.add(eid)
        i = pos[eid]
        row = meta_by_id.loc[eid]
        curve = curve.copy()
        curve["t_days"] = pd.to_numeric(curve.t_days, errors="coerce")
        curve["band"] = curve.band.astype(str).str.strip().str.lower()
        curve["magnitude"] = pd.to_numeric(curve.mag, errors="coerce")
        if "photometric_system" not in curve:
            curve["photometric_system"] = "lsst"
        curve["photometric_system"] = curve.photometric_system.astype(str).str.lower()
        curve = curve.loc[
            curve.band.isin(bands) & curve.photometric_system.isin(systems)
            & np.isfinite(curve.t_days) & (curve.t_days > 0)
            & np.isfinite(curve.magnitude)
        ]
        target = curve.loc[curve.photometric_system == system]
        select = curve.loc[curve.photometric_system == selection_system]
        m1 = interpolate_band(target, band1, centres)
        m2 = interpolate_band(target, band2, centres)
        s1 = interpolate_band(select, band1, centres)
        s2 = interpolate_band(select, band2, centres)
        a1 = envelope_core.event_extinction_mag(row, system, band1)
        a2 = envelope_core.event_extinction_mag(row, system, band2)
        sa1 = envelope_core.event_extinction_mag(row, selection_system, band1)
        sa2 = envelope_core.event_extinction_mag(row, selection_system, band2)
        arrays["colour_extincted"][i] = m1 - m2
        arrays["colour_intrinsic"][i] = (m1 - a1) - (m2 - a2)
        arrays["mag_sel_ext_1"][i] = s1
        arrays["mag_sel_ext_2"][i] = s2
        arrays["mag_sel_int_1"][i] = s1 - sa1
        arrays["mag_sel_int_2"][i] = s2 - sa2
        if (i + 1) % interval == 0:
            print(f"  loaded rows for {i + 1:,}/{n:,} events")
    print(f"Finished loading: {len(seen):,}/{n:,} events with saved rows")
    return metadata, event_ids, arrays


def select_loaded_metadata_subset(metadata: pd.DataFrame, maximum_events: int,
                                  subsample_seed: int) -> pd.DataFrame:
    """Recover the exact nested subset recorded by ``load_population_once``."""
    rank_column = f"_subsample_rank_{subsample_seed}"
    if rank_column not in metadata:
        # Compatibility fallback for callers supplying a conventional table.
        return envelope_core.select_metadata_subset(
            metadata, maximum_events, subsample_seed
        )
    subset = metadata.loc[metadata[rank_column] < int(maximum_events)].copy()
    expected = min(
        int(maximum_events), int(np.isfinite(metadata[rank_column]).sum())
    )
    if len(subset) != expected:
        # Keep the explicit error rather than silently analysing the wrong
        # population.
        raise RuntimeError(
            f"Stored subset rank {rank_column} produced {len(subset)} events; "
            f"expected {expected}"
        )
    return subset.sort_values(
        "event_id", key=lambda series: pd.to_numeric(series, errors="coerce")
    ).reset_index(drop=True)


def last_adaptive_edge(a: argparse.Namespace, sample: pd.DataFrame, arrays: dict[str, np.ndarray], indices: np.ndarray, depths: dict[str, float], bands: tuple[str, str], edges: np.ndarray, centres: np.ndarray) -> tuple[float, int, float, float, float]:
    ids = sample.event_id.astype(str).tolist()
    split = assign_event_splits(ids, a.train_fraction, a.validation_fraction, a.builder_seed)
    train = split.split.to_numpy() == "train"
    if a.stage in envelope_core.MW_EXTINCTED_STAGES:
        colours = arrays["colour_extincted"][indices]
    else:
        colours = arrays["colour_intrinsic"][indices]
    available = np.isfinite(arrays["colour_intrinsic"][indices]) & np.isfinite(arrays["colour_extincted"][indices])
    if a.stage in envelope_core.M5_STAGES:
        if a.stage == "m5_no_mw":
            one, two = "mag_sel_int_1", "mag_sel_int_2"
        else:
            one, two = "mag_sel_ext_1", "mag_sel_ext_2"
        available &= np.isfinite(arrays[one][indices]) & np.isfinite(arrays[two][indices])
        available &= (arrays[one][indices] <= depths[bands[0]]) & (arrays[two][indices] <= depths[bands[1]])
    tails = parse_floats(a.tail_percentiles or envelope_core.DEFAULT_TAIL_GRID)
    for fallback in (25, 30, 35, 40, 42, 44, 45, 46, 47, 48, 49, 50):
        if fallback not in tails:
            tails.append(float(fallback))
    seed_offset = envelope_core.STAGE_BOOTSTRAP_SEED_OFFSETS[a.stage]
    rng = np.random.default_rng(a.builder_seed + 1000 * seed_offset)
    count = 0
    low_q = high_q = np.nan
    n_train_last = np.nan
    for b, _ in enumerate(centres):
        values = colours[train & available[:, b], b].astype(float)
        if len(values) < a.minimum_events:
            break
        _, selected = percentile_diagnostics(
            values, tails, getattr(a, "bootstrap", True), a.n_boot,
            a.bootstrap_threshold_mag,
            a.minimum_tail_events, rng,
        )
        if selected is None:
            break
        count += 1
        n_train_last = float(len(values))
        low_q = float(selected["low_percentile"])
        high_q = float(selected["high_percentile"])
    if count == 0:
        return np.nan, 0, np.nan, np.nan, np.nan
    return float(edges[count]), count, low_q, high_q, n_train_last


def configuration(a: argparse.Namespace, root: Path, sizes: list[int]) -> dict[str, object]:
    return {
        "method": (
            "single_load_in_memory_exact_builder_bootstrap_early_stop"
            if getattr(a, "bootstrap", True) else
            "single_load_in_memory_tail_support_early_stop_no_bootstrap"
        ),
        "run_dir": str(root), "color_pair": a.color_pair,
        "photometric_system": a.photometric_system, "stage": a.stage,
        "sample_sizes": sizes, "repetitions": a.repetitions,
        "subsample_seed": a.subsample_seed, "builder_seed": a.builder_seed,
        "t_min": a.t_min, "t_max": a.t_max, "bin_width": a.bin_width,
        "bootstrap": getattr(a, "bootstrap", True),
        "n_boot": a.n_boot, "minimum_events": a.minimum_events,
        "minimum_tail_events": a.minimum_tail_events,
        "bootstrap_threshold_mag": a.bootstrap_threshold_mag,
        "train_fraction": a.train_fraction,
        "validation_fraction": a.validation_fraction,
        "tail_percentiles": a.tail_percentiles,
        "m5_scenario": a.m5_scenario, "threshold_scope": a.threshold_scope,
        "tolerance_days": a.tolerance_days,
        "required_fraction": a.required_fraction,
    }


def main() -> None:
    a = interactive_configuration(default_configuration())
    root, sizes, bands, edges, centres = validate(a)
    if a.stability_metric == "colour":
        run_multi_percentile_convergence(a, root, sizes, bands, edges, centres)
        return
    if a.stability_metric == "legacy-colour":
        run_colour_convergence(a, root, sizes, bands, edges, centres)
        return
    output = (
        a.output_dir.expanduser().resolve()
        if a.output_dir else
        root / "analysis" /
        f"convergence_fast_{a.color_pair.replace('-', '_')}_"
        f"{a.photometric_system}_{a.stage}"
        f"{'_no_bootstrap' if not getattr(a, 'bootstrap', True) else ''}"
    )
    output.mkdir(parents=True, exist_ok=True)
    conf = configuration(a, root, sizes)
    conf_path = output / "fast_configuration.json"
    if a.resume and conf_path.exists() and json.loads(conf_path.read_text()) != conf:
        raise ValueError(f"Output directory has a different fast-run configuration: {output}")
    conf_path.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")

    bargs = make_builder_args(a, root)
    depths = get_depths(a, bargs, root, bands)
    metadata, event_ids, arrays = load_population_once(a, root, bands, centres, sizes)
    position = {event_id: i for i, event_id in enumerate(event_ids)}
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    total = len(sizes) * a.repetitions
    counter = 0
    for rep in range(a.repetitions):
        subset_seed = a.subsample_seed + rep
        for size in sizes:
            counter += 1
            checkpoint = checkpoint_dir / f"rep_{rep:03d}_N_{size:09d}.json"
            if a.resume and checkpoint.is_file():
                row = json.loads(checkpoint.read_text(encoding="utf-8"))
                records.append(row)
                print(f"[{counter}/{total}] reusing repetition={rep + 1}, N={size:,}")
                continue
            print(f"[{counter}/{total}] repetition={rep + 1}/{a.repetitions}, N={size:,}")
            sample = select_loaded_metadata_subset(metadata, size, subset_seed)
            indices = np.asarray([position[eid] for eid in sample.event_id.astype(str)], dtype=int)
            edge, n_bins, low_q, high_q, n_train = last_adaptive_edge(
                a, sample, arrays, indices, depths, bands, edges, centres
            )
            row = {
                "repetition": rep, "subsample_seed": subset_seed,
                "N_requested": size, "N_input_events": len(sample),
                "N_train_last_bin": n_train, "stage": a.stage,
                "last_valid_time_days": edge,
                "N_consecutive_valid_bins": n_bins,
                "last_low_percentile": low_q, "last_high_percentile": high_q,
                "build_output": "in_memory_fast_path",
            }
            checkpoint.write_text(json.dumps(row, allow_nan=True) + "\n", encoding="utf-8")
            records.append(row)

    raw = pd.DataFrame(records)
    raw, summary, reference, threshold = aggregate_results(
        raw, a.tolerance_days, a.required_fraction
    )
    raw_path = output / "envelope_population_convergence_raw.csv"
    summary_path = output / "envelope_population_convergence_summary.csv"
    plot_path = output / "envelope_population_png"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_plot(raw, summary, reference, threshold, a, plot_path)
    result = dict(conf)
    result.update({
        "last_bin_definition": "last initial consecutive valid envelope-bin upper edge",
        "reference": "last-bin time at largest tested population, paired by repetition",
        "stable_population_threshold": threshold,
        "largest_N_reference_median_days": reference,
        "stability_rule": "first tested N where median and required fraction of repetitions are within tolerance, and all larger tested N also pass",
    })
    (output / "envelope_population_convergence_configuration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("Reference last edge:", reference, "days")
    print("Stable population threshold:", f"{threshold:,} KNe" if threshold is not None else "not determined")
    print("Plot:", plot_path)
    print("Summary:", summary_path)
    print("Raw repetitions:", raw_path)


if __name__ == "__main__":
    main()
