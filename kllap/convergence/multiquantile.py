"""Fixed-percentile and temporal-extent convergence for KNe colour envelopes.

This module is called by :mod:`analyze_envelope_population_convergence_fast`.
It deliberately separates two questions:

* are the percentile *values* stable to a requested magnitude tolerance on a
  fixed time interval (the envelope shape)?
* how far in time is each percentile family statistically usable (the
  envelope extent)?

The largest requested population is the reference.  Every selected event is
used: this analysis deliberately performs no train/validation/test split.
Only three predeclared fixed-percentile families are evaluated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kllap.convergence.colour import colour_errors, safe_quantile
import kllap.envelope.core as envelope_core


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

        diagnostics, _ = envelope_core.percentile_diagnostics(
            values,
            diagnostic_lower,
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
            "both reference and sample must satisfy tail and bootstrap support"
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
    source_files = {
        "convergence/multiquantile.py": Path(__file__),
        "convergence/command.py": Path(fast.__file__),
        "envelope/command.py": Path(builder.__file__),
        "envelope/core.py": Path(envelope_core.__file__),
        "convergence/common.py": Path(common.__file__),
        "input/summary.csv": root / "summary.csv",
    }
    conf["source_hashes"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_files.items()
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
    import kllap.colour.common as common

    if a.tolerance_mode != "absolute":
        raise ValueError(
            "The multi-percentile primary analysis requires "
            "--tolerance-mode absolute"
        )
    if not np.isfinite(a.absolute_tolerance_mag) or a.absolute_tolerance_mag <= 0:
        raise ValueError("absolute-tolerance-mag must be finite and positive")
    if a.minimum_events < 1 or a.minimum_tail_events < 0:
        raise ValueError("Invalid event-count validity limits")
    if a.bootstrap_threshold_mag < 0:
        raise ValueError("bootstrap-threshold-mag cannot be negative")
    output = a.output_dir.expanduser().resolve() if a.output_dir else (
        root / "analysis" /
        f"convergence_multiquantile_{a.color_pair.replace('-', '_')}_"
        f"{a.photometric_system}_{a.stage}"
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
