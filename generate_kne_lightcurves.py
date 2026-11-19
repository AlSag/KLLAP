"""Generate synthetic LSST/PS1/ZTF kilonova light curves with FIESTA.

This is a self-contained scientific workflow.  Its sections explicitly group:

1. deterministic calculations and table transformations;
2. diagnostic figure generation;
3. FIESTA model backends and their numerical conventions;
4. the user-friendly terminal interface;
5. run orchestration and output writing.

Run ``python generate_kne_lightcurves.py`` and answer the terminal questions.
The public terminal interface intentionally has no long command-line option
surface.  Code that needs automation can construct an ``argparse.Namespace``
from ``default_runtime_configuration()`` and call ``run_generation()``
directly, without a subprocess or hidden runtime file.

The supported models are ``Bu2026_MLP``, ``Bu2019_MLP``, ``MetzgerModel`` and
``MetzgerFullModel``.  Every production run records physical choices, random
seeds, distance treatment, OpSim context, storage settings and model metadata.
Generated products include ``summary.csv``, ``run_info.txt``, performance
metadata, and either per-event CSV files or sharded Parquet populations.

For synthetic curves in BOOM format, choose LSST,
"Synthetic OpSim RA/Dec/MJD only", and enable "Additional BOOM CSVs".
This writes ``boom_input/lightcurve_0000.csv`` etc. and ``summary.csv`` even
when scientific storage is Parquet only. Two experiments are available:
complete exports keep every finite model point and apply no depth selection;
m5-limited exports keep only detections (mag <= m5). Both modes calculate
uncertainties with the Ivezić model using the configured m5 values.
Events without detections have no BOOM CSV or BOOM summary row; they remain in
the scientific population and root summary. The default population target is a
fixed number of injected physical KNe, which preserves the parent population and
measures m5 visibility. An optional conditional benchmark instead injects extra
events until a requested number of non-empty BOOM light curves is reached. BOOM
files are renumbered contiguously from zero, while the root summary preserves each
original injection index. CSV sample size does not restrict the BOOM export.
mag_err ALWAYS means a magnitude uncertainty in these CSVs.
Both exports use configurable fixed m5 values in the Ivezić error model. The
complete export does not use m5 as a detection threshold. Neither mode adds
random noise, and both use the synthetic cadence rather than OpSim visits.
Finite synthetic points fainter than m5 stay in the scientific CSV/Parquet,
where build_color_envelopes.py can classify them as upper limits for analysis.
Enabling this export sets synthetic_depth_cut=none to preserve those points.
Invalid model points are never interpreted as non-detections.
BOOM keeps one point per time_mjd: the default 30-second staggering of
simultaneous bands prevents those points from being discarded.
For an exceptionally short interval at the edge of the resampled model grid,
the step is reduced locally so that it cannot cross the next synthetic epoch.
Use 0 only when preserving model timestamps for a consumer that supports
simultaneous bands.
When plots are enabled, ``png/lightcurve_boom_0000.png`` etc. show the exported
detections with magnitude errors. The synthetic and
numerical-floor plots remain model diagnostics.
To plot an existing export without rerunning FIESTA:
``python generate_kne_lightcurves.py --plot-boom-input runs/RUN/boom_input``
The optional ``--max-events 20`` limits this plotting-only command (0 = all).

The scientific environment must provide NumPy, pandas, Matplotlib, JAX,
Astropy, ``rubin_sim`` and FIESTA.  OpSim products are prepared by running
``python get_opsim_baseline.py`` first.
"""

# Provenance: this analysis-specific generator was developed from FIESTA's
# MIT-licensed ``examples/generate_lightcurves.py``. The immutable upstream
# reference used for the lineage audit is:
# https://github.com/nuclear-multimessenger-astronomy/fiestaEM/blob/7bec6b39392a47f6e937f8f7a237b8742bc334ae/examples/generate_lightcurves.py
# See THIRD_PARTY_NOTICES.md and CITATION.cff for license and citation details.


# =============================================================================
# IMPORTS
# =============================================================================

# --- PRNG ---

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict

import numpy as np
try:
    import jax
except ModuleNotFoundError:  # Available in the scientific runtime, not needed for inspection.
    jax = None
import secrets

# --- FIESTA ---

try:
    from fiesta.inference.prior import Uniform
    try:
        # fiestaEM <= 0.2.x
        from fiesta.inference.prior_dict import ConstrainedPrior
    except ImportError:
        # fiestaEM >= 0.3.0
        from fiesta.inference.prior import ConstrainedPrior
except ModuleNotFoundError:  # Keeps the file importable for review and light tests.
    Uniform = ConstrainedPrior = None

# --- COSMOLOGY FOR DISTANCE SAMPLING ---
# Used only to draw events uniformly in comoving volume.
# Bulla/FIESTA still receives luminosity_distance and redshift as input parameters.
try:
    from astropy.cosmology import Planck18 as COSMO
    from astropy.cosmology import z_at_value
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except ModuleNotFoundError:
    COSMO = z_at_value = SkyCoord = u = None

# --- DATAFRAME TOOL ---

import pandas as pd

# --- OUTPUT DIRECTORIES ---

from pathlib import Path
from datetime import datetime

# --- RUBIN SIM (LSST photometric model) ---

try:
    from rubin_sim.phot_utils import (
        PhotometricParameters,
        calc_mag_error_m5,
        calc_snr_m5,
        rubin_bandpasses,
    )
except ModuleNotFoundError:
    PhotometricParameters = None
    calc_mag_error_m5 = calc_snr_m5 = rubin_bandpasses = None

from kne_pipeline_common import (
    add_lsstlike_candidate_ids,
    build_synthetic_measurement_table,
    draw_posterior_rows,
    enrich_visits_with_field_index_metadata,
    format_run_info_sections,
    interpolate_magnitude_log_time,
    load_posterior_samples,
    load_opsim_run_metadata,
    normalize_opsim_visit_columns,
    prior_bounds_dict,
    prompt_choice,
    prompt_text,
    resample_synthetic_output_grid,
    slugify,
)

# ============================================================================
# SCIENTIFIC CALCULATIONS AND TABLE TRANSFORMATIONS
# ============================================================================


RUBIN_BANDS = tuple("ugrizy")


def summarize_field(v: pd.DataFrame) -> dict[str, object]:
    """Read sky coordinates and target name from the selected field."""
    summary = {}
    for column in ("fieldRA", "fieldDec"):
        values = pd.to_numeric(
            v.get(column, pd.Series(np.nan, index=v.index)), errors="coerce"
        ).to_numpy(float)
        summary[column] = (
            float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan
        )
    target_values = v.get("target_name", pd.Series("", index=v.index))
    target_values = target_values.dropna().astype(str)
    summary["target_name"] = target_values.iloc[0] if len(target_values) else ""
    return summary


def _trim_lsstlike_after_last_detection(
    observations: pd.DataFrame,
    days_after_last_detection: float = 10.0,
) -> tuple[pd.DataFrame, float, float, bool]:
    """Keep valid visits through a fixed time after the last detection."""
    if days_after_last_detection < 0:
        raise ValueError("days_after_last_detection must be non-negative")

    out = observations.loc[
        np.isfinite(pd.to_numeric(observations["mag_true"], errors="coerce"))
        & np.isfinite(pd.to_numeric(observations["t_days"], errors="coerce"))
    ].copy()
    out = out.sort_values("observationStartMJD").reset_index(drop=True)

    detected = out["detected"].astype(bool)
    if not detected.any():
        return out, np.nan, np.nan, False

    last_detection = float(out.loc[detected, "t_days"].iloc[-1])
    window_end = last_detection + float(days_after_last_detection)
    out = out.loc[out["t_days"] <= window_end].reset_index(drop=True)
    return out, last_detection, window_end, True


def _apply_synthetic_m5_median_cut(
    measurements: pd.DataFrame,
    objectid: str,
    m5_medians: dict[str, float],
) -> pd.DataFrame:
    """Keep finite synthetic points at or brighter than their band's m5."""
    out = measurements.copy()
    magnitudes = pd.to_numeric(out["mag"], errors="coerce").to_numpy(float)
    thresholds = pd.to_numeric(
        out["band"].map(m5_medians), errors="coerce"
    ).to_numpy(float)
    keep = (
        np.isfinite(magnitudes)
        & np.isfinite(thresholds)
        & (magnitudes <= thresholds)
    )
    out = out.loc[keep].reset_index(drop=True)

    out["objectid"] = ""
    if len(out):
        out.at[0, "objectid"] = str(objectid)
    out["candidate_id"] = [
        f"{objectid}_{index}" for index in range(1, len(out) + 1)
    ]
    return out


def _derive_sky_seed(
    intrinsic_seed: int,
    latitude_mode: str,
    threshold_deg: float,
    latitude_max_deg: float | None = None,
) -> int:
    """Derive a reproducible sky seed for one latitude selection."""
    role_code = {"all": 0xA11, "low": 0x10A, "high": 0xA19, "bin": 0xB17}[
        latitude_mode
    ]
    threshold_code = int(round(float(threshold_deg) * 1000.0))
    seed_value = int(intrinsic_seed) ^ 0x534B59 ^ role_code ^ threshold_code
    if latitude_mode == "bin":
        if latitude_max_deg is None:
            raise ValueError("latitude_max_deg is required for latitude_mode='bin'")
        maximum_code = int(round(float(latitude_max_deg) * 1000.0))
        seed_value ^= (maximum_code * 0x9E3779B1) & 0xFFFFFFFF
    return int(seed_value % (2**32))

# ============================================================================
# FIGURE GENERATION
# ============================================================================

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "DejaVu Sans"


def plot_boom_measurements(measurements, outfile):
    """Plot BOOM detections with magnitude errors; omit legacy blank-mag rows."""
    required = {"t_days", "band", "mag", "mag_err"}
    missing = required - set(measurements.columns)
    if missing:
        raise ValueError(f"Cannot plot BOOM CSV: missing columns {sorted(missing)}")
    table = measurements.copy()
    for name in ("t_days", "mag", "mag_err"):
        table[name] = pd.to_numeric(table[name], errors="raise")
    table = table.loc[np.isfinite(table["t_days"]) & np.isfinite(table["mag_err"])
                      & np.isfinite(table["mag"]) & (table["mag_err"] >= 0)]
    if table.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    handles = []
    has_detection = False
    for band in "ugrizy":
        rows = table.loc[table["band"] == band].sort_values("t_days")
        color = BASE_BAND_COLORS[band]
        detections = rows.loc[np.isfinite(rows["mag"]) & (rows["mag_err"] >= 0)]
        if detections.empty:
            continue
        handles.append(Line2D([], [], color=color, marker="o", linestyle="none", label=f"LSST {band}"))
        if not detections.empty:
            ax.errorbar(detections["t_days"], detections["mag"], yerr=detections["mag_err"],
                        fmt="o", linestyle="none", color=color, markersize=3,
                        elinewidth=0.9, capsize=2, zorder=3)
            has_detection = True
    if not handles:
        plt.close(fig)
        return False
    if has_detection:
        handles.append(Line2D([], [], color="black", marker="o", linestyle="none",
                              label="Detection (1-sigma errors)"))
    if (table["t_days"] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel("Time since merger [days]")
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    ax.margins(y=0.12)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, fontsize=9)
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return True


def replot_boom_input(input_dir, max_events=0):
    """Read existing exported CSVs; no model evaluation or modification of data."""
    if max_events < 0:
        raise ValueError("max_events must be >= 0")
    input_dir = Path(input_dir).expanduser().resolve()
    paths = sorted(input_dir.glob("lightcurve_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No lightcurve_*.csv in {input_dir}")
    plot_dir = input_dir.parent / "png" if input_dir.name == "boom_input" else input_dir / "png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in paths:
        if max_events and written >= max_events:
            break
        suffix = path.stem.removeprefix("lightcurve_")
        rows = pd.read_csv(path, float_precision="round_trip")
        written += int(plot_boom_measurements(rows, plot_dir / f"lightcurve_boom_{suffix}.png"))
    print(f"Saved {written} BOOM plots in {plot_dir}")


def _plot_synthetic_only(
    times, mag_dict, filters, filter_colors, mag_lim, outfile
):
    """Save a diagnostic plot of the dense synthetic light curves."""
    times = np.asarray(times, float)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    finite_chunks = []
    for filter_name in filters:
        magnitudes = np.asarray(mag_dict[filter_name], float).copy()
        magnitudes[magnitudes > mag_lim] = np.nan
        ax.plot(
            times,
            magnitudes,
            linewidth=2.6,
            color=filter_colors.get(filter_name),
            label=filter_name,
        )
        finite = magnitudes[np.isfinite(magnitudes)]
        if finite.size:
            finite_chunks.append(finite)

    ax.axhline(
        mag_lim,
        linestyle="--",
        linewidth=1.1,
        alpha=0.7,
        color="gray",
        label=f"lim {mag_lim:g} mag",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Time [days] (observer-frame)")
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    if not finite_chunks:
        plt.close(fig)
        return
    all_magnitudes = np.concatenate(finite_chunks)
    ax.set_ylim(mag_lim + 0.2, np.nanmin(all_magnitudes) - 0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        framealpha=0.95,
        title="Filters (synthetic)",
    )
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_saved_synthetic_measurements(
    measurements,
    filter_colors,
    outfile,
    event_index,
    filter_systems,
    filter_by_system,
):
    """Plot the final synthetic table exactly as it is written to storage."""
    required = {"t_days", "band", "mag"}
    missing = required.difference(measurements.columns)
    if missing:
        raise ValueError(
            f"Cannot plot synthetic event {event_index}: missing columns "
            f"{sorted(missing)}"
        )

    table = measurements.copy()
    table["t_days"] = pd.to_numeric(table["t_days"], errors="coerce")
    table["mag"] = pd.to_numeric(table["mag"], errors="coerce")
    table = table.loc[
        np.isfinite(table["t_days"])
        & (table["t_days"] > 0)
        & np.isfinite(table["mag"])
    ].copy()
    if table.empty:
        return False

    if "photometric_system" not in table:
        table["photometric_system"] = "lsst"
    table["photometric_system"] = (
        table["photometric_system"].astype(str).str.lower()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    plotted = False
    for system in filter_systems:
        for band, filter_name in filter_by_system[system].items():
            subset = table.loc[
                (table["photometric_system"] == system)
                & (table["band"].astype(str) == band)
            ]
            if subset.empty:
                continue
            subset = subset.sort_values("t_days")
            ax.plot(
                subset["t_days"],
                subset["mag"],
                marker="o",
                markersize=2.2,
                linewidth=1.6,
                linestyle="-" if system == "lsst" else "--",
                color=filter_colors.get(filter_name),
                label=f"{system.upper()} {band}",
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xscale("log")
    ax.set_xlabel("Time since merger [days]")
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(title="Photometric band", ncol=2, frameon=True, framealpha=0.95)
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return True


def _plot_numerical_floor_diagnostic(
    times,
    raw_magnitudes_by_filter,
    filters,
    filter_colors,
    apparent_cut_mag,
    outfile,
):
    """Show the full model prediction and the distance-dependent cut."""
    times = np.asarray(times, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    finite_for_limits = []

    for filter_name in filters:
        raw = np.asarray(raw_magnitudes_by_filter[filter_name], dtype=float)
        valid = np.isfinite(times) & (times > 0) & np.isfinite(raw)
        if not np.any(valid):
            continue
        finite_for_limits.append(raw[valid])
        indices = np.flatnonzero(valid)
        crossing = indices[raw[indices] >= float(apparent_cut_mag)]
        color = filter_colors.get(filter_name)
        band_label = filter_name.replace("lsst", "")

        if crossing.size == 0:
            ax.plot(
                times[valid], raw[valid], color=color, linewidth=2.1,
                label=band_label,
            )
            continue

        first = int(crossing[0])
        retained = valid & (np.arange(len(times)) < first)
        if np.any(retained):
            ax.plot(
                times[retained], raw[retained], color=color, linewidth=2.1,
                label=band_label,
            )
        rejected = valid & (np.arange(len(times)) >= max(first - 1, 0))
        ax.plot(
            times[rejected],
            raw[rejected],
            color=color,
            linewidth=1.7,
            linestyle=":",
            alpha=0.9,
            label=band_label if not np.any(retained) else None,
        )

    ax.axhline(
        float(apparent_cut_mag),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Numerical-support cut ({apparent_cut_mag:.2f} mag)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Time since merger [days]")
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if finite_for_limits:
        values = np.concatenate(finite_for_limits)
        ax.set_ylim(np.nanmax(values) + 0.3, np.nanmin(values) - 0.3)
    ax.legend(title="LSST band", ncol=2, frameon=True, framealpha=0.95)
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_synthetic_and_lsstlike(
    times,
    mag_dict,
    observations,
    filters,
    filter_colors,
    mag_lim,
    outfile,
    band_to_filter,
):
    """Overlay dense model curves, detections, and visit upper limits."""
    times = np.asarray(times, float)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    for filter_name in filters:
        magnitudes = np.asarray(mag_dict[filter_name], float).copy()
        magnitudes[magnitudes > mag_lim] = np.nan
        ax.plot(
            times,
            magnitudes,
            linewidth=2.6,
            color=filter_colors.get(filter_name),
        )

    table = observations.copy()
    table = table[np.isfinite(table["mag_true"])]
    positive_times = times[times > 0]
    table = table[
        (table["t_days"] >= np.nanmin(positive_times))
        & (table["t_days"] <= np.nanmax(times))
    ].copy()
    table["t_plot"] = np.where(
        table["t_days"].to_numpy() <= 0, 1e-3, table["t_days"].to_numpy()
    )

    for band in sorted(table["band"].unique()):
        filter_name = band_to_filter.get(band)
        if filter_name is None or filter_name not in filters:
            continue
        color = filter_colors.get(filter_name)
        subset = table[table["band"] == band]
        detections = subset[
            subset["detected"]
            & np.isfinite(subset["mag_obs"])
            & np.isfinite(subset["mag_err"])
        ]
        upper_limits = subset[
            (~subset["detected"]) & np.isfinite(subset["mag_ulim"])
        ]
        if len(detections):
            ax.errorbar(
                detections["t_plot"],
                detections["mag_obs"],
                yerr=detections["mag_err"],
                fmt="o",
                linestyle="none",
                markersize=4,
                elinewidth=1.0,
                capsize=0,
                color=color,
            )
        if len(upper_limits):
            ax.scatter(
                upper_limits["t_plot"],
                upper_limits["mag_ulim"],
                marker="v",
                s=28,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
            )

    ax.axhline(
        mag_lim, linestyle="--", linewidth=1.1, alpha=0.7, color="gray",
        label=f"lim {mag_lim:g} mag",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Time since trigger [days]")
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_xlim(np.nanmin(positive_times), np.nanmax(times))

    visible = []
    for filter_name in filters:
        magnitudes = np.asarray(mag_dict[filter_name], float).copy()
        magnitudes[magnitudes > mag_lim] = np.nan
        if np.any(np.isfinite(magnitudes)):
            visible.append(magnitudes[np.isfinite(magnitudes)])
    for column in ("mag_obs", "mag_ulim"):
        values = table[column].to_numpy(float)
        if np.any(np.isfinite(values)):
            visible.append(values[np.isfinite(values)])
    if visible:
        all_magnitudes = np.concatenate(visible)
        ax.set_ylim(mag_lim + 0.2, np.nanmin(all_magnitudes) - 0.3)

    handles, labels = [], []
    for filter_name in filters:
        color = filter_colors.get(filter_name)
        handles.append(Line2D([0], [0], color=color, lw=2.6))
        labels.append(f"{filter_name} (synthetic)")
    for filter_name in filters:
        color = filter_colors.get(filter_name)
        handles.append(
            Line2D(
                [0], [0], marker="v", color=color, lw=0, markersize=6,
                markerfacecolor="none", markeredgewidth=1.2,
            )
        )
        labels.append(f"{filter_name} (upper-limit)")
    for filter_name in filters:
        color = filter_colors.get(filter_name)
        handles.append(
            Line2D([0], [0], marker="o", color=color, lw=0, markersize=5)
        )
        labels.append(f"{filter_name} (obs)")
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        framealpha=0.95,
    )
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_lsstlike_only(
    observations,
    filters,
    filter_colors,
    event_index,
    outfile,
    band_to_filter,
    tmax_days=None,
):
    """Plot visit detections and upper limits without dense model curves."""
    del event_index  # retained in the public signature for diagnostic context
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    table = observations[
        np.isfinite(observations["mag_true"]) & (observations["t_days"] > 0)
    ].copy()
    if tmax_days is not None:
        table = table[table["t_days"] <= tmax_days]
    if table.empty:
        plt.close(fig)
        return

    for band in sorted(table["band"].unique()):
        filter_name = band_to_filter.get(band)
        if filter_name is None or filter_name not in filters:
            continue
        subset = table[table["band"] == band]
        detections = subset[
            subset["detected"]
            & np.isfinite(subset["mag_obs"])
            & np.isfinite(subset["mag_err"])
        ]
        upper_limits = subset[
            (~subset["detected"]) & np.isfinite(subset["mag_ulim"])
        ]
        color = filter_colors.get(filter_name)
        if len(detections):
            ax.errorbar(
                detections["t_days"],
                detections["mag_obs"],
                yerr=detections["mag_err"],
                fmt="o",
                linestyle="none",
                markersize=4,
                elinewidth=1.0,
                capsize=0,
                color=color,
            )
        if len(upper_limits):
            ax.scatter(
                upper_limits["t_days"],
                upper_limits["mag_ulim"],
                marker="v",
                s=28,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Time since trigger [days]")
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    visible = []
    for column in ("mag_obs", "mag_ulim"):
        values = table[column].to_numpy(float)
        if np.any(np.isfinite(values)):
            visible.append(values[np.isfinite(values)])
    if visible:
        magnitudes = np.concatenate(visible)
        ax.set_ylim(np.max(magnitudes) + 0.2, np.min(magnitudes) - 0.3)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    handles, labels = [], []
    for band in sorted(table["band"].unique()):
        filter_name = band_to_filter.get(band)
        if filter_name is None or filter_name not in filters:
            continue
        subset = table[table["band"] == band]
        color = filter_colors.get(filter_name)
        has_detection = np.any(
            subset["detected"]
            & np.isfinite(subset["mag_obs"])
            & np.isfinite(subset["mag_err"])
        )
        has_upper_limit = np.any(
            (~subset["detected"]) & np.isfinite(subset["mag_ulim"])
        )
        if has_detection:
            handles.append(
                Line2D(
                    [0], [0], marker="o", linestyle="none", color=color,
                    markerfacecolor=color, markeredgecolor=color, markersize=5,
                )
            )
            labels.append(filter_name)
        if has_upper_limit:
            handles.append(
                Line2D(
                    [0], [0], marker="v", linestyle="none", color=color,
                    markerfacecolor="none", markeredgecolor=color, markersize=6,
                )
            )
            labels.append(f"ul_{filter_name}")
    if labels:
        ax.legend(
            handles,
            labels,
            title="Filters",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            framealpha=0.95,
        )
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)

# ============================================================================
# LIGHT-CURVE MODEL BACKENDS
# ============================================================================

import importlib
import inspect
import math
from importlib.metadata import PackageNotFoundError, version


MODEL_NAMES = ("Bu2026_MLP", "Bu2019_MLP", "MetzgerModel", "MetzgerFullModel")
ANALYTICAL_MODELS = frozenset(("MetzgerModel", "MetzgerFullModel"))
METZGER_PARAMETERS = ("log10_mej", "log10_vej", "beta", "log10_kappa_r")
AUDITED_FIESTA_COMMIT = "6ba093cc13cb768a089b07768e550a6a46b737ca"


def metzger_prior_bounds(args):
    """Return chosen simulation bounds, not surrogate training limits."""
    return dict(zip(METZGER_PARAMETERS, (
        tuple(args.metzger_log10_mej_range), tuple(args.metzger_log10_vej_range),
        tuple(args.metzger_beta_range), tuple(args.metzger_log10_kappa_range),
    )))


def validate_model_options(args):
    if args.model not in MODEL_NAMES:
        raise ValueError(f"Unsupported FIESTA model: {args.model}")
    if args.model != "Bu2026_MLP" and args.parameter_source != "broad":
        raise ValueError(
            f"{args.model} requires --parameter-source broad. The supported "
            "Bu2026/AT2017gfo posterior has a different parameterization."
        )
    if args.model not in ANALYTICAL_MODELS:
        return
    if args.fiesta_surrogate_dir is not None:
        raise ValueError("A Metzger model needs no --fiesta-surrogate-dir or surrogate weights")
    for name, (lower, upper) in metzger_prior_bounds(args).items():
        if not all(math.isfinite(v) for v in (lower, upper)) or lower >= upper:
            raise ValueError(f"{name} requires finite MIN < MAX")
    if args.metzger_beta_range[0] <= 0:
        raise ValueError("Metzger beta must be positive")
    if args.metzger_log10_vej_range[1] >= 0:
        raise ValueError("Metzger velocities must be below c (log10_vej < 0)")
    if not (math.isfinite(args.analytical_t_min_days)
            and math.isfinite(args.analytical_t_max_days)
            and 0 < args.analytical_t_min_days < args.analytical_t_max_days):
        raise ValueError("Analytical integration requires 0 < t_min < t_max (finite, source days)")
    if args.analytical_n_times < 2:
        raise ValueError("--analytical-n-times must be at least 2")
    if args.model == "MetzgerFullModel":
        if not (0 < args.metzger_vmax < 1):
            raise ValueError("--metzger-vmax must be between 0 and 1")
        if args.metzger_log10_vej_range[1] >= math.log10(args.metzger_vmax):
            raise ValueError("MetzgerFullModel requires every v_min < vmax")


def model_configuration(args):
    """Configuration also stored in each new Metzger intrinsic catalogue."""
    if args.model not in ANALYTICAL_MODELS:
        return {}
    result = {
        "source_time_min_days": args.analytical_t_min_days,
        "source_time_max_days": args.analytical_t_max_days,
        "source_time_points": args.analytical_n_times,
        "source_time_spacing": "logarithmic",
        "inclination_used_by_model": False,
        "population_prior_bounds": {k: list(v) for k, v in metzger_prior_bounds(args).items()},
    }
    if args.model == "MetzgerFullModel":
        result.update(vmax_fraction_c=args.metzger_vmax,
                      neutron_precursor=args.metzger_neutron_precursor)
    return result


def create_fiesta_model(args, filters):
    """Instantiate only the selected backend; return model and metadata."""
    validate_model_options(args)
    try:
        installed_version = version("fiestaEM")
    except PackageNotFoundError:
        installed_version = "unavailable (source checkout?)"
    metadata = {"fiesta_version": installed_version}
    if args.model in ANALYTICAL_MODELS:
        # Explicit energy evolution benefits from double precision; retain the
        # legacy surrogate precision when a surrogate is selected instead.
        import jax
        jax.config.update("jax_enable_x64", True)
        try:
            module = importlib.import_module("fiesta.models.analytical_models")
            model_class = getattr(module, args.model)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"The installed FIESTA does not expose {args.model}. See "
                "docs/metzger_generation.md for an isolated, pinned environment."
            ) from exc
        import jax.numpy as jnp
        times = jnp.geomspace(args.analytical_t_min_days, args.analytical_t_max_days,
                             args.analytical_n_times)
        kwargs = {"filters": list(filters), "times": times}
        if args.model == "MetzgerFullModel":
            kwargs.update(neutron_precursor=args.metzger_neutron_precursor, vmax=args.metzger_vmax)
        model = model_class(**kwargs)
        if tuple(model.parameter_names) != METZGER_PARAMETERS:
            raise RuntimeError("FIESTA Metzger parameterization differs from the audited implementation")
        metadata.update(backend="FIESTA semi-analytical", api="AnalyticalModel",
                        implementation_file=inspect.getfile(model_class),
                        implementation_sha256=hashlib.sha256(
                            Path(inspect.getfile(model_class)).read_bytes()).hexdigest(),
                        audited_fiesta_commit=AUDITED_FIESTA_COMMIT,
                        configuration=model_configuration(args),
                        jax_enable_x64=True,
                        prediction_time_frame="source; generator multiplies by (1+z)")
    else:
        try:
            model_class = getattr(importlib.import_module("fiesta.inference.lightcurve_model"), "BullaFlux")
            api = "legacy BullaFlux"
        except (ImportError, AttributeError):
            model_class = getattr(importlib.import_module("fiesta.models"), "FluxSurrogate")
            api = "FluxSurrogate"
        directory = str(args.fiesta_surrogate_dir.expanduser().resolve()) if args.fiesta_surrogate_dir else None
        model = model_class(name=args.model, filters=list(filters), directory=directory)
        metadata.update(backend="FIESTA surrogate", api=api,
                        prediction_time_frame="observer (FIESTA redshift conversion)")
    return model, metadata


def saved_observer_times(times, params, model_name):
    """Analytical FIESTA returns source times; flux surrogates return observer times."""
    times = np.asarray(times, dtype=float)
    if model_name in ANALYTICAL_MODELS:
        z = float(params["redshift"])
        if not math.isfinite(z) or z < 0:
            raise ValueError("redshift must be finite and non-negative")
        return times * (1.0 + z)
    return times

# ============================================================================
# BOOM EXPORT AND TERMINAL CONFIGURATION
# ============================================================================


# ----------------------------------------------------------------------------
# BOOM photometry and export
# ----------------------------------------------------------------------------

# Assumed single-visit depths, matching the user's BOOM CSV convention. These
# are configurable simulation inputs, not the actual depth of an OpSim visit.
BOOM_M5_DEFAULT = {"u": 23.7, "g": 24.8, "r": 24.4, "i": 23.9, "z": 23.3, "y": 22.3}
BOOM_GAMMA = {"u": 0.038, "g": 0.039, "r": 0.039, "i": 0.039, "z": 0.039, "y": 0.039}
BOOM_COLUMNS = ["objectid", "candidate_id", "time_mjd", "t_days", "band", "mag", "mag_err"]


def synthetic_to_boom_csv(measurements, event_index, m5_by_band,
                          sigma_sys=0.005, duplicate_shift_seconds=30.0,
                          export_mode="m5-detected"):
    """Convert ONE event's LSST model table to the supplied BOOM CSV format.

    Ivezić et al. (2019), ApJ 873, 111, equations 4-5 / Table 2:
    https://faculty.washington.edu/ivezic/Publications/Document-5462.pdf
    sigma_mag^2 = sigma_sys^2 + (0.04-gamma)*x + gamma*x^2;
    x = 10**(0.4*(mag-m5)). No random noise is drawn.

    ``export_mode='complete'`` keeps every finite model point and uses m5 only
    in the Ivezić uncertainty. ``export_mode='m5-detected'`` additionally uses
    mag <= m5 as a selection. Fainter points are excluded only in the second mode.
    mag and mag_err are always finite; mag_err is ALWAYS an uncertainty.
    If present, photometric_system selects LSST; other systems are untouched.
    BOOM keeps one point per time_mjd; staggering preserves different
    simultaneous bands. The requested step is shortened locally when necessary to
    remain strictly before the next original epoch. This is an export
    convention only: it does not represent actual OpSim visits.
    """
    if not np.isfinite(sigma_sys) or sigma_sys < 0:
        raise ValueError("sigma_sys must be finite and >= 0")
    if export_mode not in {"complete", "m5-detected"}:
        raise ValueError("export_mode must be 'complete' or 'm5-detected'")
    if not np.isfinite(duplicate_shift_seconds) or duplicate_shift_seconds < 0:
        raise ValueError("duplicate_shift_seconds must be finite and >= 0")
    if measurements.empty:
        return pd.DataFrame(columns=BOOM_COLUMNS)
    required = {"time_mjd", "t_days", "band", "mag"}
    missing = required - set(measurements.columns)
    if missing:
        raise ValueError(f"BOOM synthetic export: missing columns {sorted(missing)}")
    rows = measurements.copy()
    if "photometric_system" in rows:
        rows = rows.loc[rows["photometric_system"] == "lsst"].copy()
    for name in ("time_mjd", "t_days", "mag"):
        rows[name] = pd.to_numeric(rows[name], errors="raise")
    rows = rows.loc[np.isfinite(rows[["time_mjd", "t_days", "mag"]]).all(axis=1)]
    rows = rows.sort_values("time_mjd", kind="stable").reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame(columns=BOOM_COLUMNS)
    depths = rows["band"].map(m5_by_band)
    gamma = rows["band"].map(BOOM_GAMMA)
    if not np.isfinite(depths).all() or gamma.isna().any():
        raise ValueError(
            "BOOM export with Ivezić errors requires finite m5 values for all "
            "LSST bands used"
        )
    if export_mode == "complete":
        out = rows[["time_mjd", "t_days", "band", "mag"]].copy()
    else:
        detected = rows["mag"] <= depths
        if not detected.any():
            return pd.DataFrame(columns=BOOM_COLUMNS)
        # Work only on rows that BOOM will receive. Faint rows must not create
        # a false timestamp collision before they are discarded.
        out = rows.loc[
            detected, ["time_mjd", "t_days", "band", "mag"]
        ].copy()
    out_depths = out["band"].map(m5_by_band).astype(float)
    out_gamma = out["band"].map(BOOM_GAMMA).astype(float)
    x = 10.0 ** (0.4 * (out["mag"] - out_depths))
    out["mag_err"] = np.sqrt(
        sigma_sys**2 + (0.04-out_gamma)*x + out_gamma*x*x
    )

    if duplicate_shift_seconds > 0:
        original_times = out["time_mjd"].to_numpy(dtype=float)
        adjusted_times = original_times.copy()
        offsets = np.zeros(len(out), dtype=float)
        unique_times, starts, counts = np.unique(
            original_times, return_index=True, return_counts=True
        )
        requested_step_days = duplicate_shift_seconds / 86400.0
        for group_index, (start, count) in enumerate(zip(starts, counts)):
            if count <= 1:
                continue
            step_days = requested_step_days
            if group_index + 1 < len(unique_times):
                gap_days = unique_times[group_index + 1] - unique_times[group_index]
                # Dividing by count, rather than count-1, leaves a positive
                # margin before the following unshifted epoch.
                step_days = min(step_days, gap_days / count)
            group_offsets = np.arange(count, dtype=float) * step_days
            stop = start + count
            adjusted_times[start:stop] += group_offsets
            offsets[start:stop] = group_offsets
        out["time_mjd"] = adjusted_times
        out["t_days"] = out["t_days"].to_numpy(dtype=float) + offsets
        if out["time_mjd"].duplicated().any() or not out["time_mjd"].is_monotonic_increasing:
            raise RuntimeError(
                "Internal BOOM timestamp allocation failed to create unique "
                "chronological detection times"
            )

    out = out.reset_index(drop=True)
    out["objectid"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.at[0, "objectid"] = int(event_index)
    out["candidate_id"] = [f"{int(event_index):04d}_{j}" for j in range(1, len(out)+1)]
    return out[BOOM_COLUMNS]


# ----------------------------------------------------------------------------
# Terminal configuration
# ----------------------------------------------------------------------------

def default_runtime_configuration() -> argparse.Namespace:
    """Return documented defaults before the interactive questions."""
    return argparse.Namespace(
        run_name="broad_run",
        model="Bu2026_MLP",
        fiesta_surrogate_dir=None,
        runs_root=Path("runs"),
        filter_system="lsst+ps1",
        parameter_source="broad",
        posterior_file=None,
        posterior_inclination="posterior",
        opsim_dir=Path("opsim_exports_all_survey"),
        galactic_latitude_mode="all",
        galactic_latitude_threshold_deg=20.0,
        galactic_latitude_min_deg=0.0,
        galactic_latitude_max_deg=5.0,
        output_mode="detected-pairs",
        min_detections=1,
        synthetic_cadence_days=0.0,
        synthetic_schema="model",
        synthetic_depth_cut="none",
        boom_synthetic_export=False,
        boom_export_mode="complete",
        boom_target_nonempty_events=False,
        boom_max_injection_factor=20.0,
        boom_m5_by_band=dict(BOOM_M5_DEFAULT),
        boom_sigma_sys=0.005,
        boom_duplicate_shift_seconds=30.0,
        storage_format="csv",
        parquet_events_per_shard=500,
        parquet_compression="zstd",
        csv_sample_size=0,
        opsim_cache_size=32,
        surrogate_batch_size=64,
        plot_mode="none",
        max_plot_events=20,
        n_samples=80000,
        n_realizations=1,
        realization_seeds=None,
        pairing_mode="none",
        intrinsic_catalog=None,
        catalog_distance_policy="stored",
        seed=None,
        sky_seed=None,
        distance_mode="comoving-volume",
        fixed_distance_mpc=10.0,
        distance_min_mpc=100.0,
        distance_max_mpc=500.0,
        numerical_floor_absolute_mag=0.0,
        numerical_floor_margin_mag=1.0,
        metzger_log10_mej_range=(-3.0, -1.0),
        metzger_log10_vej_range=(math.log10(0.05), math.log10(0.3)),
        metzger_beta_range=(2.0, 4.0),
        metzger_log10_kappa_range=(math.log10(0.5), math.log10(30.0)),
        analytical_t_min_days=0.1,
        analytical_t_max_days=30.0,
        analytical_n_times=1000,
        metzger_vmax=0.7,
        metzger_neutron_precursor=True,
    )


def configure_runtime() -> argparse.Namespace:
    """Collect and validate a complete configuration in the terminal."""
    args = default_runtime_configuration()
    # The public workflow is deliberately interactive: scientific automation
    # calls run_generation(Namespace(...)) directly instead of invoking hidden
    # terminal flags in a subprocess.
    args.run_name = prompt_text("Readable run name", args.run_name)
    args.model = prompt_choice(
        "Kilonova light-curve model",
        [
            ("Bu2026_MLP", "FIESTA Bu2026_MLP (LSST, PS1 and ZTF)"),
            ("Bu2019_MLP", "FIESTA Bu2019_MLP (LSST, PS1 and ZTF)"),
            ("MetzgerModel", "FIESTA MetzgerModel (semi-analytical, NMMA-style)"),
            ("MetzgerFullModel", "FIESTA MetzgerFullModel (semi-analytical, Redback-style)"),
        ],
        args.model,
    )
    if args.model == "Bu2019_MLP":
        surrogate_path = prompt_text(
            "Local FIESTA surrogate directory (empty = automatic download)",
            str(args.fiesta_surrogate_dir or ""),
        ).strip()
        args.fiesta_surrogate_dir = (
            Path(surrogate_path) if surrogate_path else None
        )
    if args.model in ANALYTICAL_MODELS:
        print("Metzger population priors: uniform in the listed variables.")
        for attribute, label in (
            ("metzger_log10_mej_range", "log10(M_ej / Msun)"),
            ("metzger_log10_vej_range", "log10(v_min / c)"),
            ("metzger_beta_range", "Velocity-profile exponent beta"),
            ("metzger_log10_kappa_range", "log10(kappa_r / (cm^2/g))"),
        ):
            value = getattr(args, attribute)
            entered = prompt_text(label + " range (MIN MAX)", f"{value[0]} {value[1]}")
            limits = [float(v) for v in entered.replace(",", " ").split()]
            if len(limits) != 2:
                raise ValueError(label + " requires MIN MAX")
            setattr(args, attribute, limits)
        args.analytical_t_min_days = float(prompt_text(
            "First source-frame integration time [days]", str(args.analytical_t_min_days)))
        args.analytical_t_max_days = float(prompt_text(
            "Last source-frame integration time [days]", str(args.analytical_t_max_days)))
        args.analytical_n_times = int(prompt_text(
            "Logarithmic integration time points (not the saved cadence)", str(args.analytical_n_times)))
        if args.model == "MetzgerFullModel":
            args.metzger_vmax = float(prompt_text(
                "Maximum shell velocity [fraction of c]", str(args.metzger_vmax)))
            precursor = prompt_choice("Neutron precursor", [("yes", "Include"), ("no", "Disable")],
                                      "yes" if args.metzger_neutron_precursor else "no")
            args.metzger_neutron_precursor = precursor == "yes"
    args.parameter_source = prompt_choice(
        "Physical-parameter population",
        ([("broad", "Broad uniform parameter ranges"), ("posterior", "FIESTA posterior (.npz or AT2017gfo.tar)")]
         if args.model == "Bu2026_MLP" else [("broad", "Broad uniform parameter ranges")]),
        args.parameter_source,
    )
    if args.parameter_source == "posterior":
        args.posterior_file = Path(prompt_text("Posterior file", str(args.posterior_file or "AT2017gfo.tar")))
        args.posterior_inclination = prompt_choice(
            "Inclination treatment",
            [("posterior", "Keep inclination from each posterior row"), ("isotropic", "AT2017gfo-like ejecta, new isotropic inclination")],
            args.posterior_inclination,
        )
    args.filter_system = prompt_choice(
        "Photometric filter system",
        [
            ("lsst+ps1", "LSST ugrizy + PS1 grizy for the same events"),
            ("lsst+ztf", "LSST ugrizy + ZTF gri for the same events"),
            ("lsst", "LSST ugrizy only"),
            ("ps1", "PS1 grizy only (synthetic products; no Rubin m5)"),
            ("ztf", "ZTF gri only (synthetic products; ZTF depth applied later)"),
        ],
        args.filter_system,
    )
    if args.filter_system in {"ps1", "ztf"}:
        # Rubin OpSim visits and m5 values cannot be applied to a PS1-only
        # model evaluation. We retain the OpSim sky position, merger MJD
        # and E(B-V) metadata, but save only synthetic PS1 products.
        args.output_mode = "synthetic-mjd"
        print(
            f"{args.filter_system.upper()}-only mode: scientific product "
            "selection is fixed to 'Synthetic OpSim RA/Dec/MJD only'. "
            "Rubin LSST-like products require LSST model magnitudes."
        )
    else:
        args.output_mode = prompt_choice(
            "Scientific event-product selection",
            [
                (
                    "all-synthetic-lsstlike-detected",
                    "All synthetic + LSST-like only for detected events",
                ),
                (
                    "detected-pairs",
                    "Synthetic OpSim MJD + LSST-like, detected events only",
                ),
                (
                    "lsstlike-detected",
                    "LSST-like only, detected events only",
                ),
                (
                    "synthetic-mjd",
                    "Synthetic OpSim RA/Dec/MJD only",
                ),
            ],
            args.output_mode,
        )
    if args.output_mode in {
        "all-synthetic-lsstlike-detected",
        "detected-pairs",
        "synthetic-mjd",
    }:
        args.synthetic_cadence_days = float(
            prompt_text(
                "Saved synthetic cadence [days] (0 = native FIESTA grid)",
                str(args.synthetic_cadence_days),
            )
        )
        if args.filter_system in {"ps1", "ztf"}:
            args.synthetic_depth_cut = "none"
            print(
                "Synthetic depth cut: none (survey depth is applied by "
                "build_color_envelopes.py, not during generation)."
            )
        else:
            args.synthetic_depth_cut = prompt_choice(
                "Synthetic depth cut",
                [
                    ("none", "Keep every finite synthetic model point"),
                    (
                        "m5-median",
                        "Keep points brighter than the median OpSim m5 of their band",
                    ),
                ],
                args.synthetic_depth_cut,
            )
    else:
        args.synthetic_cadence_days = 0.0
        args.synthetic_depth_cut = "none"
    if args.filter_system not in {"ps1", "ztf"} and args.output_mode != "lsstlike-detected":
        args.boom_synthetic_export = prompt_choice(
            "Additional BOOM-format CSVs",
            [("no", "No additional export"),
             ("yes", "Export synthetic LSST points to boom_input/")],
            "yes" if args.boom_synthetic_export else "no",
        ) == "yes"
        if args.boom_synthetic_export:
            print("No random noise is added; each finite synthetic epoch is an assumed observation.")
            args.synthetic_depth_cut = "none"
            args.boom_export_mode = prompt_choice(
                "BOOM photometry experiment",
                [
                    (
                        "complete",
                        "Complete truth-level sample: all finite points, no m5 cut",
                    ),
                    (
                        "m5-detected",
                        "Depth-limited sample: mag <= m5 with Ivezic errors",
                    ),
                ],
                args.boom_export_mode,
            )
            for band in "ugrizy":
                args.boom_m5_by_band[band] = float(prompt_text(
                    f"Assumed single-visit m5 in {band} for Ivezic errors",
                    str(args.boom_m5_by_band[band]),
                ))
            if args.boom_export_mode == "complete":
                args.boom_target_nonempty_events = False
                print(
                    "m5 is used only in the Ivezic error model; no m5 "
                    "threshold is applied to the BOOM sample."
                )
            else:
                args.boom_target_nonempty_events = prompt_choice(
                    "Population size used for the BOOM experiment",
                    [
                        (
                            "injected",
                            "Fixed injected KNe population (end-to-end efficiency)",
                        ),
                        (
                            "nonempty",
                            "Fixed non-empty BOOM sample (conditional filter benchmark)",
                        ),
                    ],
                    "nonempty" if args.boom_target_nonempty_events else "injected",
                ) == "nonempty"
            print("BOOM keeps one point per time_mjd: keep a positive shift for BOOM input.")
            print("The requested step is reduced locally if a model-grid edge is closer.")
            args.boom_duplicate_shift_seconds = float(prompt_text(
                "BOOM shift between simultaneous bands [seconds] (0 = preserve model epochs)",
                str(args.boom_duplicate_shift_seconds),
            ))
    args.storage_format = prompt_choice(
        "Scientific table storage",
        [
            ("csv", "One legacy CSV per eligible event"),
            ("parquet", "Sharded Parquet datasets only"),
            ("both", "Sharded Parquet + a legacy CSV event sample"),
        ],
        args.storage_format,
    )
    if args.storage_format in {"parquet", "both"}:
        args.parquet_events_per_shard = int(prompt_text(
            "Events per Parquet shard",
            str(args.parquet_events_per_shard),
        ))
        args.parquet_compression = prompt_choice(
            "Parquet compression",
            [
                ("zstd", "Zstandard: compact and fast [recommended]"),
                ("snappy", "Snappy: fastest, less compact"),
                ("gzip", "Gzip: compact, slower"),
                ("none", "No compression"),
            ],
            args.parquet_compression,
        )
    if args.storage_format in {"csv", "both"}:
        args.csv_sample_size = int(prompt_text(
            "Random injected events eligible for individual CSVs (0 = all)",
            str(args.csv_sample_size),
        ))
    args.opsim_cache_size = int(prompt_text(
        "OpSim field tables kept in memory",
        str(args.opsim_cache_size),
    ))
    args.surrogate_batch_size = int(prompt_text(
        "Requested surrogate batch size (automatic validated fallback)",
        str(args.surrogate_batch_size),
    ))
    args.plot_mode = prompt_choice(
        "Diagnostic light-curve plots",
        [
            ("none", "Do not create PNG plots"),
            (
                "saved",
                "Plot each saved synthetic and/or LSST-like product",
            ),
        ],
        args.plot_mode,
    )
    if args.plot_mode == "saved":
        args.max_plot_events = int(
            prompt_text(
                "Maximum saved events plotted per product (0 = all)",
                str(args.max_plot_events),
            )
        )
    args.distance_mode = prompt_choice(
        "Luminosity-distance treatment",
        [
            ("comoving-volume", "Direct draw uniform in comoving volume"),
            ("fixed", "Fixed luminosity distance"),
        ],
        args.distance_mode,
    )
    if args.distance_mode == "fixed":
        args.fixed_distance_mpc = float(
            prompt_text("Fixed luminosity distance [Mpc]", str(args.fixed_distance_mpc))
        )
    if args.model == "Bu2026_MLP":
        args.numerical_floor_absolute_mag = float(
            prompt_text(
                "Bu2026 numerical floor in absolute magnitude",
                str(args.numerical_floor_absolute_mag),
            )
        )
        args.numerical_floor_margin_mag = float(
            prompt_text(
                "Safety margin before numerical floor [mag]",
                str(args.numerical_floor_margin_mag),
            )
        )
    args.opsim_dir = Path(prompt_text("OpSim export directory", str(args.opsim_dir)))
    args.galactic_latitude_mode = prompt_choice(
        "OpSim-field Galactic-latitude selection",
        [
            ("all", "All available Galactic latitudes"),
            ("high", "High latitude: |b| >= threshold"),
            ("low", "Low latitude: |b| < threshold"),
            ("bin", "Disjoint interval: min <= |b| < max"),
        ],
        args.galactic_latitude_mode,
    )
    if args.galactic_latitude_mode in {"low", "high"}:
        args.galactic_latitude_threshold_deg = float(
            prompt_text(
                "Absolute Galactic-latitude threshold [deg]",
                str(args.galactic_latitude_threshold_deg),
            )
        )
    elif args.galactic_latitude_mode == "bin":
        args.galactic_latitude_min_deg = float(
            prompt_text(
                "Inclusive lower |b| edge [deg]",
                str(args.galactic_latitude_min_deg),
            )
        )
        args.galactic_latitude_max_deg = float(
            prompt_text(
                "Exclusive upper |b| edge [deg]",
                str(args.galactic_latitude_max_deg),
            )
        )
    if args.output_mode in {
        "all-synthetic-lsstlike-detected",
        "detected-pairs",
        "lsstlike-detected",
    }:
        args.min_detections = int(
            prompt_text("Minimum number of LSST-like detections", str(args.min_detections))
        )
    if args.boom_synthetic_export and args.boom_target_nonempty_events:
        args.n_samples = int(prompt_text(
            "Number of non-empty BOOM light curves to export",
            str(args.n_samples),
        ))
        print(
            "Additional physical KNe will be injected when an event has no "
            "detection after the m5 cut."
        )
    else:
        args.n_samples = int(prompt_text(
            "Number of simulated events", str(args.n_samples)
        ))
    args.pairing_mode = prompt_choice(
        "Intrinsic-population pairing",
        [
            (
                "none",
                "Independent population (legacy behaviour)",
            ),
            (
                "create",
                "Create a reusable intrinsic KNe catalogue",
            ),
            (
                "reuse",
                "Reuse an intrinsic KNe catalogue for a paired sky run",
            ),
        ],
        args.pairing_mode,
    )
    if args.pairing_mode != "none":
        args.intrinsic_catalog = Path(
            prompt_text(
                "Shared intrinsic catalogue",
                str(args.intrinsic_catalog or "paired_intrinsic_population.csv"),
            )
        )
    if args.pairing_mode == "reuse" and args.distance_mode == "fixed":
        args.catalog_distance_policy = prompt_choice(
            "Distance used when reusing the intrinsic catalogue",
            [("current-fixed", "Use the fixed distance entered for this run"),
             ("stored", "Keep the distance and redshift stored in the catalogue")],
            args.catalog_distance_policy,
        )
    args.n_realizations = int(
        prompt_text(
            "Number of independent population realizations",
            str(args.n_realizations),
        )
    )
    seed_text = prompt_text(
        (
            "Master intrinsic-population seed (integer)"
            if args.n_realizations > 1
            else "Intrinsic-population seed (integer)"
            if args.pairing_mode != "none"
            else "Random seed (integer)"
        ),
        str(args.seed if args.seed is not None else secrets.randbits(32)),
    )
    args.seed = int(seed_text)
    if args.pairing_mode != "none":
        default_sky_seed = derive_sky_seed(
            args.seed,
            args.galactic_latitude_mode,
            args.galactic_latitude_threshold_deg,
            args.galactic_latitude_max_deg,
        )
        args.sky_seed = int(
            prompt_text(
                "Sky/noise seed (same = paired sky; different = new sky)",
                str(
                    args.sky_seed
                    if args.sky_seed is not None
                    else default_sky_seed
                ),
            )
        )
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.boom_max_injection_factor < 1:
        raise ValueError("boom_max_injection_factor must be >= 1")
    if args.boom_export_mode not in {"complete", "m5-detected"}:
        raise ValueError("boom_export_mode must be complete or m5-detected")
    if (
        args.boom_synthetic_export
        and args.boom_export_mode == "m5-detected"
        and args.boom_target_nonempty_events
        and args.pairing_mode != "none"
    ):
        raise ValueError(
            "An exact non-empty BOOM target is incompatible with intrinsic "
            "catalogue pairing because m5 selection would change catalogue "
            "membership. Use independent-population mode for the BOOM sample."
        )
    if args.n_realizations <= 0:
        raise ValueError("--n-realizations must be positive")
    if args.n_realizations > 1 and args.pairing_mode != "none":
        raise ValueError(
            "Multiple independent realizations require --pairing-mode none. "
            "Paired catalogues serve a different low/high-sky comparison."
        )
    validate_model_options(args)
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.sky_seed is not None and args.sky_seed < 0:
        raise ValueError("--sky-seed must be non-negative")
    if args.min_detections <= 0:
        raise ValueError("--min-detections must be positive")
    if args.fixed_distance_mpc <= 0:
        raise ValueError("--fixed-distance-mpc must be positive")
    if args.distance_min_mpc <= 0:
        raise ValueError("--distance-min-mpc must be positive")
    if args.distance_max_mpc <= args.distance_min_mpc:
        raise ValueError("--distance-max-mpc must exceed --distance-min-mpc")
    if not np.isfinite(args.numerical_floor_absolute_mag):
        raise ValueError("--numerical-floor-absolute-mag must be finite")
    if not np.isfinite(args.numerical_floor_margin_mag):
        raise ValueError("--numerical-floor-margin-mag must be finite")
    if args.numerical_floor_margin_mag < 0:
        raise ValueError("--numerical-floor-margin-mag must be >= 0")
    if args.synthetic_cadence_days < 0:
        raise ValueError("--synthetic-cadence-days must be >= 0")
    if args.max_plot_events < 0:
        raise ValueError("--max-plot-events must be >= 0")
    if args.parquet_events_per_shard < 1:
        raise ValueError("--parquet-events-per-shard must be >= 1")
    if args.csv_sample_size < 0:
        raise ValueError("--csv-sample-size must be >= 0")
    if args.opsim_cache_size < 0:
        raise ValueError("--opsim-cache-size must be >= 0")
    if args.surrogate_batch_size < 1:
        raise ValueError("--surrogate-batch-size must be >= 1")
    if args.pairing_mode == "none" and args.intrinsic_catalog is not None:
        raise ValueError(
            "--intrinsic-catalog requires --pairing-mode create or reuse"
        )
    if args.pairing_mode in {"create", "reuse"} and args.intrinsic_catalog is None:
        raise ValueError(
            "--intrinsic-catalog is required with --pairing-mode create or reuse"
        )
    if args.pairing_mode == "create" and args.intrinsic_catalog.expanduser().exists():
        raise FileExistsError(
            f"Intrinsic catalogue already exists: {args.intrinsic_catalog}. "
            "Select pairing mode 'reuse' or choose another path."
        )
    if args.pairing_mode == "reuse" and not args.intrinsic_catalog.expanduser().is_file():
        raise FileNotFoundError(
            f"Intrinsic catalogue not found: {args.intrinsic_catalog}"
        )
    if args.catalog_distance_policy == "current-fixed":
        if args.pairing_mode != "reuse" or args.distance_mode != "fixed":
            raise ValueError("--catalog-distance-policy current-fixed requires --pairing-mode reuse and --distance-mode fixed")
    if args.filter_system in {"ps1", "ztf"} and args.output_mode != "synthetic-mjd":
        raise ValueError(
            f"--filter-system {args.filter_system} supports only --output-mode "
            "synthetic-mjd. LSST-like products require LSST model magnitudes; "
            "use a combined lsst+... system when Rubin products are needed."
        )
    if args.storage_format in {"parquet", "both"}:
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Parquet output requires pyarrow. Install it with "
                "`python -m pip install pyarrow` or select --storage-format csv."
            ) from exc
    if not 0.0 <= args.galactic_latitude_threshold_deg <= 90.0:
        raise ValueError("--galactic-latitude-threshold-deg must lie in [0, 90]")
    if not 0.0 <= args.galactic_latitude_min_deg < args.galactic_latitude_max_deg <= 90.0:
        raise ValueError(
            "Latitude-bin edges must satisfy 0 <= min < max <= 90 deg"
        )
    if args.synthetic_depth_cut != "none":
        if args.filter_system != "lsst":
            raise ValueError(
                "The generator-level m5 cut is LSST-only. For a paired "
                "LSST+PS1 or LSST+ZTF run select 'none'; "
                "build_color_envelopes.py applies the appropriate survey "
                "depth consistently after generation."
            )
        if args.output_mode == "lsstlike-detected":
            raise ValueError(
                "--synthetic-depth-cut applies only to modes that save a "
                "synthetic CSV: all-synthetic-lsstlike-detected, "
                "detected-pairs or synthetic-mjd"
            )
    if (
        args.parameter_source == "posterior"
        and args.pairing_mode != "reuse"
        and args.posterior_file is None
    ):
        raise ValueError("--posterior-file is required when --parameter-source posterior")
    return args

# OpSim uses single-letter Rubin bands, whereas FIESTA uses explicit filter
# names.  This mapping selects the model curve required by each visit.
BAND_TO_FIESTA = {
    "u": "lsstu",
    "g": "lsstg",
    "r": "lsstr",
    "i": "lssti",
    "z": "lsstz",
    "y": "lssty",
}


# Standard Rubin throughput curves and detector/exposure parameters used by
# the m5-based photometric-noise model. They are initialized only for an
# actual run so importing this module cannot start Rubin data loading.
BP = None
PHOT = None


def initialize_rubin_photometry() -> None:
    """Initialize the shared Rubin photometric objects for one process."""
    global BP, PHOT
    if BP is not None:
        return
    BP = rubin_bandpasses()
    PHOT = {
        band: PhotometricParameters(bandpass=band)
        for band in "ugrizy"
    }


def plot_synthetic_only(times, mag_dict, filters, filter_colors, mag_lim, outfile):
    """Plot synthetic curves using the active workflow configuration."""
    return _plot_synthetic_only(
        times, mag_dict, filters, filter_colors, mag_lim, outfile
    )


def plot_saved_synthetic_measurements(
    measurements,
    filter_colors,
    outfile,
    event_index,
    filter_systems,
    filter_by_system,
):
    """Plot the final saved synthetic measurements for one event."""
    return _plot_saved_synthetic_measurements(
        measurements,
        filter_colors,
        outfile,
        event_index,
        filter_systems,
        filter_by_system,
    )


def plot_numerical_floor_diagnostic(
    times,
    raw_magnitudes_by_filter,
    filters,
    filter_colors,
    apparent_cut_mag,
    outfile,
):
    """Plot the numerical-floor diagnostic for one prediction."""
    return _plot_numerical_floor_diagnostic(
        times,
        raw_magnitudes_by_filter,
        filters,
        filter_colors,
        apparent_cut_mag,
        outfile,
    )


def plot_synthetic_and_lsstlike(times, mag_dict, obs, filters, filter_colors, mag_lim, outfile):
    """Overlay synthetic and LSST-like products for one event."""
    return _plot_synthetic_and_lsstlike(
        times,
        mag_dict,
        obs,
        filters,
        filter_colors,
        mag_lim,
        outfile,
        BAND_TO_FIESTA,
    )


def plot_lsstlike_only(
    observations,
    filters,
    filter_colors,
    event_index,
    outfile,
    tmax_days=None,
):
    """Plot the LSST-like product using the active band mapping."""
    return _plot_lsstlike_only(
        observations,
        filters,
        filter_colors,
        event_index,
        outfile,
        BAND_TO_FIESTA,
        tmax_days=tmax_days,
    )


def apply_lsst_realism_to_fiesta(
    times_model,
    mag_model_dict,
    visits_df,
    t0_mjd=None,
    det_snr=5.0,
    mag_truth_cap=None,
    rng=None,
):

    """Project a dense FIESTA light curve through an OpSim visit sequence.

    Parameters
    ----------
    times_model
        Native FIESTA observer-frame times in days.  Values must be finite,
        strictly positive, and strictly increasing.
    mag_model_dict
        Mapping from FIESTA filter name (for example ``lsstg``) to magnitudes
        sampled at ``times_model``.
    visits_df
        OpSim visit table. The compact ``MJD``, ``band``, ``m5`` schema and the
        legacy ``observationStartMJD``, ``band``, ``fiveSigmaDepth`` schema are
        accepted.
    t0_mjd
        Merger time in MJD.  If omitted, the first visit time is used.
    det_snr
        Detection threshold applied to the realized flux SNR.
    mag_truth_cap
        Optional faint-magnitude cutoff.  Truth values numerically larger than
        this limit are treated as unavailable.
    rng
        NumPy random generator used for the Gaussian flux realization.

    Returns
    -------
    pandas.DataFrame
        One row per post-merger OpSim visit.  The table retains OpSim metadata
        and adds ``t_days``, model magnitude, expected SNR,
        ``mag_err``, realized SNR, ``detected``, ``mag_obs``, and ``mag_ulim``.

    Notes
    -----
    ``fiveSigmaDepth`` (renamed ``m5``) summarizes the conditions of each visit.
    Rubin's implementation of the Ivezić et al. m5 model supplies the expected
    SNR and magnitude uncertainty.  Noise is drawn in relative flux units, so
    no absolute flux zero point is required; it cancels in flux/SNR ratios.
    """

    if rng is None:
        rng = np.random.default_rng()

    v = normalize_opsim_visit_columns(visits_df)
    v = v.sort_values("observationStartMJD").reset_index(drop=True)
    # m5 is the per-visit 5-sigma point-source depth.  It compresses sky
    # brightness, seeing, airmass, and other observing conditions into one value.
    v = v.rename(columns={"fiveSigmaDepth": "m5"})

    if t0_mjd is None:
        t0_mjd = float(v["observationStartMJD"].min())

    # Time since trigger (days). Keep only post-trigger visits.
    v["t_days"] = v["observationStartMJD"] - t0_mjd
    v = v[v["t_days"] >= 0].reset_index(drop=True)

    v["fiesta_filter"] = v["band"].map(BAND_TO_FIESTA)

    mag_true = np.full(len(v), np.nan, dtype=float)
    # Interpolate one band at a time to avoid repeated dictionary lookups for
    # every individual visit.
    for b in v["band"].unique():
        f = BAND_TO_FIESTA.get(b)
        if f is None or f not in mag_model_dict:
            continue
        mask = (v["band"].values == b)

        # Interpolate the dense synthetic lightcurve to the irregular OpSim visit times.
        # This converts the model grid (times_model) into per-visit truth magnitudes mag_true(t_days).
        mag_true[mask] = interpolate_magnitude_log_time(
            times_model,
            mag_model_dict[f],
            v.loc[mask, "t_days"].to_numpy()
        )


    if mag_truth_cap is not None:
        mag_true[mag_true > float(mag_truth_cap)] = np.nan

    v["mag_true"] = mag_true

    snr_exp = np.full(len(v), np.nan, dtype=float)
    mag_err = np.full(len(v), np.nan, dtype=float)

    for b in v["band"].unique():
        mask = (
            (v["band"].values == b)
            & np.isfinite(v["mag_true"].values)
            & np.isfinite(v["m5"].values)
        )
        if not np.any(mask):
            continue

        bp = BP[b]
        pp = PHOT[b]

        idx = np.where(mask)[0]
        for k in idx:
            mi = float(v.at[k, "mag_true"])
            m5i = float(v.at[k, "m5"])

            snr_k, _ = calc_snr_m5(mi, bp, m5i, pp)
            merr_k, _ = calc_mag_error_m5(mi, bp, m5i, pp)

            snr_exp[k] = snr_k
            mag_err[k] = merr_k

    v["snr_exp"] = snr_exp
    v["mag_err"] = mag_err

    # Convert mag -> relative flux, add Gaussian noise in flux space, then apply a detection threshold.
    # If detected: report mag_obs with mag_err; otherwise report an upper limit mag_ulim = m5.
    ok = np.isfinite(v["mag_true"].values) & np.isfinite(v["snr_exp"].values) & (v["snr_exp"].values > 0)

    f_true = np.full(len(v), np.nan, dtype=float)
    f_true[ok] = 10.0 ** (-0.4 * v.loc[ok, "mag_true"].to_numpy())

    sigma_f = np.full(len(v), np.nan, dtype=float)
    sigma_f[ok] = f_true[ok] / v.loc[ok, "snr_exp"].to_numpy()

    f_obs = f_true + rng.normal(0.0, 1.0, size=len(v)) * sigma_f

    snr_obs = np.full(len(v), np.nan, dtype=float)
    snr_obs[ok] = f_obs[ok] / sigma_f[ok]

    detected = ok & (f_obs > 0) & (snr_obs >= det_snr)

    mag_obs = np.full(len(v), np.nan, dtype=float)
    mag_obs[detected] = -2.5 * np.log10(f_obs[detected])

    v["snr_obs"] = snr_obs
    v["detected"] = detected
    v["mag_obs"] = mag_obs
    v["mag_ulim"] = np.where(~detected, v["m5"].values, np.nan)

    return v


def load_synthetic_m5_medians(
    opsim_dir: Path,
    bands: tuple[str, ...] = tuple("ugrizy"),
) -> tuple[dict[str, float], Path]:
    """Load per-band m5 medians precomputed by ``get_opsim_baseline.py``."""
    path = opsim_dir / "opsim_m5_medians_by_band.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the updated get_opsim_baseline.py, or run "
            "prepare_opsim_m5_medians.py once on this existing export; the "
            "kilonova generator intentionally does not recompute survey-level "
            "m5 medians."
        )

    table = pd.read_csv(path)
    required = {"band", "m5_median"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    table = table.copy()
    table["band"] = table["band"].astype(str).str.strip()
    table["m5_median"] = pd.to_numeric(table["m5_median"], errors="coerce")
    if table["band"].duplicated().any():
        duplicates = sorted(table.loc[table["band"].duplicated(False), "band"].unique())
        raise ValueError(f"Duplicate bands in {path}: {duplicates}")

    by_band = table.set_index("band")["m5_median"].to_dict()
    invalid = [band for band in bands if not np.isfinite(by_band.get(band, np.nan))]
    if invalid:
        raise ValueError(f"Missing or non-finite m5 medians for bands {invalid} in {path}")
    return {band: float(by_band[band]) for band in bands}, path


def trim_lsstlike_after_last_detection(
    observations: pd.DataFrame,
    days_after_last_detection: float = 10.0,
) -> tuple[pd.DataFrame, float, float, bool]:
    """Trim visits after the last detection while retaining detections."""
    return _trim_lsstlike_after_last_detection(
        observations, days_after_last_detection
    )


def apply_synthetic_m5_median_cut(
    measurements: pd.DataFrame,
    objectid: str,
    m5_medians: dict[str, float],
) -> pd.DataFrame:
    """Apply the inclusive median-depth cut and rebuild identifiers."""
    return _apply_synthetic_m5_median_cut(measurements, objectid, m5_medians)


class UniformComovingVolumeSampler:
    """Draw (z, D_L) uniformly in comoving volume between D_L limits.

    The sampler uses Planck18 from astropy. It precomputes a monotonic grid in z
    and samples uniformly in D_C^3, equivalent to uniform comoving volume in a
    flat cosmology. The angular part is irrelevant here because sky positions are
    supplied separately by OpSim fields.
    """

    def __init__(self, d_l_min_mpc: float, d_l_max_mpc: float, n_grid: int = 20000):
        """Precompute the monotonic Planck18 distance/redshift lookup grid."""
        self.d_l_min_mpc = float(d_l_min_mpc)
        self.d_l_max_mpc = float(d_l_max_mpc)
        if self.d_l_min_mpc <= 0 or self.d_l_max_mpc <= self.d_l_min_mpc:
            raise ValueError("Require 0 < D_L_min < D_L_max")

        self.z_min = float(z_at_value(COSMO.luminosity_distance, self.d_l_min_mpc * u.Mpc))
        self.z_max = float(z_at_value(COSMO.luminosity_distance, self.d_l_max_mpc * u.Mpc))
        self.z_grid = np.linspace(self.z_min, self.z_max, int(n_grid))
        self.dc_grid_mpc = COSMO.comoving_distance(self.z_grid).to_value(u.Mpc)
        self.dl_grid_mpc = COSMO.luminosity_distance(self.z_grid).to_value(u.Mpc)
        self.dc3_min = float(self.dc_grid_mpc[0] ** 3)
        self.dc3_max = float(self.dc_grid_mpc[-1] ** 3)

    def sample(self, rng: np.random.Generator, u01: float | None = None) -> tuple[float, float]:
        """Return one ``(redshift, luminosity distance in Mpc)`` draw."""
        if u01 is None:
            u01 = float(rng.uniform(0.0, 1.0))
        u01 = float(np.clip(u01, 0.0, np.nextafter(1.0, 0.0)))
        dc3 = self.dc3_min + u01 * (self.dc3_max - self.dc3_min)
        dc = dc3 ** (1.0 / 3.0)
        z = float(np.interp(dc, self.dc_grid_mpc, self.z_grid))
        d_l = float(np.interp(z, self.z_grid, self.dl_grid_mpc))
        return z, d_l


def sample_distance_redshift_for_event(
    rng: np.random.Generator,
    sampler: UniformComovingVolumeSampler | None,
    *,
    distance_sampling: str,
    fixed_redshift: float,
    fixed_luminosity_distance_mpc: float,
) -> tuple[float, float]:
    """Return one Planck18-consistent ``(z, D_L[Mpc])`` pair.

    Population runs draw independently and uniformly in comoving volume.
    Fixed-distance runs return the single configured diagnostic distance.
    """
    mode = str(distance_sampling).lower().strip()
    if mode == "fixed":
        return float(fixed_redshift), float(fixed_luminosity_distance_mpc)
    if mode == "comoving_volume":
        if sampler is None:
            raise RuntimeError("Comoving-volume sampler was not initialized")
        return sampler.sample(rng)
    raise ValueError("distance_sampling must be 'fixed' or 'comoving_volume'")


def sample_inclination_for_event(
    rng: np.random.Generator,
    *,
    sampling: str,
    minimum_rad: float,
    maximum_rad: float,
) -> float:
    """Return inclination_EM [rad].

    For isotropic orientations over [0, pi/2], draw mu=cos(i) uniformly
    in [0, 1] and return i=arccos(mu). This gives p(i) proportional to sin(i).
    """
    mode = str(sampling).lower().strip()
    i_min = float(minimum_rad)
    i_max = float(maximum_rad)
    if not (0.0 <= i_min < i_max <= np.pi):
        raise ValueError("Require 0 <= minimum_rad < maximum_rad <= pi")

    if mode == "uniform_angle":
        return float(rng.uniform(i_min, i_max))

    if mode == "uniform_cos":
        mu_lo = float(np.cos(i_max))
        mu_hi = float(np.cos(i_min))
        mu = float(rng.uniform(mu_lo, mu_hi))
        mu = float(np.clip(mu, -1.0, 1.0))
        return float(np.arccos(mu))

    raise ValueError("sampling must be 'uniform_cos' or 'uniform_angle'")


# --- PHOTOMETRIC FILTER SYSTEMS ---

# Exact FIESTA filter identifiers used by both spectral-flux surrogates.
# The selected system is chosen in the terminal menu below.
FILTER_SYSTEM_CONFIG = {
    "lsst": {
        "label": "LSST ugrizy",
        "filters": ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"],
        "by_band": {
            "u": "lsstu",
            "g": "lsstg",
            "r": "lsstr",
            "i": "lssti",
            "z": "lsstz",
            "y": "lssty",
        },
    },
    "ps1": {
        "label": "Pan-STARRS1 grizy",
        "filters": ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y"],
        "by_band": {
            "g": "ps1::g",
            "r": "ps1::r",
            "i": "ps1::i",
            "z": "ps1::z",
            "y": "ps1::y",
        },
    },
    "ztf": {
        "label": "ZTF gri",
        "filters": ["ztfg", "ztfr", "ztfi"],
        "by_band": {
            "g": "ztfg",
            "r": "ztfr",
            "i": "ztfi",
        },
    },
    "lsst+ps1": {
        "label": "LSST ugrizy + Pan-STARRS1 grizy",
        "systems": ("lsst", "ps1"),
    },
    "lsst+ztf": {
        "label": "LSST ugrizy + ZTF gri",
        "systems": ("lsst", "ztf"),
    },
}

BASE_BAND_COLORS = {
    "u": "#6a3d9a",
    "g": "#33a02c",
    "r": "#e31a1c",
    "i": "#ff7f00",
    "z": "#b15928",
    "y": "#4d4d4d",
}


def derive_sky_seed(
    intrinsic_seed,
    latitude_mode,
    threshold_deg,
    latitude_max_deg=None,
):
    """Derive a reproducible sky seed for the selected latitude region."""
    return _derive_sky_seed(
        intrinsic_seed, latitude_mode, threshold_deg, latitude_max_deg
    )


# Runtime configuration and model initialization are explicit; importing this
# module never starts a population calculation.


def realization_seed_list(args: argparse.Namespace) -> list[int]:
    """Return one reproducible intrinsic seed per requested realization."""
    if args.realization_seeds:
        seeds = [
            int(token.strip())
            for token in str(args.realization_seeds).split(",")
            if token.strip()
        ]
        if len(seeds) != int(args.n_realizations):
            raise ValueError(
                "--realization-seeds must contain exactly --n-realizations values"
            )
    else:
        master_seed = (
            int(args.seed) if args.seed is not None else secrets.randbits(32)
        )
        if int(args.n_realizations) == 1:
            seeds = [master_seed]
        else:
            children = np.random.SeedSequence(master_seed).spawn(
                int(args.n_realizations)
            )
            seeds = [
                int(child.generate_state(1, dtype=np.uint32)[0])
                for child in children
            ]
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Realization seeds must lie in [0, 2**32)")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Realization seeds must be unique")
    return seeds


def namespace_as_json_dict(args: argparse.Namespace) -> dict[str, object]:
    """Convert one parsed runtime configuration to JSON-safe values."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def generate_realization_batch(args: argparse.Namespace, seeds: list[int]) -> Path:
    """Run the unchanged single-population generator once for every seed."""
    runs_root = args.runs_root.expanduser().resolve()
    master_seed = int(args.seed) if args.seed is not None else seeds[0]
    batch_dir = runs_root / (
        f"{slugify(args.run_name)}_realizations_master_seed_{master_seed}"
    )
    batch_dir.mkdir(parents=True, exist_ok=False)

    sky_seeds = None
    if args.sky_seed is not None:
        sky_children = np.random.SeedSequence(int(args.sky_seed)).spawn(len(seeds))
        sky_seeds = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in sky_children
        ]

    manifest_rows = []
    for realization, intrinsic_seed in enumerate(seeds, start=1):
        # The child runs directly in this Python process. Preserve Path
        # objects; namespace_as_json_dict() is reserved for JSON output.
        child = dict(vars(args))
        child_run_name = f"{args.run_name}_realization_{realization:02d}"
        child.update(
            {
                "run_name": child_run_name,
                "seed": intrinsic_seed,
                "sky_seed": (
                    sky_seeds[realization - 1]
                    if sky_seeds is not None else intrinsic_seed
                ),
                "n_realizations": 1,
                "realization_seeds": str(intrinsic_seed),
            }
        )
        run_dir = runs_root / f"{slugify(child_run_name)}_seed_{intrinsic_seed}"
        print(
            f"\n=== Realization {realization}/{len(seeds)}: "
            f"intrinsic seed {intrinsic_seed} ===",
            flush=True,
        )
        run_generation(argparse.Namespace(**child))
        manifest_rows.append(
            {
                "realization": realization,
                "intrinsic_seed": intrinsic_seed,
                "sky_noise_seed": child["sky_seed"],
                "n_events": int(args.n_samples),
                "run_dir": str(run_dir),
            }
        )
        pd.DataFrame(manifest_rows).to_csv(
            batch_dir / "realizations.csv", index=False
        )

    batch_configuration = namespace_as_json_dict(args)
    batch_configuration["realization_seeds_resolved"] = seeds
    batch_configuration["manifest"] = str(batch_dir / "realizations.csv")
    (batch_dir / "batch_config.json").write_text(
        json.dumps(batch_configuration, indent=2) + "\n"
    )
    print(f"\nSaved realization manifest: {batch_dir / 'realizations.csv'}")
    return batch_dir


def run_generation(RUNTIME: argparse.Namespace) -> None:
    """Execute one fully configured population generation without prompting."""
    SAVE_BOOM_SYNTHETIC = bool(getattr(RUNTIME, "boom_synthetic_export", False))
    BOOM_EXPORT_MODE = str(getattr(RUNTIME, "boom_export_mode", "complete"))
    BOOM_M5 = dict(getattr(RUNTIME, "boom_m5_by_band", BOOM_M5_DEFAULT))
    BOOM_SIGMA_SYS = float(getattr(RUNTIME, "boom_sigma_sys", 0.005))
    BOOM_SHIFT_SECONDS = float(getattr(RUNTIME, "boom_duplicate_shift_seconds", 30.0))
    if SAVE_BOOM_SYNTHETIC:
        RUNTIME.synthetic_depth_cut = "none"
        if RUNTIME.filter_system in {"ps1", "ztf"} or RUNTIME.output_mode == "lsstlike-detected":
            raise ValueError("BOOM synthetic export requires an LSST synthetic product")
        if any(b not in BOOM_M5 or not np.isfinite(BOOM_M5[b]) for b in "ugrizy"):
            raise ValueError("Provide a finite BOOM m5 for every LSST band")
        synthetic_to_boom_csv(
            pd.DataFrame(), 0, BOOM_M5, BOOM_SIGMA_SYS,
            BOOM_SHIFT_SECONDS, BOOM_EXPORT_MODE,
        )
    missing = [
        name
        for name, dependency in (
            ("jax", jax),
            ("fiesta", Uniform),
            ("astropy", COSMO),
            ("rubin_sim", rubin_bandpasses),
        )
        if dependency is None
    ]
    if missing:
        raise ModuleNotFoundError(
            "Population generation requires the scientific environment: "
            + ", ".join(missing)
        )
    initialize_rubin_photometry()
    RUNTIME.seed = int(RUNTIME.seed)

    MODEL_NAME = RUNTIME.model
    FILTER_SYSTEM = RUNTIME.filter_system
    FILTER_SYSTEMS = (
        list(FILTER_SYSTEM_CONFIG[FILTER_SYSTEM].get("systems", (FILTER_SYSTEM,)))
    )
    FILTER_BY_SYSTEM = {
        system: dict(FILTER_SYSTEM_CONFIG[system]["by_band"])
        for system in FILTER_SYSTEMS
    }
    FILTERS = [
        filter_name
        for system in FILTER_SYSTEMS
        for filter_name in FILTER_SYSTEM_CONFIG[system]["filters"]
    ]
    # OpSim visits and m5 always refer to Rubin.  Keep this alias explicitly LSST
    # even when PS1 synthetic magnitudes are generated in parallel.
    FILTER_BY_BAND = dict(FILTER_SYSTEM_CONFIG["lsst"]["by_band"])
    FILTER_COLORS = {
        filter_name: BASE_BAND_COLORS[band]
        for system in FILTER_SYSTEMS
        for band, filter_name in FILTER_BY_SYSTEM[system].items()
    }
    print(f"Photometric filter system: {FILTER_SYSTEM_CONFIG[FILTER_SYSTEM]['label']}")
    print(f"Using filters: {FILTERS}")


    # =============================================================================
    # CONFIGURATION
    # =============================================================================

    # Optional products. Parquet is optimized for population analyses; the CSV
    # path preserves the historical one-file-per-event interface used by the VM.
    STORAGE_FORMAT = RUNTIME.storage_format
    SAVE_CSV = STORAGE_FORMAT in {"csv", "both"}
    SAVE_PARQUET = STORAGE_FORMAT in {"parquet", "both"}
    PARQUET_EVENTS_PER_SHARD = int(RUNTIME.parquet_events_per_shard)
    PARQUET_COMPRESSION = (
        None if RUNTIME.parquet_compression == "none" else RUNTIME.parquet_compression
    )
    CSV_SAMPLE_SIZE = int(RUNTIME.csv_sample_size)
    OPSIM_CACHE_SIZE = int(RUNTIME.opsim_cache_size)
    SURROGATE_BATCH_SIZE = int(RUNTIME.surrogate_batch_size)
    SAVE_PNG = False  # Legacy overlay diagnostic; intentionally not enabled here.
    SAVE_PNG_LSSTLIKE = (
        RUNTIME.plot_mode == "saved"
        and RUNTIME.output_mode in {
            "all-synthetic-lsstlike-detected",
            "detected-pairs",
            "lsstlike-detected",
        }
    )
    SAVE_PNG_SYNTHETIC_ONLY = (
        RUNTIME.plot_mode == "saved"
        and RUNTIME.output_mode in {
            "all-synthetic-lsstlike-detected",
            "detected-pairs",
            "synthetic-mjd",
        }
    )
    MAX_PLOT_EVENTS = int(RUNTIME.max_plot_events)

    # Per-event output selection. summary.csv always retains every injected event.
    OUTPUT_MODE = RUNTIME.output_mode
    MIN_DETECTIONS = int(RUNTIME.min_detections)
    SYNTHETIC_SCHEMA = RUNTIME.synthetic_schema
    SYNTHETIC_CADENCE_DAYS = float(RUNTIME.synthetic_cadence_days)
    SYNTHETIC_DEPTH_CUT = RUNTIME.synthetic_depth_cut
    USE_OPSIM_CONTEXT = True
    APPLY_LSST_REALISM = OUTPUT_MODE in {
        "all-synthetic-lsstlike-detected",
        "detected-pairs",
        "lsstlike-detected",
    }

    # LSST realism / OpSim context
    OPSIM_DIR = RUNTIME.opsim_dir.expanduser().resolve()
    GALACTIC_LATITUDE_MODE = RUNTIME.galactic_latitude_mode
    GALACTIC_LATITUDE_THRESHOLD_DEG = float(RUNTIME.galactic_latitude_threshold_deg)
    GALACTIC_LATITUDE_MIN_DEG = float(RUNTIME.galactic_latitude_min_deg)
    GALACTIC_LATITUDE_MAX_DEG = float(RUNTIME.galactic_latitude_max_deg)
    DET_SNR = 5.0
    LSSTLIKE_DAYS_AFTER_LAST_DETECTION = 10.0


    # t0 is the merger time in absolute MJD.
    # We convert OpSim visit times to "time since trigger" via t_days = MJD - t0.
    #
    # T0_MODE="uniform": merger can occur at any time within the field survey window (unconditioned).
    # T0_MODE="visit_choice": merger forced to coincide with an existing visit time (conditioned; typically biases detectability high).
    T0_MODE = "uniform"   # "uniform" or "visit_choice"

    # Distance/redshift sampling
    # -------------------------
    # "comoving_volume": draw events directly uniformly in comoving volume between
    # the luminosity-distance limits below. This makes distance weights unnecessary.
    # "fixed": use one luminosity distance and its corresponding Planck18 redshift
    # for every event.
    DISTANCE_SAMPLING = (
        "fixed" if RUNTIME.distance_mode == "fixed" else "comoving_volume"
    )
    FIXED_LUMINOSITY_DISTANCE_MPC = float(RUNTIME.fixed_distance_mpc)
    FIXED_REDSHIFT = (
        float(z_at_value(COSMO.luminosity_distance, FIXED_LUMINOSITY_DISTANCE_MPC * u.Mpc))
        if DISTANCE_SAMPLING == "fixed"
        else np.nan
    )
    LUMINOSITY_DISTANCE_MIN_MPC = float(RUNTIME.distance_min_mpc)
    LUMINOSITY_DISTANCE_MAX_MPC = float(RUNTIME.distance_max_mpc)
    N_COMOVING_DISTANCE_GRID = 20000

    # Inclination sampling
    # --------------------
    # For an isotropic orientation distribution, cos(i) must be uniform, not i.
    # The model uses inclination_EM in radians, so we draw mu=cos(i) uniformly
    # and convert back to i=arccos(mu).
    INCLINATION_SAMPLING = "uniform_cos"  # "uniform_cos" or "uniform_angle"
    INCLINATION_MIN_RAD = 0.0
    INCLINATION_MAX_RAD = np.pi / 2

    # Simulation size
    N_SAMPLES = int(RUNTIME.n_samples)
    BOOM_TARGET_NONEMPTY = bool(
        RUNTIME.boom_synthetic_export
        and RUNTIME.boom_export_mode == "m5-detected"
        and RUNTIME.boom_target_nonempty_events
    )
    BOOM_MAX_INJECTION_FACTOR = float(RUNTIME.boom_max_injection_factor)
    MAX_INJECTED_EVENTS = (
        max(N_SAMPLES, int(np.ceil(N_SAMPLES * BOOM_MAX_INJECTION_FACTOR)))
        if BOOM_TARGET_NONEMPTY
        else N_SAMPLES
    )

    # Paired-population design
    # ------------------------
    # In paired mode, intrinsic ejecta parameters, inclination, distance and
    # redshift are stored in one reusable catalogue.  Each low/high-latitude run
    # then draws only its sky context from a separate seed.  Matching event_id
    # values therefore denote the same physical KNe in every paired run.
    PAIRING_MODE = RUNTIME.pairing_mode
    CATALOG_DISTANCE_POLICY = RUNTIME.catalog_distance_policy
    INTRINSIC_CATALOG_PATH = (
        RUNTIME.intrinsic_catalog.expanduser().resolve()
        if RUNTIME.intrinsic_catalog is not None
        else None
    )

    # Physical population source is independent of the distance mode. Direct
    # comoving-volume draws need no volume weight; fixed-distance runs represent a
    # population slice at one distance and are also not reweighted.
    PARAMETER_SOURCE = RUNTIME.parameter_source
    POSTERIOR_FILE = RUNTIME.posterior_file.expanduser().resolve() if RUNTIME.posterior_file else None
    POSTERIOR_INCLINATION = RUNTIME.posterior_inclination
    RUN_NAME = slugify(RUNTIME.run_name)
    RUNS_ROOT = RUNTIME.runs_root.expanduser().resolve()

    # Bu2026 was trained with a late-time faint-flux truncation corresponding
    # approximately to M=0.  The usable edge is defined in absolute magnitude and
    # converted separately for every event, rather than imposing a fixed apparent
    # magnitude that would be valid at only one luminosity distance.
    NUMERICAL_FLOOR_ABSOLUTE_MAG = float(RUNTIME.numerical_floor_absolute_mag)
    NUMERICAL_FLOOR_MARGIN_MAG = float(RUNTIME.numerical_floor_margin_mag)
    NUMERICAL_SUPPORT_ABSOLUTE_MAG = (
        NUMERICAL_FLOOR_ABSOLUTE_MAG - NUMERICAL_FLOOR_MARGIN_MAG
    )
    APPLY_NUMERICAL_SUPPORT_CUT = MODEL_NAME == "Bu2026_MLP"


    def distance_modulus_mpc(luminosity_distance_mpc):
        """Distance modulus for a positive luminosity distance in Mpc."""
        distance = float(luminosity_distance_mpc)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("luminosity_distance_mpc must be finite and positive")
        return 5.0 * np.log10(distance) + 25.0


    def numerical_support_cut_apparent_mag(luminosity_distance_mpc):
        """Convert the configured absolute support edge to apparent magnitude."""
        return (
            NUMERICAL_SUPPORT_ABSOLUTE_MAG
            + distance_modulus_mpc(luminosity_distance_mpc)
        )


    def mask_bands_after_numerical_floor(magnitudes_by_filter, apparent_cut_mag):
        """Mask each band from its first contact with the floor region onward."""
        masked = {}
        cut = float(apparent_cut_mag)
        for filter_name, values in magnitudes_by_filter.items():
            curve = np.asarray(values, dtype=float).copy()
            crossing = np.flatnonzero(np.isfinite(curve) & (curve >= cut))
            if crossing.size:
                curve[int(crossing[0]):] = np.nan
            masked[filter_name] = curve
        return masked

    # Optional upper x-axis limit for diagnostic plots.  None displays every
    # available post-merger visit.
    TMAX_PLOT_DAYS = None


    # =============================================================================
    # INPUTS: OpSim visit tables (exported beforehand)
    # =============================================================================

    if USE_OPSIM_CONTEXT:
        OPSIM_FILES = sorted(OPSIM_DIR.glob("opsim_visits_field*.csv"))
        if not OPSIM_FILES:
            raise FileNotFoundError(f"No opsim_visits_field*.csv found in {OPSIM_DIR.resolve()}")
        N_FIELDS = len(OPSIM_FILES)
        print(f"Found {N_FIELDS} OpSim fields in {OPSIM_DIR}")
        OPSIM_RUN_METADATA = load_opsim_run_metadata(OPSIM_DIR)
        OPSIM_RUN_METADATA["opsim_N_FILES_FOUND_BY_GENERATE_LIGHTCURVES"] = N_FIELDS
        fields_index_path = OPSIM_DIR / "opsim_fields_index.csv"
        if fields_index_path.exists():
            OPSIM_FIELDS_INDEX = pd.read_csv(fields_index_path)
            print(f"Loaded OpSim field-coordinate index: {fields_index_path}")
        else:
            OPSIM_FIELDS_INDEX = pd.DataFrame()

        if not OPSIM_FIELDS_INDEX.empty:
            required = {"field_index", "ra0_deg", "dec0_deg"}
            missing = required - set(OPSIM_FIELDS_INDEX.columns)
            if missing:
                raise ValueError(
                    f"{fields_index_path} is missing required columns: {sorted(missing)}"
                )
            index_work = OPSIM_FIELDS_INDEX.copy()
            index_work["field_index"] = pd.to_numeric(
                index_work["field_index"], errors="raise"
            ).astype(int)
            ra_values = pd.to_numeric(index_work["ra0_deg"], errors="coerce").to_numpy(float)
            dec_values = pd.to_numeric(index_work["dec0_deg"], errors="coerce").to_numpy(float)
            valid_sky = np.isfinite(ra_values) & np.isfinite(dec_values)
            gal_lon = np.full(len(index_work), np.nan, dtype=float)
            gal_lat = np.full(len(index_work), np.nan, dtype=float)
            if valid_sky.any():
                sky = SkyCoord(
                    ra=ra_values[valid_sky] * u.deg,
                    dec=dec_values[valid_sky] * u.deg,
                    frame="icrs",
                ).galactic
                gal_lon[valid_sky] = sky.l.deg
                gal_lat[valid_sky] = sky.b.deg
            index_work["galactic_longitude_deg"] = gal_lon
            index_work["galactic_latitude_deg"] = gal_lat
            index_work["abs_galactic_latitude_deg"] = np.abs(gal_lat)
            OPSIM_FIELDS_INDEX = index_work

        def opsim_field_number(path: Path, fallback: int) -> int:
            match = re.search(r"field[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
            return int(match.group(1)) if match else int(fallback)

        field_latitude_by_id = (
            OPSIM_FIELDS_INDEX.drop_duplicates("field_index").set_index("field_index")[
                "abs_galactic_latitude_deg"
            ].to_dict()
            if not OPSIM_FIELDS_INDEX.empty
            else {}
        )
        ELIGIBLE_OPSIM_FILES = []
        for file_position, field_path in enumerate(OPSIM_FILES):
            field_number = opsim_field_number(field_path, file_position)
            abs_latitude = float(field_latitude_by_id.get(field_number, np.nan))
            if GALACTIC_LATITUDE_MODE == "all":
                keep_field = True
            elif not np.isfinite(abs_latitude):
                keep_field = False
            elif GALACTIC_LATITUDE_MODE == "high":
                keep_field = abs_latitude >= GALACTIC_LATITUDE_THRESHOLD_DEG
            elif GALACTIC_LATITUDE_MODE == "low":
                keep_field = abs_latitude < GALACTIC_LATITUDE_THRESHOLD_DEG
            else:
                keep_field = (
                    abs_latitude >= GALACTIC_LATITUDE_MIN_DEG
                    and (
                        abs_latitude < GALACTIC_LATITUDE_MAX_DEG
                        or (
                            GALACTIC_LATITUDE_MAX_DEG == 90.0
                            and abs_latitude <= 90.0
                        )
                    )
                )
            if keep_field:
                ELIGIBLE_OPSIM_FILES.append((field_number, field_path))
        if not ELIGIBLE_OPSIM_FILES:
            raise ValueError(
                "No OpSim field satisfies the requested Galactic-latitude selection. "
                "Check opsim_fields_index.csv and the threshold."
            )
        print(
            "Eligible OpSim fields after Galactic-latitude selection: "
            f"{len(ELIGIBLE_OPSIM_FILES):,}/{len(OPSIM_FILES):,} "
            f"(mode={GALACTIC_LATITUDE_MODE}, "
            + (
                f"interval=[{GALACTIC_LATITUDE_MIN_DEG:g}, "
                f"{GALACTIC_LATITUDE_MAX_DEG:g}] deg)"
                if GALACTIC_LATITUDE_MODE == "bin"
                else f"threshold={GALACTIC_LATITUDE_THRESHOLD_DEG:g} deg)"
            )
        )
    SYNTHETIC_M5_MEDIANS: dict[str, float] = {}
    SYNTHETIC_M5_MEDIANS_SOURCE = None
    if SYNTHETIC_DEPTH_CUT == "m5-median":
        SYNTHETIC_M5_MEDIANS, SYNTHETIC_M5_MEDIANS_SOURCE = load_synthetic_m5_medians(
            OPSIM_DIR
        )
        print(f"Synthetic depth cut: precomputed median OpSim m5 from {SYNTHETIC_M5_MEDIANS_SOURCE}")
        for _band in "ugrizy":
            print(f"  {_band}: {SYNTHETIC_M5_MEDIANS[_band]:.4f} mag")

    # Build the distance sampler once when needed. Fixed-distance runs do not draw
    # any distance and therefore do not need the comoving-volume interpolation grid.
    if DISTANCE_SAMPLING == "fixed":
        COMOVING_VOLUME_SAMPLER = None
        print(
            f"Distance sampling: fixed; D_L={FIXED_LUMINOSITY_DISTANCE_MPC:.6f} Mpc; "
            f"z={FIXED_REDSHIFT:.8f} (Planck18)"
        )
    else:
        COMOVING_VOLUME_SAMPLER = UniformComovingVolumeSampler(
            d_l_min_mpc=LUMINOSITY_DISTANCE_MIN_MPC,
            d_l_max_mpc=LUMINOSITY_DISTANCE_MAX_MPC,
            n_grid=N_COMOVING_DISTANCE_GRID,
        )
        print(
            f"Distance sampling: {DISTANCE_SAMPLING}; "
            f"D_L=[{LUMINOSITY_DISTANCE_MIN_MPC}, {LUMINOSITY_DISTANCE_MAX_MPC}] Mpc; "
            f"z=[{COMOVING_VOLUME_SAMPLER.z_min:.5f}, {COMOVING_VOLUME_SAMPLER.z_max:.5f}]"
        )


    # Optional one-parameter sensitivity scan.  When enabled, one reference draw
    # supplies every other intrinsic parameter and VARY_PARAM is scanned across its
    # prior interval.  Distance and inclination continue to be sampled separately.
    VARY_ONE_PARAM = False
    VARY_PARAM = "log10_mej_dyn"
    VARY_MODE = "linspace"  # "uniform" or "linspace"
    FIX_PARAM = None  # For example, "inclination_EM".
    FIX_VALUE = None  # Required when FIX_PARAM is not None.


    # Instantiate only the selected backend; Metzger never loads surrogate weights.
    surrogate_directory = (
        str(RUNTIME.fiesta_surrogate_dir.expanduser().resolve())
        if RUNTIME.fiesta_surrogate_dir is not None else None
    )
    model, MODEL_METADATA = create_fiesta_model(RUNTIME, FILTERS)
    FIESTA_SURROGATE_API = MODEL_METADATA["api"]
    MODEL_BACKEND = MODEL_METADATA["backend"]
    MODEL_CONFIGURATION = model_configuration(RUNTIME)
    MODEL_CONFIGURATION_JSON = json.dumps(MODEL_CONFIGURATION, sort_keys=True, separators=(",", ":"))
    print(f"Using model: {model.name} ({MODEL_BACKEND})")
    if MODEL_NAME in ANALYTICAL_MODELS:
        print("Spherical model: inclination is recorded for catalogue compatibility but does not affect the light curve.")
        print("Source integration grid:", MODEL_CONFIGURATION)


    # Broad intrinsic population.  Distance, redshift, and inclination are sampled
    # separately so their distributions can be controlled explicitly.
    if MODEL_NAME == "Bu2026_MLP":
        KN_prior = [
            # inclination_EM is sampled separately so it is uniform in cos(i).
            Uniform(xmin=-4.0, xmax=-1.31, naming=["log10_mej_dyn"]),
            Uniform(xmin=0.12, xmax=0.35, naming=["v_ej_dyn"]),
            Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
            Uniform(xmin=-4.0, xmax=-0.56, naming=["log10_mej_wind"]),
            Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"]),
            Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"]),
        ]
    elif MODEL_NAME == "Bu2019_MLP":
        # Exact training bounds recorded in Bu2019_MLP_metadata.pkl. Inclination
        # is still drawn separately and isotropically in cos(i).
        KN_prior = [
            Uniform(
                xmin=-3.0, xmax=-1.6989700043360187,
                naming=["log10_mej_dyn"],
            ),
            Uniform(
                xmin=-2.0, xmax=-0.8860566476931632,
                naming=["log10_mej_wind"],
            ),
            Uniform(xmin=0.0, xmax=90.0, naming=["phi"]),
        ]
    else:
        KN_prior = [
            Uniform(xmin=lower, xmax=upper, naming=[name])
            for name, (lower, upper) in metzger_prior_bounds(RUNTIME).items()
        ]
    prior = ConstrainedPrior(KN_prior)


    # Fixed intrinsic parameters. Distance/redshift are sampled separately below.

    fixed_params = {}

    if BOOM_TARGET_NONEMPTY:
        print(
            f"Generating until {N_SAMPLES} non-empty BOOM light curves are "
            f"available; physical parameter source={PARAMETER_SOURCE}..."
        )
    else:
        print(
            f"Generating {N_SAMPLES} lightcurves; "
            f"physical parameter source={PARAMETER_SOURCE}..."
        )
    print(f"Output mode: {OUTPUT_MODE}")
    print(f"Scientific table storage: {STORAGE_FORMAT}")
    print("Synthetic table layout: long-format (one row per time and band)")
    print(f"Synthetic table schema: {SYNTHETIC_SCHEMA}")
    if SYNTHETIC_CADENCE_DAYS <= 0:
        print("Synthetic CSV cadence: native FIESTA grid")
    else:
        print(f"Synthetic CSV cadence: {SYNTHETIC_CADENCE_DAYS:g} observer-frame days")
    if APPLY_LSST_REALISM:
        if OUTPUT_MODE == "all-synthetic-lsstlike-detected":
            print("Saving a synthetic CSV for every injected event")
            print(f"Saving LSST-like CSVs only for n_detected >= {MIN_DETECTIONS}")
        else:
            print(f"Saving event files only for n_detected >= {MIN_DETECTIONS}")
        print(
            "LSST-like window: valid KNe visits through "
            f"{LSSTLIKE_DAYS_AFTER_LAST_DETECTION:g} d after last detection"
        )


    seed = int(RUNTIME.seed) if RUNTIME.seed is not None else secrets.randbits(32)
    if RUNTIME.sky_seed is not None:
        sky_seed = int(RUNTIME.sky_seed)
    elif PAIRING_MODE != "none":
        sky_seed = derive_sky_seed(
            seed,
            GALACTIC_LATITUDE_MODE,
            GALACTIC_LATITUDE_THRESHOLD_DEG,
            GALACTIC_LATITUDE_MAX_DEG,
        )
    else:
        sky_seed = seed
    key = jax.random.PRNGKey(seed)
    print("intrinsic seed =", seed)
    print("sky/noise seed =", sky_seed)
    bounds = prior_bounds_dict(KN_prior)

    # The CSV subset is selected before any detectability filtering, using a
    # dedicated RNG so enabling/disabling CSV output never changes the simulated
    # physical population or its photometric noise.
    if SAVE_CSV and 0 < CSV_SAMPLE_SIZE < N_SAMPLES:
        csv_selection_rng = np.random.default_rng(seed ^ 0x435356)
        CSV_EVENT_INDICES = set(
            int(value)
            for value in csv_selection_rng.choice(
                N_SAMPLES, size=CSV_SAMPLE_SIZE, replace=False
            )
        )
    else:
        CSV_EVENT_INDICES = set(range(N_SAMPLES)) if SAVE_CSV else set()
    print(
        "Individual CSV eligibility:",
        f"{len(CSV_EVENT_INDICES):,}/{N_SAMPLES:,} injected event IDs",
    )
    if BOOM_TARGET_NONEMPTY:
        print(
            f"BOOM target: {N_SAMPLES:,} non-empty light curves; rejected "
            f"injections are replaced (safety limit={MAX_INJECTED_EVENTS:,} injections)."
        )

    # FIESTA posterior.npz stores one named 1-D array per parameter. Sampling a
    # common index preserves all posterior correlations. Systematic-error arrays in
    # the archive are intentionally ignored for light-curve generation.
    POSTERIOR_ROWS = None
    POSTERIOR_INDICES = None
    if PARAMETER_SOURCE == "posterior" and PAIRING_MODE != "reuse":
        if VARY_ONE_PARAM:
            raise ValueError("VARY_ONE_PARAM is incompatible with PARAMETER_SOURCE='posterior'")
        posterior_arrays = load_posterior_samples(POSTERIOR_FILE)
        n_posterior = len(next(iter(posterior_arrays.values())))
        posterior_rng = np.random.default_rng(seed)
        POSTERIOR_ROWS, POSTERIOR_INDICES = draw_posterior_rows(
            posterior_arrays,
            N_SAMPLES,
            posterior_rng,
        )
        print(f"Loaded {n_posterior:,} named posterior rows from {POSTERIOR_FILE}")
        print(f"Selected {N_SAMPLES:,} joint posterior rows; inclination mode={POSTERIOR_INCLINATION}")


    # Baseline parameter draw (used only when VARY_ONE_PARAM=True):
    # We draw one "reference" parameter set once, then optionally vary a single parameter across samples.
    # This helps diagnostic plots/sensitivity tests while keeping other parameters fixed.

    key, subkey0 = jax.random.split(key)
    base_sample = prior.sample(subkey0, 1)
    base_params = {name: float(base_sample[name][0]) for name in prior.naming}


    # Optionally force one intrinsic parameter to a fixed value.  Inclination is
    # sampled separately, so it is stored in fixed_params rather than base_params.
    if FIX_PARAM is not None:
        if FIX_VALUE is None:
            raise ValueError("FIX_VALUE must be provided if FIX_PARAM is not None")
        if FIX_PARAM == "inclination_EM":
            fixed_params["inclination_EM"] = float(FIX_VALUE)
        else:
            if FIX_PARAM not in base_params:
                raise ValueError(f"FIX_PARAM={FIX_PARAM} not in prior parameters")
            base_params[FIX_PARAM] = float(FIX_VALUE)

    # Build one value per event for the optional sensitivity scan.
    if VARY_ONE_PARAM:
        if VARY_PARAM not in bounds:
            raise ValueError(f"VARY_PARAM={VARY_PARAM} not in prior parameters")

        vmin, vmax = bounds[VARY_PARAM]

        if VARY_MODE == "uniform":
            key, subkeyv = jax.random.split(key)
            vary_values = np.asarray(
                jax.random.uniform(subkeyv, shape=(N_SAMPLES,), minval=vmin, maxval=vmax),
                dtype=float
            )
        elif VARY_MODE == "linspace":
            vary_values = np.linspace(vmin, vmax, N_SAMPLES, dtype=float)
        else:
            raise ValueError("VARY_MODE must be 'uniform' or 'linspace'")


    MODEL_INTRINSIC_PARAMETER_COLUMNS = {
        "Bu2026_MLP": [
            "log10_mej_dyn", "v_ej_dyn", "Ye_dyn", "log10_mej_wind",
            "v_ej_wind", "Ye_wind",
        ],
        "Bu2019_MLP": ["log10_mej_dyn", "log10_mej_wind", "phi"],
        "MetzgerModel": list(METZGER_PARAMETERS),
        "MetzgerFullModel": list(METZGER_PARAMETERS),
    }
    INTRINSIC_PARAMETER_COLUMNS = [
        *MODEL_INTRINSIC_PARAMETER_COLUMNS[MODEL_NAME],
        "inclination_EM",
        "redshift",
        "luminosity_distance",
    ]


    def draw_intrinsic_event(event_index, physical_rng):
        """Draw only the physical variables of one KNe.

        Sky position, merger epoch and photometric noise deliberately do not use
        ``physical_rng`` in paired mode.  This separation is what makes it safe to
        reuse the same intrinsic catalogue with different latitude selections.
        """
        nonlocal key
        posterior_index = np.nan
        if PARAMETER_SOURCE == "posterior":
            if POSTERIOR_ROWS is None or POSTERIOR_INDICES is None:
                raise RuntimeError(
                    "Posterior rows are unavailable while drawing a new catalogue"
                )
            if event_index < len(POSTERIOR_ROWS):
                posterior_index = int(POSTERIOR_INDICES[event_index])
                params = dict(POSTERIOR_ROWS[event_index])
            elif BOOM_TARGET_NONEMPTY:
                # Refill draws are deterministic per injected event and preserve
                # correlations by selecting one complete posterior row.
                refill_rng = np.random.default_rng(
                    (int(seed) ^ 0x424F4F4D) + int(event_index)
                )
                posterior_index = int(refill_rng.integers(0, n_posterior))
                params = {
                    name: float(values[posterior_index])
                    for name, values in posterior_arrays.items()
                }
            else:
                raise IndexError("Posterior event index exceeds the requested population")
            if POSTERIOR_INCLINATION == "isotropic":
                params.pop("inclination_EM", None)
        elif VARY_ONE_PARAM:
            if event_index >= len(vary_values):
                raise RuntimeError(
                    "Exact non-empty BOOM targeting is unavailable for the "
                    "internal one-parameter sensitivity scan"
                )
            params = dict(base_params)
            params[VARY_PARAM] = float(vary_values[event_index])
        else:
            key, subkey = jax.random.split(key)
            sample = prior.sample(subkey, 1)
            params = {name: float(sample[name][0]) for name in prior.naming}
        params.update(fixed_params)
        if "inclination_EM" not in params:
            params["inclination_EM"] = sample_inclination_for_event(
                physical_rng,
                sampling=INCLINATION_SAMPLING,
                minimum_rad=INCLINATION_MIN_RAD,
                maximum_rad=INCLINATION_MAX_RAD,
            )
        z, d_l = sample_distance_redshift_for_event(
            rng=physical_rng,
            sampler=COMOVING_VOLUME_SAMPLER,
            distance_sampling=DISTANCE_SAMPLING,
            fixed_redshift=FIXED_REDSHIFT,
            fixed_luminosity_distance_mpc=FIXED_LUMINOSITY_DISTANCE_MPC,
        )
        params["redshift"] = float(z)
        params["luminosity_distance"] = float(d_l)
        return params, posterior_index


    def intrinsic_catalog_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


    def create_intrinsic_catalog(path):
        """Draw and atomically save the physical population shared by paired runs."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        contexts = []
        for event_index in range(N_SAMPLES):
            physical_rng = np.random.default_rng(seed + event_index)
            params, posterior_index = draw_intrinsic_event(event_index, physical_rng)
            contexts.append((dict(params), posterior_index))
            row = {
                "event_id": f"{event_index:06d}",
                "event_index": int(event_index),
                "model": MODEL_NAME,
                "model_configuration_json": MODEL_CONFIGURATION_JSON,
                "intrinsic_seed": int(seed),
                "parameter_source": PARAMETER_SOURCE,
                "posterior_inclination": (
                    POSTERIOR_INCLINATION if PARAMETER_SOURCE == "posterior" else ""
                ),
                "distance_sampling": DISTANCE_SAMPLING,
                "inclination_sampling": (
                    "posterior"
                    if PARAMETER_SOURCE == "posterior"
                    and POSTERIOR_INCLINATION == "posterior"
                    else INCLINATION_SAMPLING
                ),
                "posterior_index": posterior_index,
            }
            row.update({name: float(params[name]) for name in INTRINSIC_PARAMETER_COLUMNS})
            rows.append(row)
        catalogue = pd.DataFrame(rows)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        catalogue.to_csv(temporary, index=False, float_format="%.17g")
        os.replace(temporary, destination)
        return contexts


    def load_intrinsic_catalog(path):
        """Load and strictly validate one reusable intrinsic population."""
        source = Path(path)
        catalogue = pd.read_csv(source, dtype={"event_id": str})
        required = {
            "event_id", "event_index", "intrinsic_seed", "parameter_source",
            "posterior_inclination", "distance_sampling", "inclination_sampling",
            "posterior_index",
            *INTRINSIC_PARAMETER_COLUMNS,
        }
        missing = required - set(catalogue.columns)
        if missing:
            raise ValueError(
                f"Intrinsic catalogue {source} is missing columns: {sorted(missing)}"
            )
        if len(catalogue) != N_SAMPLES:
            raise ValueError(
                f"Intrinsic catalogue contains {len(catalogue):,} events but "
                f"--n-samples={N_SAMPLES:,}. Use the catalogue size exactly."
            )
        if "model" in catalogue:
            catalogue_models = set(catalogue["model"].astype(str))
            if catalogue_models != {MODEL_NAME}:
                raise ValueError(
                    f"Catalogue model={sorted(catalogue_models)} does not match "
                    f"the selected model {MODEL_NAME!r}. Intrinsic catalogues "
                    "cannot be reused across different light-curve models."
                )
        elif MODEL_NAME != "Bu2026_MLP":
            raise ValueError(
                "This legacy intrinsic catalogue has no model column and can only "
                "be interpreted as Bu2026_MLP; select a catalogue for this model."
            )
        else:
            print(
                "Warning: legacy intrinsic catalogue has no model column; "
                "interpreting it as Bu2026_MLP."
            )
        if MODEL_NAME in ANALYTICAL_MODELS:
            if "model_configuration_json" not in catalogue:
                raise ValueError("Metzger catalogue lacks its model configuration; create a new catalogue")
            configurations = [json.loads(value) for value in catalogue["model_configuration_json"]]
            if any(value != MODEL_CONFIGURATION for value in configurations):
                raise ValueError("Metzger catalogue integration grid, model options or priors differ from this run")
        expected_indices = np.arange(N_SAMPLES, dtype=int)
        event_indices = pd.to_numeric(
            catalogue["event_index"], errors="raise"
        ).to_numpy(int)
        if not np.array_equal(event_indices, expected_indices):
            raise ValueError(
                "Intrinsic catalogue event_index must be exactly 0..N-1 in order"
            )
        expected_ids = np.array([f"{index:06d}" for index in expected_indices])
        if not np.array_equal(catalogue["event_id"].to_numpy(str), expected_ids):
            raise ValueError(
                "Intrinsic catalogue event_id must be zero-padded and aligned with "
                "event_index"
            )
        sources = set(catalogue["parameter_source"].astype(str))
        if sources != {PARAMETER_SOURCE}:
            raise ValueError(
                f"Catalogue parameter_source={sorted(sources)} does not match "
                f"the selected source {PARAMETER_SOURCE!r}"
            )
        catalogue_seeds = set(
            pd.to_numeric(catalogue["intrinsic_seed"], errors="raise").astype(int)
        )
        if catalogue_seeds != {seed}:
            raise ValueError(
                f"Catalogue intrinsic_seed={sorted(catalogue_seeds)} does not "
                f"match --seed={seed}. Reuse the original intrinsic seed."
            )
        distance_modes = set(catalogue["distance_sampling"].astype(str))
        if distance_modes != {DISTANCE_SAMPLING}:
            raise ValueError(
                f"Catalogue distance_sampling={sorted(distance_modes)} does not "
                f"match the selected mode {DISTANCE_SAMPLING!r}"
            )
        if PARAMETER_SOURCE == "posterior":
            catalogue_inclinations = set(
                catalogue["posterior_inclination"].astype(str)
            )
            if catalogue_inclinations != {POSTERIOR_INCLINATION}:
                raise ValueError(
                    "Catalogue posterior-inclination treatment does not match "
                    f"the selected mode {POSTERIOR_INCLINATION!r}"
                )
        numeric = catalogue[INTRINSIC_PARAMETER_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"Non-finite intrinsic values in {source}")
        contexts = []
        for row_index, row in catalogue.iterrows():
            params = {
                name: float(numeric.iloc[row_index][name])
                for name in INTRINSIC_PARAMETER_COLUMNS
            }
            if CATALOG_DISTANCE_POLICY == "current-fixed":
                params["redshift"] = float(FIXED_REDSHIFT)
                params["luminosity_distance"] = float(FIXED_LUMINOSITY_DISTANCE_MPC)
            if MODEL_NAME in ANALYTICAL_MODELS:
                if params["beta"] <= 0 or params["log10_vej"] >= 0:
                    raise ValueError("Invalid Metzger velocity or beta in catalogue")
                if MODEL_NAME == "MetzgerFullModel" and params["log10_vej"] >= np.log10(RUNTIME.metzger_vmax):
                    raise ValueError("Catalogue Metzger v_min must be smaller than vmax")
            posterior_value = pd.to_numeric(row["posterior_index"], errors="coerce")
            posterior_index = (
                int(posterior_value) if np.isfinite(posterior_value) else np.nan
            )
            contexts.append((params, posterior_index))
        return contexts


    INTRINSIC_CATALOG_CONTEXTS = None
    INTRINSIC_CATALOG_HASH = None
    if PAIRING_MODE == "create":
        INTRINSIC_CATALOG_CONTEXTS = create_intrinsic_catalog(
            INTRINSIC_CATALOG_PATH
        )
        INTRINSIC_CATALOG_HASH = intrinsic_catalog_sha256(INTRINSIC_CATALOG_PATH)
        print(
            f"Created paired intrinsic catalogue: {INTRINSIC_CATALOG_PATH} "
            f"({N_SAMPLES:,} events; sha256={INTRINSIC_CATALOG_HASH[:12]}...)"
        )
    elif PAIRING_MODE == "reuse":
        INTRINSIC_CATALOG_CONTEXTS = load_intrinsic_catalog(
            INTRINSIC_CATALOG_PATH
        )
        INTRINSIC_CATALOG_HASH = intrinsic_catalog_sha256(INTRINSIC_CATALOG_PATH)
        print(
            f"Reusing paired intrinsic catalogue: {INTRINSIC_CATALOG_PATH} "
            f"({N_SAMPLES:,} events; sha256={INTRINSIC_CATALOG_HASH[:12]}...)"
        )

        if CATALOG_DISTANCE_POLICY == "current-fixed":
            print(f"Catalogue ejecta retained; distance/redshift replaced by D_L={FIXED_LUMINOSITY_DISTANCE_MPC:g} Mpc, z={FIXED_REDSHIFT:.8g}")


    # =============================================================================
    # OUTPUT DIRECTORY (one folder per run, containing .csv and .png)
    # =============================================================================

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    if PAIRING_MODE == "none":
        run_dir = RUNS_ROOT / f"{RUN_NAME}_seed_{seed}"
    else:
        run_dir = RUNS_ROOT / (
            f"{RUN_NAME}_intrinsic_seed_{seed}_sky_seed_{sky_seed}"
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    outdir = run_dir / "csv"
    parquetdir = run_dir / "parquet"
    pngdir = run_dir / "png"
    boomdir = run_dir / "boom_input"
    boom_summary_rows = []
    if SAVE_BOOM_SYNTHETIC:
        boomdir.mkdir(parents=True, exist_ok=False)

    if SAVE_CSV:
        outdir.mkdir(parents=True, exist_ok=False)
    if SAVE_PARQUET:
        parquetdir.mkdir(parents=True, exist_ok=False)
    if SAVE_PNG or SAVE_PNG_LSSTLIKE or SAVE_PNG_SYNTHETIC_ONLY:
        pngdir.mkdir(parents=True, exist_ok=False)

    # Build a concise reproducibility record.  Detailed per-event quantities live
    # in summary.csv, while the complete OpSim export configuration remains beside
    # the selected OpSim visit files.
    timestamp = datetime.now().isoformat(timespec="seconds")

    population_info = {
        "light_curve_model": MODEL_NAME,
        "parameter_source": PARAMETER_SOURCE,
        "posterior_file": POSTERIOR_FILE if PARAMETER_SOURCE == "posterior" else None,
        "posterior_inclination": (
            POSTERIOR_INCLINATION if PARAMETER_SOURCE == "posterior" else None
        ),
        "broad_prior_bounds": bounds if PARAMETER_SOURCE == "broad" else None,
        "fixed_parameters": fixed_params or None,
        "pairing_mode": PAIRING_MODE,
        "catalog_distance_policy": CATALOG_DISTANCE_POLICY,
        "intrinsic_catalog": INTRINSIC_CATALOG_PATH,
        "intrinsic_catalog_sha256": INTRINSIC_CATALOG_HASH,
    }

    distance_info = {
        "sampling": DISTANCE_SAMPLING,
        "cosmology": "Astropy Planck18",
        "fixed_luminosity_distance_mpc": (
            FIXED_LUMINOSITY_DISTANCE_MPC if DISTANCE_SAMPLING == "fixed" else None
        ),
        "fixed_redshift": FIXED_REDSHIFT if DISTANCE_SAMPLING == "fixed" else None,
        "luminosity_distance_min_mpc": (
            LUMINOSITY_DISTANCE_MIN_MPC
            if DISTANCE_SAMPLING == "comoving_volume"
            else None
        ),
        "luminosity_distance_max_mpc": (
            LUMINOSITY_DISTANCE_MAX_MPC
            if DISTANCE_SAMPLING == "comoving_volume"
            else None
        ),
        "inclination_sampling": (
            "posterior"
            if PARAMETER_SOURCE == "posterior" and POSTERIOR_INCLINATION == "posterior"
            else INCLINATION_SAMPLING
        ),
    }

    output_info = {
        "mode": OUTPUT_MODE,
        "storage_format": STORAGE_FORMAT,
        "csv_selected_injected_events": len(CSV_EVENT_INDICES),
        "csv_sample_size_requested": CSV_SAMPLE_SIZE,
        "parquet_events_per_shard": (
            PARQUET_EVENTS_PER_SHARD if SAVE_PARQUET else None
        ),
        "parquet_compression": RUNTIME.parquet_compression if SAVE_PARQUET else None,
        "opsim_cache_size": OPSIM_CACHE_SIZE,
        "surrogate_batch_size_requested": SURROGATE_BATCH_SIZE,
        "synthetic_layout": "long-format",
        "synthetic_schema": SYNTHETIC_SCHEMA,
        "synthetic_cadence_days": (
            "native FIESTA grid"
            if SYNTHETIC_CADENCE_DAYS <= 0
            else SYNTHETIC_CADENCE_DAYS
        ),
        "synthetic_depth_cut": SYNTHETIC_DEPTH_CUT,
        "synthetic_m5_medians": SYNTHETIC_M5_MEDIANS or None,
        "synthetic_m5_medians_source": SYNTHETIC_M5_MEDIANS_SOURCE,
        "synthetic_magnitude_saved_in_mag": "model",
        "synthetic_depth_comparison_column": (
            "mag" if SYNTHETIC_DEPTH_CUT == "m5-median" else None
        ),
        "bu2026_numerical_floor_absolute_mag": (
            NUMERICAL_FLOOR_ABSOLUTE_MAG if APPLY_NUMERICAL_SUPPORT_CUT else None
        ),
        "numerical_floor_safety_margin_mag": (
            NUMERICAL_FLOOR_MARGIN_MAG if APPLY_NUMERICAL_SUPPORT_CUT else None
        ),
        "faintest_supported_absolute_mag": (
            NUMERICAL_SUPPORT_ABSOLUTE_MAG if APPLY_NUMERICAL_SUPPORT_CUT else None
        ),
        "apparent_cut_formula": (
            "M_floor - margin + 5*log10(D_L/Mpc) + 25"
            if APPLY_NUMERICAL_SUPPORT_CUT else None
        ),
        "numerical_floor_cut_applied_before_m5": (
            APPLY_NUMERICAL_SUPPORT_CUT
        ),
        "plot_mode": RUNTIME.plot_mode,
        "max_plot_events_per_product": MAX_PLOT_EVENTS,
        "requested_population_counting_unit": (
            "nonempty_boom_lightcurves"
            if BOOM_TARGET_NONEMPTY else "injected_physical_events"
        ),
        "minimum_detections": (
            MIN_DETECTIONS
            if APPLY_LSST_REALISM
            else None
        ),
    }

    run_info_sections = {
        "run": {
            "timestamp": timestamp,
            "seed": seed,
            "intrinsic_seed": seed,
            "sky_noise_seed": sky_seed,
            "name": RUN_NAME,
            "n_events": N_SAMPLES,
            "n_events_requested": N_SAMPLES,
            "requested_population_counting_unit": (
                "nonempty_boom_lightcurves"
                if BOOM_TARGET_NONEMPTY else "injected_physical_events"
            ),
        },
        "model": {
            "surrogate": model.name,
            "backend": MODEL_BACKEND,
            **MODEL_METADATA,
            "fiesta_surrogate_api": FIESTA_SURROGATE_API,
            "explicit_surrogate_directory": surrogate_directory,
            "filter_system": FILTER_SYSTEM,
            "photometric_systems_saved": FILTER_SYSTEMS,
            "filters": FILTERS,
        },
        "population": population_info,
        "distance_and_orientation": distance_info,
        "output": output_info,
    }

    if SAVE_BOOM_SYNTHETIC:
        run_info_sections["boom_synthetic_export"] = {
            "directory": str(boomdir),
            "export_mode": BOOM_EXPORT_MODE,
            "m5_by_band": BOOM_M5,
            "gamma_by_band": BOOM_GAMMA,
            "sigma_sys_mag": BOOM_SIGMA_SYS,
            "selection_rule": (
                "every finite model point; no m5 selection"
                if BOOM_EXPORT_MODE == "complete"
                else "finite model mag <= m5[band]"
            ),
            "error_model": (
                "Ivezic et al. 2019 equations 4-5; m5 affects uncertainty "
                "but is not a selection threshold"
                if BOOM_EXPORT_MODE == "complete"
                else "Ivezic et al. 2019 equations 4-5"
            ),
            "nondetection_encoding": (
                "not applicable; depth selection disabled"
                if BOOM_EXPORT_MODE == "complete"
                else "not exported to BOOM; retained as synthetic magnitudes for envelope analysis"
            ),
            "random_noise_added": False,
            "cadence": "synthetic grid; assumed observations, NOT actual OpSim visits",
            "duplicate_shift_seconds": BOOM_SHIFT_SECONDS,
            "event_selection": (
                "inject exactly n_events_requested physical KNe and export every "
                "event with finite numerical model support"
                if BOOM_EXPORT_MODE == "complete"
                else "inject until exactly n_events_requested synthetic products "
                "have >=1 detection; conditional visibility-selected benchmark"
                if BOOM_TARGET_NONEMPTY
                else "inject exactly n_events_requested physical KNe; export "
                "the subset with >=1 detection"
            ),
            "target_nonempty_lightcurves": (
                N_SAMPLES if BOOM_TARGET_NONEMPTY else None
            ),
            "maximum_injection_factor": (
                BOOM_MAX_INJECTION_FACTOR if BOOM_TARGET_NONEMPTY else None
            ),
            "numerical_support": "invalid/removed model points are NOT upper limits",
            "reference": (
                "Ivezic et al. 2019, ApJ 873, 111, equations 4-5, Table 2"
            ),
        }

    if USE_OPSIM_CONTEXT:
        run_info_sections["opsim_context"] = {
            "directory": OPSIM_DIR,
            "opsim_dir": OPSIM_DIR,
            "opsim_dir_resolved": OPSIM_DIR.resolve(),
            "field_files_found": N_FIELDS,
            "survey_summary": OPSIM_RUN_METADATA.get("opsim_survey_summary_file"),
            "merger_time_sampling": T0_MODE,
            "galactic_latitude_mode": GALACTIC_LATITUDE_MODE,
            "galactic_latitude_threshold_deg": GALACTIC_LATITUDE_THRESHOLD_DEG,
            "galactic_latitude_min_deg": GALACTIC_LATITUDE_MIN_DEG,
            "galactic_latitude_max_deg": GALACTIC_LATITUDE_MAX_DEG,
            "eligible_field_files": len(ELIGIBLE_OPSIM_FILES),
        }

    if APPLY_LSST_REALISM:
        run_info_sections["lsstlike_observations"] = {
            "detection_snr": DET_SNR,
            "window_anchor": "last_detection",
            "days_after_last_detection": LSSTLIKE_DAYS_AFTER_LAST_DETECTION,
            "rows_outside_finite_fiesta_truth_removed": True,
        }


    if VARY_ONE_PARAM:
        run_info_sections["sensitivity_scan"] = {
            "parameter": VARY_PARAM,
            "mode": VARY_MODE,
            "bounds": bounds[VARY_PARAM],
            "n_values": len(vary_values),
            "reference_parameters": {
                name: value for name, value in base_params.items() if name != VARY_PARAM
            },
        }

    (run_dir / "run_info.txt").write_text(
        format_run_info_sections(run_info_sections)
    )

    # =============================================================================
    # MAIN LOOP: generate fixed-N injections or refill until the BOOM target
    # =============================================================================

    # -------------------------------------------------------------------------
    # OPTIMIZATION: OpSim cache (copies preserve event-level independence)
    # -------------------------------------------------------------------------
    class OpsimVisitCache:
        """Small least-recently-used cache of fully prepared OpSim field tables.

        The returned frame is always a deep copy: event-level processing may add or
        modify columns without mutating the cached canonical table. A size of zero
        disables caching while preserving identical scientific behaviour.
        """

        def __init__(self, maxsize):
            self.maxsize = int(maxsize)
            self._tables = OrderedDict()
            self.hits = 0
            self.misses = 0

        def get(self, path):
            cache_key = str(Path(path).expanduser().resolve())
            if self.maxsize > 0 and cache_key in self._tables:
                self.hits += 1
                table = self._tables.pop(cache_key)
                self._tables[cache_key] = table
                return table.copy(deep=True)

            self.misses += 1
            table = normalize_opsim_visit_columns(pd.read_csv(path))
            table = enrich_visits_with_field_index_metadata(
                table,
                Path(path),
                OPSIM_FIELDS_INDEX,
            )
            if self.maxsize > 0:
                self._tables[cache_key] = table
                while len(self._tables) > self.maxsize:
                    self._tables.popitem(last=False)
            return table.copy(deep=True)


    class ParquetShardWriter:
        """Write a long-format population dataset in interruption-safe shards.

        Each input frame receives a repeated integer ``event_id`` column. Frames
        are buffered by event count, concatenated, and written atomically through a
        temporary file. Completed shards therefore remain usable if a later model
        evaluation fails or a long run is interrupted.
        """

        def __init__(self, directory, product, events_per_shard, compression):
            self.directory = Path(directory)
            self.product = str(product)
            self.events_per_shard = int(events_per_shard)
            self.compression = compression
            self.frames = []
            self.event_ids = []
            self.shard_index = 0
            self.manifest_rows = []

        def add(self, event_id, frame):
            out = frame.copy()
            if "event_id" in out.columns:
                out["event_id"] = int(event_id)
            else:
                out.insert(0, "event_id", int(event_id))
            # Repeating objectid makes every row self-contained for Arrow filters.
            if "objectid" in out.columns:
                out["objectid"] = f"{int(event_id):04d}"
            self.frames.append(out)
            self.event_ids.append(int(event_id))
            if len(self.event_ids) >= self.events_per_shard:
                self.flush()

        def flush(self):
            if not self.frames:
                return
            # pandas 3 uses Copy-on-Write; the legacy ``copy=False`` argument is
            # deprecated and no longer changes the eager/lazy copy policy.
            combined = pd.concat(self.frames, ignore_index=True)
            filename = f"{self.product}_part_{self.shard_index:05d}.parquet"
            destination = self.directory / filename
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            combined.to_parquet(
                temporary,
                engine="pyarrow",
                compression=self.compression,
                index=False,
            )
            os.replace(temporary, destination)
            self.manifest_rows.append({
                "product": self.product,
                "file": filename,
                "event_id_min": min(self.event_ids),
                "event_id_max": max(self.event_ids),
                "N_events": len(self.event_ids),
                "N_rows": len(combined),
            })
            self.frames.clear()
            self.event_ids.clear()
            self.shard_index += 1

        def close(self):
            self.flush()
            return list(self.manifest_rows)


    def prepare_event_parameter_context(event_index):
        """Prepare one physical event and its sky/noise random stream.

        Legacy runs keep the historical single per-event NumPy stream. Paired
        runs load the immutable physical vector from the shared catalogue and use
        ``sky_seed`` only for OpSim field choice, merger epoch and noise.
        """
        if INTRINSIC_CATALOG_CONTEXTS is not None:
            stored_params, stored_posterior_index = INTRINSIC_CATALOG_CONTEXTS[
                event_index
            ]
            params = dict(stored_params)
            posterior_index = stored_posterior_index
            rng_i = np.random.default_rng(sky_seed + event_index)
        else:
            rng_i = np.random.default_rng(seed + event_index)
            params, posterior_index = draw_intrinsic_event(event_index, rng_i)
        return {
            "event_index": int(event_index),
            "params": params,
            "posterior_index": posterior_index,
            "rng": rng_i,
        }


    # -------------------------------------------------------------------------
    # OPTIMIZATION: validated surrogate batching, with scalar fallback
    # -------------------------------------------------------------------------
    def _split_batched_array(values, batch_size, name, allow_shared=False):
        """Normalize a possible batch-first or batch-last surrogate array."""
        array = np.asarray(values)
        if allow_shared and array.ndim == 1:
            return [array.copy() for _ in range(batch_size)]
        if array.ndim >= 2 and array.shape[0] == batch_size:
            return [np.asarray(array[index]).squeeze() for index in range(batch_size)]
        if array.ndim >= 2 and array.shape[-1] == batch_size:
            return [np.asarray(array[..., index]).squeeze() for index in range(batch_size)]
        raise ValueError(
            f"Cannot identify batch axis in {name}: shape={array.shape}, "
            f"batch_size={batch_size}"
        )


    def normalize_batched_prediction(prediction, batch_size):
        """Convert one vectorized ``model.vpredict`` result to scalar-like pairs."""
        times, magnitudes = prediction
        if not isinstance(magnitudes, dict):
            raise TypeError("Batched surrogate magnitudes are not a dictionary")
        time_items = _split_batched_array(
            times, batch_size, "times", allow_shared=True
        )
        magnitude_items = [dict() for _ in range(batch_size)]
        for filter_name, values in magnitudes.items():
            split_values = _split_batched_array(
                values, batch_size, f"magnitudes[{filter_name}]", allow_shared=False
            )
            for index, value in enumerate(split_values):
                magnitude_items[index][filter_name] = value
        return list(zip(time_items, magnitude_items))


    def predictions_agree(reference, candidate, rtol=1e-6, atol=1e-7):
        """Require identical structure and numerical agreement for batch opt-in."""
        ref_times, ref_magnitudes = reference
        test_times, test_magnitudes = candidate
        if set(ref_magnitudes) != set(test_magnitudes):
            return False
        if not np.allclose(
            np.asarray(ref_times), np.asarray(test_times),
            rtol=rtol, atol=atol, equal_nan=True,
        ):
            return False
        return all(
            np.allclose(
                np.asarray(ref_magnitudes[name]),
                np.asarray(test_magnitudes[name]),
                rtol=rtol, atol=atol, equal_nan=True,
            )
            for name in ref_magnitudes
        )


    class ValidatedSurrogatePredictor:
        """Use FIESTA's vectorized API when it matches scalar predictions.

        No unvalidated batch prediction reaches the scientific products. The first
        successful vectorized call is compared with scalar predictions for up to
        two events. An exception, an unrecognized output shape, or a numerical
        mismatch permanently selects the scalar path for the remainder of the run.

        Important
        ---------
        A dictionary containing arrays must be passed to ``surrogate.vpredict``.
        Passing it to scalar ``surrogate.predict`` makes the physical-parameter
        axis collide with the batch dimension. ``vpredict`` applies
        the *same* scalar ``predict`` function through ``jax.vmap``; it changes only
        how independent events are evaluated, not the surrogate or its physics.
        """

        def __init__(self, surrogate, requested_batch_size):
            self.surrogate = surrogate
            self.requested_batch_size = int(requested_batch_size)
            self.batch_supported = None if self.requested_batch_size > 1 else False
            self.fallback_reason = "batch size is 1" if self.requested_batch_size == 1 else ""
            self.scalar_calls = 0
            self.batch_calls = 0

        def _scalar(self, params):
            self.scalar_calls += 1
            return self.surrogate.predict(params)

        def predict_contexts(self, contexts):
            if self.batch_supported is False or len(contexts) == 1:
                return [self._scalar(context["params"]) for context in contexts]

            parameter_names = tuple(contexts[0]["params"])
            if any(tuple(context["params"]) != parameter_names for context in contexts):
                self.batch_supported = False
                self.fallback_reason = "parameter dictionaries do not share one schema"
                return [self._scalar(context["params"]) for context in contexts]
            batched_params = {
                name: np.asarray([context["params"][name] for context in contexts])
                for name in parameter_names
            }
            try:
                self.batch_calls += 1
                vectorized_predict = getattr(self.surrogate, "vpredict", None)
                if vectorized_predict is None:
                    raise AttributeError(
                        "the installed FIESTA surrogate has no vpredict method"
                    )
                candidates = normalize_batched_prediction(
                    vectorized_predict(batched_params), len(contexts)
                )
                if self.batch_supported is None:
                    check_count = min(2, len(contexts))
                    references = [
                        self._scalar(contexts[index]["params"])
                        for index in range(check_count)
                    ]
                    if not all(
                        predictions_agree(references[index], candidates[index])
                        for index in range(check_count)
                    ):
                        raise ValueError(
                            "vectorized predictions disagree with scalar references"
                        )
                    self.batch_supported = True
                    print(
                        "Validated vectorized model prediction against scalar "
                        f"references; batch size={len(contexts)}"
                    )
                return candidates
            except Exception as exc:
                self.batch_supported = False
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                print(
                    "Model batch mode unavailable or inconsistent; using the "
                    f"validated scalar path. Reason: {self.fallback_reason}"
                )
                return [self._scalar(context["params"]) for context in contexts]


    def iter_event_predictions():
        """Yield predictions until the injected or non-empty BOOM target is met."""
        start = 0
        while start < MAX_INJECTED_EVENTS:
            if BOOM_TARGET_NONEMPTY and len(boom_summary_rows) >= N_SAMPLES:
                return
            stop = min(start + SURROGATE_BATCH_SIZE, MAX_INJECTED_EVENTS)
            contexts = [
                prepare_event_parameter_context(event_index)
                for event_index in range(start, stop)
            ]
            predictions = SURROGATE_PREDICTOR.predict_contexts(contexts)
            for context, prediction in zip(contexts, predictions):
                if BOOM_TARGET_NONEMPTY and len(boom_summary_rows) >= N_SAMPLES:
                    return
                yield context, prediction
            start = stop
        if BOOM_TARGET_NONEMPTY and len(boom_summary_rows) < N_SAMPLES:
            raise RuntimeError(
                "Could not reach the requested non-empty BOOM population: "
                f"{len(boom_summary_rows):,}/{N_SAMPLES:,} accepted after "
                f"{MAX_INJECTED_EVENTS:,} injected events. Increase "
                "boom_max_injection_factor or review the m5/distance settings."
            )


    OPSIM_VISIT_CACHE = OpsimVisitCache(OPSIM_CACHE_SIZE)
    SURROGATE_PREDICTOR = ValidatedSurrogatePredictor(model, SURROGATE_BATCH_SIZE)
    SYNTHETIC_PARQUET_WRITER = (
        ParquetShardWriter(
            parquetdir, "synthetic", PARQUET_EVENTS_PER_SHARD, PARQUET_COMPRESSION
        )
        if SAVE_PARQUET else None
    )
    LSSTLIKE_PARQUET_WRITER = (
        ParquetShardWriter(
            parquetdir, "lsstlike", PARQUET_EVENTS_PER_SHARD, PARQUET_COMPRESSION
        )
        if SAVE_PARQUET else None
    )

    def build_event_summary(
        event_index,
        params,
        posterior_index,
        field_idx,
        t0_mjd,
        field_metadata,
        detection_evaluated,
        n_detected,
        t_first_det_days,
        t_last_det_days,
        t_first_nondet_days,
        lsstlike_window_end_days,
        synthetic_csv_saved,
        lsstlike_csv_saved,
        synthetic_parquet_saved,
        lsstlike_parquet_saved,
        numerical_support_cut_mag,
    ):
        """Build one summary row; every injected event gets exactly one row."""
        detected_any = int(n_detected > 0) if detection_evaluated else np.nan
        return {
            "i": int(event_index),
            "event_id": f"{int(event_index):06d}",
            "objectid": f"{int(event_index):04d}",
            "pairing_mode": PAIRING_MODE,
            "catalog_distance_policy": CATALOG_DISTANCE_POLICY,
            "intrinsic_catalog_sha256": INTRINSIC_CATALOG_HASH or "",
            "intrinsic_seed": int(seed),
            "sky_noise_seed": int(sky_seed),
            "field_idx": field_idx,
            "t0_mjd": t0_mjd,
            "detected_any": detected_any,
            "n_detected": int(n_detected) if detection_evaluated else np.nan,
            "t_first_det_days": t_first_det_days,
            "last_detection_time_since_merger_days": t_last_det_days,
            "t_first_nondet_days": t_first_nondet_days,
            "lsstlike_data_end_time_since_merger_days": lsstlike_window_end_days,
            "detection_evaluated": bool(detection_evaluated),
            "output_mode": OUTPUT_MODE,
            "model": MODEL_NAME,
            "model_backend": MODEL_BACKEND,
            "model_configuration_json": MODEL_CONFIGURATION_JSON,
            "inclination_used_by_model": MODEL_NAME not in ANALYTICAL_MODELS,
            **{name: float(params.get(name, np.nan)) for name in METZGER_PARAMETERS},
            "synthetic_csv_saved": bool(synthetic_csv_saved),
            "lsstlike_csv_saved": bool(lsstlike_csv_saved),
            "synthetic_parquet_saved": bool(synthetic_parquet_saved),
            "lsstlike_parquet_saved": bool(lsstlike_parquet_saved),
            "luminosity_distance": float(params["luminosity_distance"]),
            "numerical_support_cut_apparent_mag_no_mw": float(
                numerical_support_cut_mag
            ),
            "redshift": float(params["redshift"]),
            "posterior_index": posterior_index,
            "inclination_EM": float(params["inclination_EM"]),
            "cos_inclination_EM": float(np.cos(params["inclination_EM"])),
            "inclination_sampling": (
                "posterior"
                if PARAMETER_SOURCE == "posterior" and POSTERIOR_INCLINATION == "posterior"
                else INCLINATION_SAMPLING
            ),
            "log10_mej_dyn": float(params.get("log10_mej_dyn", np.nan)),
            "v_ej_dyn": float(params.get("v_ej_dyn", np.nan)),
            "Ye_dyn": float(params.get("Ye_dyn", np.nan)),
            "log10_mej_wind": float(params.get("log10_mej_wind", np.nan)),
            "v_ej_wind": float(params.get("v_ej_wind", np.nan)),
            "Ye_wind": float(params.get("Ye_wind", np.nan)),
            "phi_deg": float(params.get("phi", np.nan)),
            "KNtheta_deg": float(np.degrees(params["inclination_EM"])),
            "fieldRA": float(field_metadata.get("fieldRA", np.nan)),
            "fieldDec": float(field_metadata.get("fieldDec", np.nan)),
            "galactic_longitude_deg": float(
                field_metadata.get("galactic_longitude_deg", np.nan)
            ),
            "galactic_latitude_deg": float(
                field_metadata.get("galactic_latitude_deg", np.nan)
            ),
            "abs_galactic_latitude_deg": float(
                field_metadata.get("abs_galactic_latitude_deg", np.nan)
            ),
            "target_name": str(field_metadata.get("target_name", "")),
        }


    summaries = []
    n_synthetic_plots = 0
    n_boom_plots = 0
    n_lsstlike_plots = 0

    generation_started = time.perf_counter()

    for event_context, prediction in iter_event_predictions():
        i = int(event_context["event_index"])
        params = event_context["params"]
        posterior_index = event_context["posterior_index"]
        rng_i = event_context["rng"]
        times, raw_mag = prediction
        times = saved_observer_times(times, params, MODEL_NAME)
        d_l = float(params["luminosity_distance"])
        if APPLY_NUMERICAL_SUPPORT_CUT:
            event_numerical_support_cut_mag = numerical_support_cut_apparent_mag(d_l)
            mag = mask_bands_after_numerical_floor(
                raw_mag,
                apparent_cut_mag=event_numerical_support_cut_mag,
            )
        else:
            event_numerical_support_cut_mag = np.inf
            mag = {
                filter_name: np.asarray(values, dtype=float)
                for filter_name, values in raw_mag.items()
            }

        # Per-event output/context defaults. In modes without LSST realism,
        # detection fields remain unavailable (NaN), not false detections.
        field_idx = np.nan
        t0 = np.nan
        visits = None
        field_metadata = {
            "fieldRA": np.nan,
            "fieldDec": np.nan,
            "target_name": "",
        }

        detection_evaluated = False
        n_det = 0
        t_first = np.nan
        t_last = np.nan
        t_first_nondet = np.nan
        lsstlike_window_anchor = np.nan
        lsstlike_window_end = np.nan
        lsstlike_window_anchor_found = False
        csv_selected = (
            i in CSV_EVENT_INDICES
            or (
                BOOM_TARGET_NONEMPTY
                and SAVE_CSV
                and CSV_SAMPLE_SIZE == 0
            )
        )
        save_synthetic_csv = OUTPUT_MODE in {
            "all-synthetic-lsstlike-detected",
            "synthetic-mjd",
        }
        save_lsstlike_csv = False

        # OpSim supplies a fixed sky position and a survey time range. In
        # synthetic-mjd mode it is metadata only: its visit cadence/depth is never
        # applied to the FIESTA curve.
        if USE_OPSIM_CONTEXT:
            selected_position = int(rng_i.integers(0, len(ELIGIBLE_OPSIM_FILES)))
            field_idx, selected_opsim_file = ELIGIBLE_OPSIM_FILES[selected_position]
            visits = OPSIM_VISIT_CACHE.get(selected_opsim_file)
            field_metadata = summarize_field(visits)
            ra_event = float(field_metadata.get("fieldRA", np.nan))
            dec_event = float(field_metadata.get("fieldDec", np.nan))
            if np.isfinite(ra_event) and np.isfinite(dec_event):
                gal_event = SkyCoord(
                    ra=ra_event * u.deg, dec=dec_event * u.deg, frame="icrs"
                ).galactic
                field_metadata["galactic_longitude_deg"] = float(gal_event.l.deg)
                field_metadata["galactic_latitude_deg"] = float(gal_event.b.deg)
                field_metadata["abs_galactic_latitude_deg"] = abs(float(gal_event.b.deg))

            mjd_values = pd.to_numeric(visits["observationStartMJD"], errors="coerce").to_numpy(float)
            mjd_values = mjd_values[np.isfinite(mjd_values)]
            if len(mjd_values) == 0:
                raise ValueError(f"No finite observationStartMJD in {selected_opsim_file}")
            mjd_min = float(np.min(mjd_values))
            mjd_max = float(np.max(mjd_values))
            if T0_MODE == "uniform":
                t0 = float(rng_i.uniform(mjd_min, mjd_max))
            elif T0_MODE == "visit_choice":
                t0 = float(rng_i.choice(mjd_values))
            else:
                raise ValueError("T0_MODE must be 'uniform' or 'visit_choice'")

            if OUTPUT_MODE in {
                "all-synthetic-lsstlike-detected",
                "detected-pairs",
                "synthetic-mjd",
            }:
                ra = float(field_metadata.get("fieldRA", np.nan))
                dec = float(field_metadata.get("fieldDec", np.nan))
                if not (np.isfinite(ra) and np.isfinite(dec)):
                    raise ValueError(
                        "synthetic-mjd needs fieldRA/fieldDec either in each OpSim visit CSV "
                        "or as ra0_deg/dec0_deg in opsim_fields_index.csv."
                    )
        if APPLY_LSST_REALISM:
            # Convert the dense model into one simulated measurement per
            # post-merger OpSim visit. Retain only visits supported by the finite
            # KNe truth and stop 10 days after the last detection. Candidate
            # identifiers are assigned after this final row selection.
            obs = apply_lsst_realism_to_fiesta(
                times_model=np.asarray(times, float),
                mag_model_dict=mag,
                visits_df=visits,
                t0_mjd=t0,
                det_snr=DET_SNR,
                mag_truth_cap=event_numerical_support_cut_mag,
                rng=rng_i,
            )
            (
                obs,
                lsstlike_window_anchor,
                lsstlike_window_end,
                lsstlike_window_anchor_found,
            ) = trim_lsstlike_after_last_detection(
                obs,
                days_after_last_detection=LSSTLIKE_DAYS_AFTER_LAST_DETECTION,
            )
            obs = add_lsstlike_candidate_ids(obs, event_index=i)

            det_mask = obs["detected"].to_numpy(bool)
            nondet_mask = ~det_mask
            n_det = int(det_mask.sum())
            det_any = int(n_det > 0)
            t_first_nondet = (
                float(obs.loc[nondet_mask, "t_days"].min())
                if np.any(nondet_mask)
                else np.nan
            )
            detection_evaluated = True

            if OUTPUT_MODE == "all-synthetic-lsstlike-detected":
                save_lsstlike_csv = n_det >= MIN_DETECTIONS
                save_synthetic_csv = True
            elif OUTPUT_MODE == "detected-pairs":
                save_lsstlike_csv = n_det >= MIN_DETECTIONS
                save_synthetic_csv = save_lsstlike_csv
            elif OUTPUT_MODE == "lsstlike-detected":
                save_lsstlike_csv = n_det >= MIN_DETECTIONS
                save_synthetic_csv = False

            if SAVE_CSV and csv_selected and save_lsstlike_csv:
                obs.to_csv(outdir / f"lightcurve_LSSTlike_{i:04d}.csv", index=False)
            if SAVE_PARQUET and save_lsstlike_csv:
                LSSTLIKE_PARQUET_WRITER.add(i, obs)

            t_first = float(obs.loc[det_mask, "t_days"].min()) if det_any else np.nan
            t_last  = float(obs.loc[det_mask, "t_days"].max()) if det_any else np.nan

            summaries.append(
                build_event_summary(
                    event_index=i,
                    params=params,
                    posterior_index=posterior_index,
                    field_idx=field_idx,
                    t0_mjd=t0,
                    field_metadata=field_metadata,
                    detection_evaluated=detection_evaluated,
                    n_detected=n_det,
                    t_first_det_days=t_first,
                    t_last_det_days=t_last,
                    t_first_nondet_days=t_first_nondet,
                    lsstlike_window_end_days=lsstlike_window_end,
                    synthetic_csv_saved=SAVE_CSV and csv_selected and save_synthetic_csv,
                    lsstlike_csv_saved=SAVE_CSV and csv_selected and save_lsstlike_csv,
                    synthetic_parquet_saved=SAVE_PARQUET and save_synthetic_csv,
                    lsstlike_parquet_saved=SAVE_PARQUET and save_lsstlike_csv,
                    numerical_support_cut_mag=event_numerical_support_cut_mag,
                )
            )

            if SAVE_PNG and save_lsstlike_csv:
                plot_synthetic_and_lsstlike(
                    times=times,
                    mag_dict=mag,
                    obs=obs,
                    filters=FILTERS,
                    filter_colors=FILTER_COLORS,
                    mag_lim=event_numerical_support_cut_mag,
                    outfile=pngdir / f"lightcurve_overlay_{i:04d}.png"
                )


            if (
                SAVE_PNG_LSSTLIKE
                and save_lsstlike_csv
                and (MAX_PLOT_EVENTS == 0 or n_lsstlike_plots < MAX_PLOT_EVENTS)
            ):
                plot_lsstlike_only(
                    observations=obs,
                    filters=FILTERS,
                    filter_colors=FILTER_COLORS,
                    event_index=i,
                    outfile=pngdir / f"lightcurve_LSSTlike_{i:04d}.png",
                    tmax_days=TMAX_PLOT_DAYS,
                )
                n_lsstlike_plots += 1

        if not APPLY_LSST_REALISM:
            summaries.append(
                build_event_summary(
                    event_index=i,
                    params=params,
                    posterior_index=posterior_index,
                    field_idx=field_idx,
                    t0_mjd=t0,
                    field_metadata=field_metadata,
                    detection_evaluated=False,
                    n_detected=0,
                    t_first_det_days=np.nan,
                    t_last_det_days=np.nan,
                    t_first_nondet_days=np.nan,
                    lsstlike_window_end_days=np.nan,
                    synthetic_csv_saved=SAVE_CSV and csv_selected and save_synthetic_csv,
                    lsstlike_csv_saved=False,
                    synthetic_parquet_saved=SAVE_PARQUET and save_synthetic_csv,
                    lsstlike_parquet_saved=False,
                    numerical_support_cut_mag=event_numerical_support_cut_mag,
                )
            )

        # Resampling, when requested, affects only this saved synthetic table.
        # The LSST-like calculation above always used the native FIESTA grid.
        t, synthetic_mag = resample_synthetic_output_grid(
            times_model=times,
            mag_model_dict=mag,
            filters=FILTERS,
            cadence_days=SYNTHETIC_CADENCE_DAYS,
        )

        # Build the saved long-format table directly: one row per finite time-band
        # measurement.  Candidate IDs are assigned only after the faint-magnitude
        # cut and removal of non-finite intrinsic magnitudes.
        objectid = f"{i:04d}"
        time_mjd = None
        if OUTPUT_MODE in {
            "all-synthetic-lsstlike-detected",
            "detected-pairs",
            "synthetic-mjd",
        }:
            time_mjd = float(t0) + t

        # Build one table per system, then concatenate them.  Both systems use the
        # exact same physical vector, distance, merger epoch and OpSim field.  The
        # explicit ``photometric_system`` column prevents identically named g/r/i
        # bands from ever being mixed by downstream code.
        system_tables = []
        for photometric_system in FILTER_SYSTEMS:
            system_filter_by_band = FILTER_BY_SYSTEM[photometric_system]
            system_filters = list(system_filter_by_band.values())
            system_table = build_synthetic_measurement_table(
                objectid=objectid,
                times_days=t,
                magnitudes_by_filter=synthetic_mag,
                filters=system_filters,
                filter_by_band=system_filter_by_band,
                time_mjd=time_mjd,
                magnitude_limit=event_numerical_support_cut_mag,
            )
            system_table.insert(2, "photometric_system", photometric_system)
            system_tables.append(system_table)

        df = pd.concat(system_tables, ignore_index=True) if system_tables else pd.DataFrame()
        # Preserve the legacy point-candidate contract: objectid appears once and
        # candidate IDs are unique even when the two systems share a band letter.
        if not df.empty:
            df["objectid"] = ""
            df.at[0, "objectid"] = objectid
            df["candidate_id"] = [
                f"{objectid}_{measurement_index}"
                for measurement_index in range(1, len(df) + 1)
            ]

        if SAVE_BOOM_SYNTHETIC and save_synthetic_csv:
            boom_event_index = len(boom_summary_rows)
            boom_rows = synthetic_to_boom_csv(
                df, boom_event_index, BOOM_M5, BOOM_SIGMA_SYS,
                BOOM_SHIFT_SECONDS, BOOM_EXPORT_MODE,
            )
            if BOOM_EXPORT_MODE == "complete" and boom_rows.empty:
                raise RuntimeError(
                    "Complete truth-level BOOM export found no finite model "
                    f"point for injected event {i}. The run is stopped rather "
                    "than silently producing an incomplete test population."
                )
            summaries[-1]["boom_n_export_rows"] = len(boom_rows)
            summaries[-1]["boom_n_detected"] = (
                len(boom_rows) if BOOM_EXPORT_MODE == "m5-detected" else np.nan
            )
            summaries[-1]["boom_csv_saved"] = not boom_rows.empty
            summaries[-1]["boom_event_index"] = (
                int(boom_event_index) if not boom_rows.empty else np.nan
            )
            if not boom_rows.empty:
                boom_rows.to_csv(
                    boomdir / f"lightcurve_{boom_event_index:04d}.csv", index=False
                )
            boom_detected = boom_rows["mag"].notna()
            boom_summary = dict(summaries[-1])
            # Keep upstream OpSim-based selection statistics clearly separate.
            for name in ("n_detected", "detected_any", "detection_evaluated",
                         "t_first_det_days", "last_detection_time_since_merger_days",
                         "t_first_nondet_days", "lsstlike_data_end_time_since_merger_days",
                         "output_mode"):
                boom_summary[f"upstream_{name}"] = boom_summary.get(name)
            boom_summary.update({
                "injected_event_index": int(i),
                "boom_event_index": int(boom_event_index),
                "objectid": int(boom_event_index),
                "n_detected": (
                    int(boom_detected.sum())
                    if BOOM_EXPORT_MODE == "m5-detected" else np.nan
                ),
                "detected_any": (
                    int(boom_detected.any())
                    if BOOM_EXPORT_MODE == "m5-detected" else np.nan
                ),
                "detection_evaluated": BOOM_EXPORT_MODE == "m5-detected",
                "n_export_rows": len(boom_rows),
                "empty_lightcurve": boom_rows.empty,
                "t_first_exported_days": boom_rows.loc[boom_detected, "t_days"].min(),
                "t_last_exported_days": boom_rows.loc[boom_detected, "t_days"].max(),
                "t_first_det_days": (
                    boom_rows.loc[boom_detected, "t_days"].min()
                    if BOOM_EXPORT_MODE == "m5-detected" else np.nan
                ),
                "last_detection_time_since_merger_days": (
                    boom_rows.loc[boom_detected, "t_days"].max()
                    if BOOM_EXPORT_MODE == "m5-detected" else np.nan
                ),
                "t_first_nondet_days": np.nan,
                "lsstlike_data_end_time_since_merger_days": np.nan,
                "output_mode": (
                    "synthetic-boom-complete-truth"
                    if BOOM_EXPORT_MODE == "complete"
                    else "synthetic-boom-detections-only"
                ),
                "boom_csv_saved": True,
            })
            if not boom_rows.empty:
                boom_summary_rows.append(boom_summary)
            if SAVE_PNG_SYNTHETIC_ONLY and (MAX_PLOT_EVENTS == 0 or n_boom_plots < MAX_PLOT_EVENTS):
                n_boom_plots += int(plot_boom_measurements(
                    boom_rows,
                    pngdir / f"lightcurve_boom_{boom_event_index:04d}.png",
                ))

        if SYNTHETIC_DEPTH_CUT == "m5-median":
            df = apply_synthetic_m5_median_cut(
                measurements=df,
                objectid=objectid,
                m5_medians=SYNTHETIC_M5_MEDIANS,
            )

        if (
            SAVE_PNG_SYNTHETIC_ONLY
            and save_synthetic_csv
            and (MAX_PLOT_EVENTS == 0 or n_synthetic_plots < MAX_PLOT_EVENTS)
        ):
            plot_written = plot_saved_synthetic_measurements(
                measurements=df,
                filter_colors=FILTER_COLORS,
                outfile=pngdir / f"lightcurve_synthetic_{i:04d}.png",
                event_index=i,
                filter_systems=FILTER_SYSTEMS,
                filter_by_system=FILTER_BY_SYSTEM,
            )
            if plot_written and APPLY_NUMERICAL_SUPPORT_CUT:
                plot_numerical_floor_diagnostic(
                    times=times,
                    raw_magnitudes_by_filter=raw_mag,
                    filters=FILTERS,
                    filter_colors=FILTER_COLORS,
                    apparent_cut_mag=event_numerical_support_cut_mag,
                    outfile=pngdir / f"lightcurve_numerical_floor_{i:04d}.png",
                )
            if plot_written:
                n_synthetic_plots += 1

        if SAVE_CSV and csv_selected and save_synthetic_csv:
            df.to_csv(outdir / f"lightcurve_{i:04d}.csv", index=False)
        if SAVE_PARQUET and save_synthetic_csv:
            SYNTHETIC_PARQUET_WRITER.add(i, df)


    parquet_manifest_rows = []
    if SAVE_PARQUET:
        parquet_manifest_rows.extend(SYNTHETIC_PARQUET_WRITER.close())
        parquet_manifest_rows.extend(LSSTLIKE_PARQUET_WRITER.close())
        manifest = pd.DataFrame(
            parquet_manifest_rows,
            columns=[
                "product", "file", "event_id_min", "event_id_max",
                "N_events", "N_rows",
            ],
        )
        manifest.to_csv(parquetdir / "parquet_manifest.csv", index=False)
        manifest.to_parquet(
            parquetdir / "parquet_manifest.parquet",
            engine="pyarrow",
            compression=PARQUET_COMPRESSION,
            index=False,
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    if SAVE_BOOM_SYNTHETIC:
        boom_summary_df = pd.DataFrame(boom_summary_rows)
        if boom_summary_df.empty:
            boom_summary_df = pd.DataFrame(columns=list(dict.fromkeys([
                *summary_df.columns, "n_export_rows",
                "empty_lightcurve", "boom_csv_saved",
            ])))
        boom_summary_df.to_csv(boomdir / "summary.csv", index=False)
        if (
            (BOOM_TARGET_NONEMPTY or BOOM_EXPORT_MODE == "complete")
            and len(boom_summary_df) != N_SAMPLES
        ):
            raise RuntimeError(
                "Internal BOOM population count mismatch: expected "
                f"{N_SAMPLES:,}, wrote {len(boom_summary_df):,}"
            )
    if SAVE_PARQUET:
        summary_df.to_parquet(
            run_dir / "summary.parquet",
            engine="pyarrow",
            compression=PARQUET_COMPRESSION,
            index=False,
        )

    elapsed_seconds = time.perf_counter() - generation_started
    n_injected = len(summary_df)
    n_boom_nonempty = len(boom_summary_rows)
    boom_nonempty_fraction = (
        n_boom_nonempty / n_injected
        if SAVE_BOOM_SYNTHETIC and n_injected else None
    )
    boom_visibility_efficiency = (
        boom_nonempty_fraction
        if BOOM_EXPORT_MODE == "m5-detected" else None
    )
    performance = {
        "elapsed_seconds": elapsed_seconds,
        "events_requested": N_SAMPLES,
        "requested_population_counting_unit": (
            "nonempty_boom_lightcurves"
            if BOOM_TARGET_NONEMPTY else "injected_physical_events"
        ),
        "events_injected": n_injected,
        "boom_nonempty_lightcurves": n_boom_nonempty if SAVE_BOOM_SYNTHETIC else None,
        "boom_nonempty_export_fraction": boom_nonempty_fraction,
        "boom_visibility_efficiency": boom_visibility_efficiency,
        "events_per_second": n_injected / elapsed_seconds if elapsed_seconds > 0 else None,
        "surrogate_batch_size_requested": SURROGATE_BATCH_SIZE,
        "surrogate_vectorized_batch_used": bool(SURROGATE_PREDICTOR.batch_supported),
        "surrogate_batch_fallback_reason": SURROGATE_PREDICTOR.fallback_reason or None,
        "surrogate_scalar_calls": SURROGATE_PREDICTOR.scalar_calls,
        "surrogate_batch_calls": SURROGATE_PREDICTOR.batch_calls,
        "opsim_cache_size": OPSIM_CACHE_SIZE,
        "opsim_cache_hits": OPSIM_VISIT_CACHE.hits,
        "opsim_cache_misses": OPSIM_VISIT_CACHE.misses,
    }
    run_info_sections["run"]["n_events_injected"] = n_injected
    if SAVE_BOOM_SYNTHETIC:
        run_info_sections["boom_synthetic_export"].update({
            "nonempty_lightcurves_written": n_boom_nonempty,
            "injected_physical_events": n_injected,
            "nonempty_export_fraction": boom_nonempty_fraction,
            "visibility_efficiency_after_m5": boom_visibility_efficiency,
        })
    (run_dir / "runtime_performance.json").write_text(
        json.dumps(performance, indent=2) + "\n"
    )
    run_info_sections["runtime_performance"] = performance
    (run_dir / "run_info.txt").write_text(
        format_run_info_sections(run_info_sections)
    )

    print("Saved:", run_dir / "summary.csv")
    if SAVE_BOOM_SYNTHETIC:
        (boomdir / "run_info.txt").write_text(format_run_info_sections(run_info_sections))
        boom_mode_readme = (
            "Complete mode: every finite point is exported; m5 enters the "
            "Ivezic mag_err calculation but is not a selection threshold.\n"
            if BOOM_EXPORT_MODE == "complete"
            else "Depth-limited mode: mag <= m5; mag_err is the Ivezic "
            "1-sigma uncertainty.\n"
        )
        boom_population_readme = (
            "The requested population size counts injected physical KNe; all "
            "events with finite model support are exported.\n"
            if BOOM_EXPORT_MODE == "complete"
            else "The requested BOOM population size counts non-empty light "
            "curves; extra injections replace empty events.\n"
            if BOOM_TARGET_NONEMPTY
            else "The requested population size counts injected physical KNe; "
            "BOOM contains only the m5-visible subset.\n"
        )
        (boomdir / "README.txt").write_text(
            "Synthetic LSST observations; no random noise added.\n"
            + boom_mode_readme
            +
            "Cadence is synthetic, not actual OpSim visits. See run_info.txt for settings.\n"
            "Timestamp staggering is an export convention, not a new model evaluation.\n"
            "The requested step is reduced locally when required by a shorter adjacent model interval.\n"
            "BOOM keeps one point per time_mjd; staggering preserves simultaneous bands.\n"
            "When plotting is enabled, ../png/lightcurve_boom_*.png shows exported points with errors.\n"
            + boom_population_readme
            + "Injected-event count and export statistics are recorded in run_info.txt and runtime_performance.json.\n"
            "The root summary and scientific CSV/Parquet retain the population for envelope analysis.\n"
            "Use build_color_envelopes.py on that scientific population to analyse upper limits.\n"
            "summary.csv counts refer to these CSVs; upstream_* retains original statistics.\n"
            "Use this folder as the BOOM input data directory.\n"
        )
        print("Saved synthetic BOOM input:", boomdir, f"({len(boom_summary_rows)} events)")
        if boom_summary_rows and BOOM_EXPORT_MODE == "m5-detected":
            print("BOOM detections:", int(boom_summary_df["n_detected"].sum()))
        if BOOM_EXPORT_MODE == "m5-detected":
            print(
                "BOOM m5 visibility:",
                f"{n_boom_nonempty:,}/{n_injected:,} injected events = "
                f"{100.0 * boom_visibility_efficiency:.2f}%",
            )
        else:
            print(
                "BOOM truth-level completeness:",
                f"{n_boom_nonempty:,}/{n_injected:,} injected events = "
                f"{100.0 * boom_nonempty_fraction:.2f}%",
            )
    if PAIRING_MODE != "none":
        print(
            "Paired intrinsic population:",
            INTRINSIC_CATALOG_PATH,
            f"sha256={INTRINSIC_CATALOG_HASH}",
        )
    if SAVE_PARQUET:
        print("Saved Parquet population:", parquetdir)
    print(
        "Performance:",
        f"{elapsed_seconds:.1f} s total, {performance['events_per_second']:.3f} events/s;",
        f"OpSim cache hits/misses={OPSIM_VISIT_CACHE.hits}/{OPSIM_VISIT_CACHE.misses};",
        "vectorized surrogate batch="
        f"{performance['surrogate_vectorized_batch_used']}",
    )
    if RUNTIME.plot_mode == "saved":
        print(
            "Saved diagnostic plots:",
            f"synthetic={n_synthetic_plots}, LSST-like={n_lsstlike_plots}, BOOM={n_boom_plots}",
            f"in {pngdir}",
        )

def main() -> None:
    """Collect terminal choices, then execute one or several populations."""
    if "--plot-boom-input" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description="Plot BOOM detections with magnitude errors")
        parser.add_argument("--plot-boom-input", type=Path, required=True)
        parser.add_argument("--max-events", type=int, default=0, help="Maximum plots (0 = all)")
        options = parser.parse_args()
        replot_boom_input(options.plot_boom_input, options.max_events)
        return
    if len(sys.argv) == 1 and prompt_choice(
        "Workflow", [("generate", "Generate a population"), ("plot", "Plot existing BOOM CSVs")], "generate"
    ) == "plot":
        directory = Path(prompt_text("BOOM input directory", "boom_input"))
        limit = int(prompt_text("Maximum plots (0 = all)", "20"))
        replot_boom_input(directory, limit)
        return
    runtime = configure_runtime()
    seeds = realization_seed_list(runtime)
    if runtime.n_realizations > 1:
        generate_realization_batch(runtime, seeds)
    else:
        runtime.seed = seeds[0]
        run_generation(runtime)


if __name__ == "__main__":
    main()
