"""Generate synthetic and Rubin/LSST-like kilonova light curves.

The script uses the FIESTA ``Bu2026_MLP`` surrogate of POSSIS to generate a
population of multi-band kilonova light curves.  Every mode uses an exported
Rubin OpSim sky context; LSST-like modes additionally apply the OpSim cadence
and the Rubin photometric-noise model.

Quick start
-----------
Place ``kne_pipeline_common.py`` next to this file and run either::

    python generate_kne_lightcurves.py

for the interactive terminal menu, or for example::

    python generate_kne_lightcurves.py \
        --run-name broad_population \
        --filter-system lsst+ps1 \
        --output-mode detected-pairs \
        --opsim-dir opsim_exports_all_survey \
        --n-samples 1000 \
        --seed 12345

The runtime environment must provide NumPy, pandas, Matplotlib, JAX, Astropy,
``rubin_sim``, and FIESTA with access to the ``Bu2026_MLP`` model. Every mode
requires an OpSim export directory produced by ``get_opsim_baseline.py``. Run
``python generate_kne_lightcurves.py --help``
for every non-interactive option.

Paired Galactic-latitude populations
------------------------------------
For controlled low/high-latitude comparisons, first create one intrinsic
catalogue and then reuse it in every other sky run.  The catalogue fixes the
ejecta parameters, inclination, distance and redshift for each ``event_id``.
For controlled fixed-distance comparisons, reuse mode can instead retain the
ejecta parameters and inclination while replacing only distance and redshift.
The separate sky seed controls the OpSim field, merger epoch and noise.  For
example, create the low-latitude member with::

    python generate_kne_lightcurves.py \
        --run-name pair_b10_low --n-samples 2000 \
        --galactic-latitude-mode low \
        --galactic-latitude-threshold-deg 10 \
        --pairing-mode create \
        --intrinsic-catalog paired_intrinsic_population.csv \
        --seed 12345 --sky-seed 10001

and create the matching high-latitude member with::

    python generate_kne_lightcurves.py \
        --run-name pair_b10_high --n-samples 2000 \
        --galactic-latitude-mode high \
        --galactic-latitude-threshold-deg 10 \
        --pairing-mode reuse \
        --intrinsic-catalog paired_intrinsic_population.csv \
        --seed 12345 --sky-seed 10002

All other physical, photometric and output options must be identical.  The
same intrinsic catalogue can be reused for further thresholds; only the
latitude selection, run name and sky seed should change.

Per-event workflow
------------------
1. Draw intrinsic ejecta parameters from broad uniform priors or from complete
   rows of a named FIESTA posterior.
2. Draw an isotropic viewing angle and either a fixed distance or a distance
   that is uniform in comoving volume under the Astropy Planck18 cosmology.
3. Evaluate ``Bu2026_MLP`` on its native observer-frame time grid.
4. When an OpSim context is required, draw one exported source position and a
   merger time ``t0`` within that field's survey interval.
5. For LSST-like products, interpolate the dense model to post-merger visit
   times, apply optional Milky-Way extinction, compute the expected Rubin SNR,
   draw Gaussian noise in relative flux, and classify detections.

Output modes
------------
``all-synthetic-lsstlike-detected``
    Save the synthetic prediction for every injected event and save its
    LSST-like file only when at least ``--min-detections`` visits pass the
    detection threshold.
``detected-pairs``
    Save an OpSim-context synthetic/LSST-like pair only when at least
    ``--min-detections`` visits pass the detection threshold.
``lsstlike-detected``
    Save only the LSST-like file for events meeting ``--min-detections``.
``synthetic-mjd``
    Save only dense MW-extincted synthetic predictions with an OpSim-derived
    sky context and an absolute MJD grid.

In every mode that saves detected LSST-like products, the retained table contains detections and
non-detections covered by the FIESTA truth grid, and stops 10 days after the
last detection. ``summary.csv`` still records every injected event.

Output files
------------
``parquet/synthetic_part_XXXXX.parquet``
    Optional long-format synthetic population shards. ``event_id`` is repeated
    on every row for efficient event and band filtering.
``parquet/lsstlike_part_XXXXX.parquet``
    Optional LSST-like population shards, written for the same scientifically
    eligible events as the legacy LSST-like CSV products.
``csv/lightcurve_XXXX.csv``
    Long-format table with one row per finite synthetic
    ``(time, photometric_system, band)`` measurement. Columns include
    ``objectid``, ``candidate_id``, ``t_days``, optional ``time_mjd``,
    ``photometric_system``, ``band`` and ``mag``. ``mag`` is always the
    MW-extincted magnitude; the intrinsic magnitude is not written.
``csv/lightcurve_LSSTlike_XXXX.csv``
    One row per post-merger OpSim visit, including visit time, band, m5, model
    truth, expected and observed SNR, magnitude uncertainty, detection flag,
    measured magnitude or upper limit, and a unique candidate identifier.
``summary.csv``
    Exactly one row per injected event with physical parameters, distance,
    redshift, viewing angle, sky/extinction metadata, and detection summary.
``run_info.txt``
    Human-readable configuration snapshot for reproducibility.
``runtime_performance.json``
    Actual surrogate batch/fallback status, OpSim cache statistics and overall
    generation throughput.

Important conventions
---------------------
* Magnitudes are AB magnitudes; distances are Mpc; times are days.
* FIESTA returns observer-frame times and already includes cosmological time
  dilation through the supplied redshift.
* Synthetic measurements and LSST-like visit measurements use distinct IDs.
* Synthetic CSVs store only MW-extincted magnitudes. LSST-like SNR and
  detection calculations use the same extincted truth.
* The native FIESTA grid is always used for LSST-like interpolation.  A positive
  ``--synthetic-cadence-days`` changes only the saved synthetic CSV.
* ``--synthetic-depth-cut m5-median`` optionally removes saved synthetic points
  fainter than the precomputed median OpSim ``fiveSigmaDepth`` of their LSST
  band. ``get_opsim_baseline.py`` writes these values once to
  ``opsim_m5_medians_by_band.csv``; this generator only reads that file.
* The late-time Bu2026 numerical floor is removed from every saved product one
  magnitude before the approximate absolute-magnitude floor by default.  The
  corresponding apparent cut is recomputed from each event's luminosity
  distance. Diagnostic plots keep the complete raw Bu2026 prediction: the
  excluded continuation is dotted and the cut itself is shown as a dashed
  horizontal line.
* ``--plot-mode saved`` writes diagnostic PNGs for the products actually saved
  by the selected output mode. ``--max-plot-events`` limits the number of
  plotted events (20 by default; 0 means all saved events).
* The generator uses JAX keys for broad-prior draws and per-event NumPy random
  generators for distance, viewing angle, OpSim field, merger time, and noise.
* ``--storage-format both`` writes the complete sharded Parquet population and
  a reproducible individual-CSV sample without recomputing light curves.
* OpSim field tables are cached with an LRU policy. Every event receives a copy,
  so caching cannot leak event-level mutations.
* ``--filter-system lsst+ps1`` evaluates both passband sets in one surrogate
  call for each physical event, so distance, ejecta parameters, inclination,
  merger epoch and OpSim field are exactly paired between systems.
* ``--filter-system ps1`` evaluates and saves only PS1 ``grizy`` synthetic
  light curves. Rubin OpSim supplies no PS1 visit depths, so PS1-only runs do
  not create LSST-like products and do not apply a Rubin ``m5`` cut.
* Requested surrogate batches use FIESTA's public ``vpredict`` method and are
  accepted only after comparison with scalar reference predictions; otherwise
  the run falls back automatically without changing the requested population.
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
import jax
import secrets

# --- FIESTA ---

from fiesta.inference.prior import Uniform
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.lightcurve_model import BullaFlux

# --- COSMOLOGY FOR DISTANCE SAMPLING ---
# Used only to draw events uniformly in comoving volume.
# Bulla/FIESTA still receives luminosity_distance and redshift as input parameters.
from astropy.cosmology import Planck18 as COSMO
from astropy.cosmology import z_at_value
from astropy.coordinates import SkyCoord
import astropy.units as u

# --- DATAFRAME TOOL ---

import pandas as pd

# --- PLOT AND OUTDIR ---

import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# --- RUBIN SIM (LSST photometric model) ---

from rubin_sim.phot_utils import (
    DustValues,
    PhotometricParameters,
    calc_mag_error_m5,
    calc_snr_m5,
    rubin_bandpasses,
)


# --- Plot helpers (only needed if overlay plots are enabled) ---
from matplotlib.lines import Line2D
# NOTE: Line2D is only required for plot_synthetic_and_lsstlike().

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



def get_rubin_dust_coefficients() -> dict[str, float]:
    """Return one Rubin extinction coefficient for each ``ugrizy`` band.

    ``DustValues.r_x`` is supplied by ``rubin_sim``.  The returned mapping is
    used in ``A_band = R_band * E(B-V)``.  Both short keys (``g``) and prefixed
    keys (``lsstg``) are accepted to remain compatible with rubin_sim releases;
    a missing band is treated as a configuration error rather than silently
    assigning an approximate coefficient.
    """
    dust = DustValues()
    raw = getattr(dust, "r_x", None)
    if raw is None:
        raise RuntimeError("DustValues() has no attribute r_x")
    out = {}
    for b in "ugrizy":
        if b in raw:
            out[b] = float(raw[b])
        elif f"lsst{b}" in raw:
            out[b] = float(raw[f"lsst{b}"])
        else:
            raise KeyError(f"Could not find Rubin dust coefficient for band {b!r} in DustValues.r_x keys={list(raw)}")
    return out


def ensure_mw_extinction_columns(visits: pd.DataFrame) -> pd.DataFrame:
    """Ensure visit table has ebv_mw, per-row A_mw, and per-band A_*_mw columns.

    Compact OpSim exports recover ``ebv_mw`` and per-band metadata from
    ``opsim_fields_index.csv`` before this helper is called. Legacy verbose
    visit files remain supported. Missing A values are derived from E(B-V)
    and Rubin coefficients when possible.
    """
    v = visits.copy()
    if "ebv_mw" not in v.columns:
        # Backward-compatible aliases, if a user manually added an E(B-V) column.
        for cand in ["ebv", "E_BV", "EBV", "sfd_ebv"]:
            if cand in v.columns:
                v["ebv_mw"] = pd.to_numeric(v[cand], errors="coerce")
                break
    if "ebv_mw" not in v.columns:
        v["ebv_mw"] = np.nan
    v["ebv_mw"] = pd.to_numeric(v["ebv_mw"], errors="coerce")

    for b in "ugrizy":
        if f"R_{b}_mw" not in v.columns:
            v[f"R_{b}_mw"] = float(DUST_R_X[b])
        if f"A_{b}_mw" not in v.columns:
            v[f"A_{b}_mw"] = v["ebv_mw"] * float(DUST_R_X[b])

    if "R_mw" not in v.columns:
        v["R_mw"] = v["band"].astype(str).map({b: float(DUST_R_X[b]) for b in "ugrizy"})
    if "A_mw" not in v.columns:
        v["A_mw"] = v["band"].astype(str).map({b: float(DUST_R_X[b]) for b in "ugrizy"}) * v["ebv_mw"]

    return v


def get_event_mw_extinction_summary(visits: pd.DataFrame) -> dict[str, object]:
    """Summarize E(B-V)_MW and Rubin A_band values for one selected field/event."""
    v = ensure_mw_extinction_columns(visits)
    ebv_vals = pd.to_numeric(v.get("ebv_mw", pd.Series(np.nan, index=v.index)), errors="coerce").to_numpy(float)
    ebv = float(np.nanmedian(ebv_vals)) if np.any(np.isfinite(ebv_vals)) else np.nan
    out = {"ebv_mw": ebv}
    for column in ("fieldRA", "fieldDec"):
        values = pd.to_numeric(v.get(column, pd.Series(np.nan, index=v.index)), errors="coerce").to_numpy(float)
        out[column] = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan
    target_values = v.get("target_name", pd.Series("", index=v.index)).dropna().astype(str)
    out["target_name"] = target_values.iloc[0] if len(target_values) else ""
    for b in "ugrizy":
        out[f"R_{b}_mw"] = float(DUST_R_X[b])
        col = f"A_{b}_mw"
        vals = pd.to_numeric(v.get(col, pd.Series(np.nan, index=v.index)), errors="coerce").to_numpy(float)
        if np.any(np.isfinite(vals)):
            out[col] = float(np.nanmedian(vals))
        else:
            out[col] = float(DUST_R_X[b] * ebv) if np.isfinite(ebv) else np.nan
    return out

# Standard Rubin throughput curves and detector/exposure parameters used by
# the m5-based photometric-noise model.  These are shared by all events.
BP = rubin_bandpasses()
PHOT = {band: PhotometricParameters(bandpass=band) for band in "ugrizy"}
DUST_R_X = get_rubin_dust_coefficients()
print("Rubin MW dust coefficients R_x =", DUST_R_X)


def plot_synthetic_only(times, mag_dict, filters, filter_colors, mag_lim, outfile):
    """Save a diagnostic plot of the dense synthetic FIESTA light curves."""
    t = np.asarray(times, float)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    for f in filters:
        y = np.asarray(mag_dict[f], float).copy()
        y[y > mag_lim] = np.nan
        ax.plot(t, y, linewidth=2.6, color=filter_colors.get(f, None), label=f)

    ax.axhline(mag_lim, linestyle="--", linewidth=1.1, alpha=0.7, color="gray",
               label=f"lim {mag_lim:g} mag")

    # FIESTA model times are strictly positive, so a logarithmic axis is valid.
    ax.set_xscale("log")
    ax.set_xlabel("Time [days] (observer-frame)")
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    finite_chunks = []
    for f in filters:
        yy = np.asarray(mag_dict[f], float).copy()
        yy[yy > mag_lim] = np.nan
        yy = yy[np.isfinite(yy)]
        if yy.size > 0:
            finite_chunks.append(yy)

    if len(finite_chunks) == 0:
        plt.close(fig)
        return

    y_all = np.concatenate(finite_chunks)
    ybright = np.nanmin(y_all)
    ax.set_ylim(mag_lim + 0.2, ybright - 0.3)
        
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=True, framealpha=0.95, title="Filters (synthetic)")

    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_saved_synthetic_measurements(
    measurements,
    filter_colors,
    outfile,
    event_index,
):
    """Plot the final synthetic table exactly as it is written to CSV.

    Unlike the legacy dense-model diagnostic above, this plot is built after
    Milky-Way extinction, the optional resampling, the magnitude cap, and the
    optional median-m5 cut. It therefore represents the saved synthetic
    product rather than the unprocessed FIESTA prediction.
    """
    required = {"t_days", "band", "mag"}
    missing = required.difference(measurements.columns)
    if missing:
        raise ValueError(
            f"Cannot plot synthetic event {event_index}: missing columns "
            f"{sorted(missing)}"
        )

    plot_table = measurements.copy()
    plot_table["t_days"] = pd.to_numeric(plot_table["t_days"], errors="coerce")
    plot_table["mag"] = pd.to_numeric(plot_table["mag"], errors="coerce")
    plot_table = plot_table.loc[
        np.isfinite(plot_table["t_days"])
        & (plot_table["t_days"] > 0)
        & np.isfinite(plot_table["mag"])
    ].copy()
    if plot_table.empty:
        return False

    if "photometric_system" not in plot_table:
        plot_table["photometric_system"] = "lsst"
    plot_table["photometric_system"] = (
        plot_table["photometric_system"].astype(str).str.lower()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    plotted = False
    for system in FILTER_SYSTEMS:
        for band, filter_name in FILTER_BY_SYSTEM[system].items():
            subset = plot_table.loc[
                (plot_table["photometric_system"] == system)
                & (plot_table["band"].astype(str) == band)
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
    ax.set_ylabel("AB magnitude (including Milky-Way extinction)")
    ax.invert_yaxis()
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(title="Photometric band", ncol=2, frameon=True, framealpha=0.95)
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return True


def plot_numerical_floor_diagnostic(
    times,
    raw_magnitudes_by_filter,
    filters,
    filter_colors,
    apparent_cut_mag,
    outfile,
):
    """Show the complete Bu2026 prediction and the distance-dependent cut.

    The solid part is numerically retained.  Starting at the first point with
    ``m >= apparent_cut_mag``, the raw model continuation is drawn with a dotted
    line even though it is excluded from every saved light curve and from the
    LSST-like calculation.  Magnitudes are shown before Milky-Way extinction so
    the same numerical-support cut applies to every band.
    """
    t = np.asarray(times, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    finite_for_limits = []

    for filter_name in filters:
        raw = np.asarray(raw_magnitudes_by_filter[filter_name], dtype=float)
        valid = np.isfinite(t) & (t > 0) & np.isfinite(raw)
        if not np.any(valid):
            continue
        finite_for_limits.append(raw[valid])
        indices = np.flatnonzero(valid)
        crossing = indices[raw[indices] >= float(apparent_cut_mag)]
        color = filter_colors.get(filter_name)
        band_label = filter_name.replace("lsst", "")

        if crossing.size == 0:
            ax.plot(t[valid], raw[valid], color=color, linewidth=2.1, label=band_label)
            continue

        first = int(crossing[0])
        retained = valid & (np.arange(len(t)) < first)
        if np.any(retained):
            ax.plot(
                t[retained], raw[retained], color=color, linewidth=2.1,
                label=band_label,
            )

        # Include the last retained sample in the dotted segment so the visual
        # transition remains connected across the support boundary.
        rejected = valid & (np.arange(len(t)) >= max(first - 1, 0))
        ax.plot(
            t[rejected], raw[rejected], color=color, linewidth=1.7,
            linestyle=":", alpha=0.9,
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
    ax.set_ylabel("AB magnitude (before Milky-Way extinction)")
    ax.invert_yaxis()
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    if finite_for_limits:
        all_values = np.concatenate(finite_for_limits)
        ax.set_ylim(np.nanmax(all_values) + 0.3, np.nanmin(all_values) - 0.3)
    ax.legend(title="LSST band", ncol=2, frameon=True, framealpha=0.95)
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)



def plot_synthetic_and_lsstlike(times, mag_dict, obs, filters, filter_colors, mag_lim, outfile):
    """Overlay dense model curves, detections, and per-visit upper limits."""
    t = np.asarray(times, float)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    # Dense synthetic truth curves.
    for f in filters:
        y = np.asarray(mag_dict[f], float)
        y = y.copy()
        y[y > mag_lim] = np.nan
        ax.plot(t, y, linewidth=2.6, color=filter_colors.get(f, None))

    # Keep only visits with model truth inside the surrogate time interval.
    obs2 = obs.copy()
    obs2 = obs2[np.isfinite(obs2["mag_true"])]
    obs2 = obs2[(obs2["t_days"] >= np.nanmin(t[t > 0])) & (obs2["t_days"] <= np.nanmax(t))]

    # A non-positive time cannot be displayed on a logarithmic axis.
    eps = 1e-3
    obs2["t_plot"] = np.where(obs2["t_days"].to_numpy() <= 0, eps, obs2["t_days"].to_numpy())

    # Plot detections and non-detections band by band.
    for b in sorted(obs2["band"].unique()):
        f = BAND_TO_FIESTA.get(b, None)
        if f is None or f not in filters:
            continue

        col = filter_colors.get(f, None)
        sub = obs2[obs2["band"] == b]

        det = sub[sub["detected"] & np.isfinite(sub["mag_obs"]) & np.isfinite(sub["mag_err"])]
        nd  = sub[(~sub["detected"]) & np.isfinite(sub["mag_ulim"])]

        if len(det) > 0:
            ax.errorbar(
                det["t_plot"], det["mag_obs"],
                yerr=det["mag_err"],
                fmt="o", linestyle="none", markersize=4,
                elinewidth=1.0, capsize=0,
                color=col,
            )

        if len(nd) > 0:
            ax.scatter(
                nd["t_plot"], nd["mag_ulim"],
                marker="v", s=28,
                facecolors="none", edgecolors=col,
                linewidths=1.2
            )

    # Display the configured faint-magnitude storage limit.
    ax.axhline(mag_lim, linestyle="--", linewidth=1.1, alpha=0.7, color="gray", label=f"lim {mag_lim:g} mag")

    # Plot axes and limits.
    ax.set_xscale("log")
    ax.set_xlabel("Time since trigger [days]")
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.set_xlim(np.nanmin(t[t > 0]), np.nanmax(t))

    # Determine magnitude limits from every visible product.
    y_all = []
    for f in filters:
        y = np.asarray(mag_dict[f], float)
        y = y.copy()
        y[y > mag_lim] = np.nan
        y_all.append(y)
    if len(obs2) > 0:
        y_all.append(obs2["mag_obs"].to_numpy())
        y_all.append(obs2["mag_ulim"].to_numpy())

    y_all = np.concatenate([yy[np.isfinite(yy)] for yy in y_all if np.any(np.isfinite(yy))])
    if len(y_all) > 0:
        ybright = np.nanmin(y_all)
        ax.set_ylim(mag_lim + 0.2, ybright - 0.3)

    # Build separate legend entries for truth, upper limits, and detections.
    handles, labels = [], []
    for f in filters:
        col = filter_colors.get(f, None)
        handles.append(Line2D([0],[0], color=col, lw=2.6))
        labels.append(f"{f} (synthetic)")
    for f in filters:
        col = filter_colors.get(f, None)
        handles.append(Line2D([0],[0], marker="v", color=col, lw=0, markersize=6,
                              markerfacecolor="none", markeredgewidth=1.2))
        labels.append(f"{f} (upper-limit)")
    for f in filters:
        col = filter_colors.get(f, None)
        handles.append(Line2D([0],[0], marker="o", color=col, lw=0, markersize=5))
        labels.append(f"{f} (obs)")

    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.95)


    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_lsstlike_only(
    observations,
    filters,
    filter_colors,
    event_index,
    outfile,
    tmax_days=None,
):
    """Plot LSST-like detections and upper limits without dense model curves.

    This function contains only diagnostic visualization.  Moving it outside
    the simulation loop keeps the scientific workflow readable and does not
    change any CSV, SNR, detection, or random-number calculation.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    # Logarithmic time axes cannot display t <= 0.  Also discard visits for
    # which the FIESTA model has no finite truth magnitude.
    obs_plot = observations[
        np.isfinite(observations["mag_true"])
        & (observations["t_days"] > 0)
    ].copy()
    if tmax_days is not None:
        obs_plot = obs_plot[obs_plot["t_days"] <= tmax_days]

    if obs_plot.empty:
        plt.close(fig)
        return

    for band in sorted(obs_plot["band"].unique()):
        filter_name = BAND_TO_FIESTA.get(band)
        if filter_name is None or filter_name not in filters:
            continue

        subset = obs_plot[obs_plot["band"] == band]
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

    # Derive display limits from all finite detections and upper limits.
    visible_magnitudes = []
    for column in ("mag_obs", "mag_ulim"):
        values = obs_plot[column].to_numpy(float)
        if np.any(np.isfinite(values)):
            visible_magnitudes.append(values[np.isfinite(values)])
    if visible_magnitudes:
        y_all = np.concatenate(visible_magnitudes)
        ax.set_ylim(np.max(y_all) + 0.2, np.min(y_all) - 0.3)

    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    # Give detections and upper limits separate legend entries in each band.
    legend_handles = []
    legend_labels = []
    for band in sorted(obs_plot["band"].unique()):
        filter_name = BAND_TO_FIESTA.get(band)
        if filter_name is None or filter_name not in filters:
            continue

        subset = obs_plot[obs_plot["band"] == band]
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
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=5,
                )
            )
            legend_labels.append(filter_name)
        if has_upper_limit:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="v",
                    linestyle="none",
                    color=color,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markersize=6,
                )
            )
            legend_labels.append(f"ul_{filter_name}")

    if legend_labels:
        ax.legend(
            legend_handles,
            legend_labels,
            title="Filters",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            framealpha=0.95,
        )

    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)



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
        accepted. MW-extinction metadata must already have been restored from
        ``opsim_fields_index.csv`` when absent from the visit table.
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
        and adds ``t_days``, intrinsic/extincted truth, expected SNR,
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
    v = ensure_mw_extinction_columns(v)

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

    # Always keep the raw model magnitude. In compact-extincted mode, also
    # save the Milky-Way-extincted truth and use it for LSST realism:
    #   m_ext = m_no_mw + A_band.
    v["mag_true_no_mw"] = mag_true
    A_mw = pd.to_numeric(v.get("A_mw", np.nan), errors="coerce").to_numpy(float)
    mag_true_mw = mag_true + A_mw
    v["mw_extinction_applied_to_snr"] = bool(APPLY_MW_EXTINCTION_TO_LSST_REALISM)

    if APPLY_MW_EXTINCTION_TO_LSST_REALISM:
        v["mag_true_mw_extincted"] = mag_true_mw
        v["mag_true"] = mag_true_mw
    else:
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
    """Keep valid KNe visits through a fixed time after the last detection.

    Rows without finite ``mag_true`` are outside the supported FIESTA truth
    interval (or beyond the configured truth cap), so they are not meaningful
    KNe upper limits and are removed before locating the last detection. The
    anchor is determined from the complete valid visit sequence before any
    temporal trimming, so this operation can never remove a detection. If no
    valid visit is detected, all valid visits are returned but no LSST-like
    file will pass the positive ``min_detections`` selection.
    """
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


def apply_synthetic_m5_median_cut(
    measurements: pd.DataFrame,
    objectid: str,
    m5_medians: dict[str, float],
) -> pd.DataFrame:
    """Keep only synthetic points brighter than their band's median OpSim m5.

    The saved ``mag`` column is compared with m5. In ``compact-extincted``
    mode, this column has already been replaced by the MW-extincted magnitude.
    Identifiers are rebuilt after filtering so the first retained row carries
    ``objectid`` and every retained point has one contiguous, unique
    ``candidate_id``.
    """
    out = measurements.copy()
    comparison_magnitude = pd.to_numeric(
        out["mag"], errors="coerce"
    ).to_numpy(float)
    thresholds = pd.to_numeric(
        out["band"].map(m5_medians), errors="coerce"
    ).to_numpy(float)
    keep = (
        np.isfinite(comparison_magnitude)
        & np.isfinite(thresholds)
        & (comparison_magnitude <= thresholds)
    )
    out = out.loc[keep].reset_index(drop=True)

    out["objectid"] = ""
    if len(out):
        out.at[0, "objectid"] = str(objectid)
    out["candidate_id"] = [
        f"{objectid}_{measurement_index}"
        for measurement_index in range(1, len(out) + 1)
    ]
    return out


def keep_only_mw_extincted_synthetic_magnitude(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Store only the MW-extincted synthetic magnitude under ``mag``.

    ``build_synthetic_measurement_table`` temporarily creates both ``mag``
    (intrinsic) and ``mag_mw_extincted``. This helper replaces ``mag`` with the
    extincted value and removes the extra column, matching the VM convention
    requested for this generator.
    """
    if "mag_mw_extincted" not in measurements.columns:
        raise ValueError(
            "MW-extincted synthetic output requires finite A_band values from "
            "the selected OpSim field"
        )
    out = measurements.copy()
    out["mag"] = pd.to_numeric(out["mag_mw_extincted"], errors="coerce")
    return out.drop(columns=["mag_mw_extincted"])


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
) -> tuple[float, float]:
    """Return one Planck18-consistent ``(z, D_L[Mpc])`` pair.

    Population runs draw independently and uniformly in comoving volume.
    Fixed-distance runs return the single configured diagnostic distance.
    """
    mode = str(DISTANCE_SAMPLING).lower().strip()
    if mode == "fixed":
        return float(FIXED_REDSHIFT), float(FIXED_LUMINOSITY_DISTANCE_MPC)
    if mode == "comoving_volume":
        if sampler is None:
            raise RuntimeError("Comoving-volume sampler was not initialized")
        return sampler.sample(rng)
    raise ValueError("DISTANCE_SAMPLING must be 'fixed' or 'comoving_volume'")


def sample_inclination_for_event(rng: np.random.Generator) -> float:
    """Return inclination_EM [rad].

    For isotropic orientations over [0, pi/2], draw mu=cos(i) uniformly
    in [0, 1] and return i=arccos(mu). This gives p(i) proportional to sin(i).
    """
    mode = str(INCLINATION_SAMPLING).lower().strip()
    i_min = float(INCLINATION_MIN_RAD)
    i_max = float(INCLINATION_MAX_RAD)
    if not (0.0 <= i_min < i_max <= np.pi):
        raise ValueError("Require 0 <= INCLINATION_MIN_RAD < INCLINATION_MAX_RAD <= pi")

    if mode == "uniform_angle":
        return float(rng.uniform(i_min, i_max))

    if mode == "uniform_cos":
        mu_lo = float(np.cos(i_max))
        mu_hi = float(np.cos(i_min))
        mu = float(rng.uniform(mu_lo, mu_hi))
        mu = float(np.clip(mu, -1.0, 1.0))
        return float(np.arccos(mu))

    raise ValueError("INCLINATION_SAMPLING must be 'uniform_cos' or 'uniform_angle'")


# --- PHOTOMETRIC FILTER SYSTEMS ---

# Exact filter identifiers understood by FIESTA/sncosmo.
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

# Schlafly & Finkbeiner (2011), R_V=3.1 coefficients commonly used for PS1.
# Rubin coefficients continue to come directly from rubin_sim.DustValues, so
# each photometric system is extinguished in its own passbands.
PS1_DUST_R_X = {
    "g": 3.172,
    "r": 2.271,
    "i": 1.682,
    "z": 1.322,
    "y": 1.087,
}

# Optical R_band = A_band/E(B-V) values for a standard R_V=3.1 Milky-Way
# extinction law.  These are used only to keep the saved ZTF synthetic
# magnitudes and the reconstructed no-MW magnitudes internally consistent.
# As for every broad-band coefficient, the exact value has a weak dependence
# on the assumed source spectrum and extinction law; the adopted constants are
# therefore recorded in run_info.txt and summary.csv.
ZTF_DUST_R_X = {
    "g": 3.303,
    "r": 2.285,
    "i": 1.698,
}

DUST_R_X_BY_SYSTEM = {
    "lsst": DUST_R_X,
    "ps1": PS1_DUST_R_X,
    "ztf": ZTF_DUST_R_X,
}


def add_multisystem_extinction_summary(
    event_summary: dict[str, object],
) -> dict[str, object]:
    """Add unambiguous per-system ``R`` and ``A`` values for one event.

    The OpSim field index supplies ``E(B-V)`` and legacy Rubin ``A_g_mw``-like
    columns.  Paired products additionally store ``A_lsst_g_mw`` and
    ``A_ps1_g_mw`` or ``A_ztf_g_mw`` in ``summary.csv``.  The long light-curve table therefore
    needs only ``photometric_system``, ``band`` and the already extinguished
    magnitude; extinction metadata is not repeated at every epoch.
    """
    out = dict(event_summary)
    ebv = float(out.get("ebv_mw", np.nan))
    for system in FILTER_SYSTEMS:
        for band, coefficient in DUST_R_X_BY_SYSTEM[system].items():
            coefficient = float(coefficient)
            out[f"R_{system}_{band}_mw"] = coefficient
            out[f"A_{system}_{band}_mw"] = (
                coefficient * ebv if np.isfinite(ebv) else np.nan
            )
    return out

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
    """Derive a reproducible sky seed that differs by latitude selection.

    The historical all/low/high formula is preserved exactly.  Latitude-bin
    runs additionally encode both interval edges, so every disjoint bin gets
    an independent but reproducible sky assignment.
    """
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


def build_runtime_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without parsing or prompting.

    Running the script with no arguments starts the interactive menu.  Passing
    one or more arguments runs non-interactively unless ``--interactive`` is
    also supplied.  ``--help`` therefore provides a complete reproducible
    alternative to the terminal menu.
    """
    parser = argparse.ArgumentParser(
        description="Generate broad-uniform or posterior-based KNe light curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-name",
        default="broad_run",
        help="Readable label used in the output-directory name",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Parent directory in which the new run directory is created",
    )
    parser.add_argument(
        "--filter-system",
        choices=["lsst", "ps1", "ztf", "lsst+ps1", "lsst+ztf"],
        default="lsst+ps1",
        help=(
            "Generate one survey system, or evaluate LSST together with PS1 "
            "or ZTF for the same physical event, distance, merger epoch and "
            "OpSim field. Standalone PS1/ZTF modes are synthetic-only; ZTF "
            "depth cuts are applied later by build_color_envelopes.py."
        ),
    )
    parser.add_argument(
        "--parameter-source",
        choices=["broad", "posterior"],
        default="broad",
        help="Use broad uniform priors or complete rows from a FIESTA posterior",
    )
    parser.add_argument(
        "--posterior-file",
        type=Path,
        help="posterior.npz or a tar archive containing posterior.npz",
    )
    parser.add_argument(
        "--posterior-inclination",
        choices=["posterior", "isotropic"],
        default="posterior",
        help="Keep the joint posterior inclination or replace it by an isotropic draw",
    )
    parser.add_argument(
        "--opsim-dir",
        type=Path,
        default=Path("opsim_exports_all_survey"),
        help=(
            "Directory containing opsim_visits_field*.csv and the m5 summary "
            "written by get_opsim_baseline.py"
        ),
    )
    parser.add_argument(
        "--galactic-latitude-mode",
        choices=["all", "high", "low", "bin"],
        default="all",
        help=(
            "Restrict OpSim field selection: all fields, |b| greater than or "
            "equal to the threshold, |b| below the threshold, or a disjoint "
            "interval min <= |b| < max"
        ),
    )
    parser.add_argument(
        "--galactic-latitude-threshold-deg",
        type=float,
        default=20.0,
        help="Absolute Galactic-latitude boundary used by high/low modes",
    )
    parser.add_argument(
        "--galactic-latitude-min-deg",
        type=float,
        default=0.0,
        help="Inclusive lower |b| edge used by latitude-bin mode",
    )
    parser.add_argument(
        "--galactic-latitude-max-deg",
        type=float,
        default=5.0,
        help="Exclusive upper |b| edge used by latitude-bin mode (90 is inclusive)",
    )
    parser.add_argument(
        "--output-mode",
        choices=[
            "all-synthetic-lsstlike-detected",
            "detected-pairs",
            "lsstlike-detected",
            "synthetic-mjd",
        ],
        default="detected-pairs",
        help=(
            "all-synthetic-lsstlike-detected: all synthetic + LSST-like for detected events; "
            "detected-pairs: synthetic + LSST-like for detected events; "
            "lsstlike-detected: LSST-like only for detected events; "
            "synthetic-mjd: synthetic only with OpSim context and absolute MJD"
        ),
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=1,
        help="Minimum detected=True points required to save an LSST-like file",
    )
    parser.add_argument(
        "--synthetic-cadence-days",
        type=float,
        default=0.0,
        help=(
            "Cadence of saved synthetic CSVs in observer-frame days; "
            "0 preserves the native FIESTA grid"
        ),
    )
    parser.add_argument(
        "--synthetic-depth-cut",
        choices=["none", "m5-median"],
        default="none",
        help=(
            "Optional band-dependent cut on saved synthetic points using the "
            "precomputed median fiveSigmaDepth from get_opsim_baseline.py"
        ),
    )
    parser.add_argument(
        "--storage-format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help=(
            "Persist eligible per-event products as legacy CSV files, as "
            "sharded Parquet datasets, or in both forms"
        ),
    )
    parser.add_argument(
        "--parquet-events-per-shard",
        type=int,
        default=500,
        help="Number of events buffered in each Parquet shard",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=["zstd", "snappy", "gzip", "none"],
        default="zstd",
        help="Compression codec used for Parquet shards",
    )
    parser.add_argument(
        "--csv-sample-size",
        type=int,
        default=0,
        help=(
            "Number of injected event IDs selected reproducibly for legacy "
            "per-event CSV output; 0 allows every eligible event"
        ),
    )
    parser.add_argument(
        "--opsim-cache-size",
        type=int,
        default=32,
        help="Maximum number of fully prepared OpSim field tables kept in memory",
    )
    parser.add_argument(
        "--surrogate-batch-size",
        type=int,
        default=64,
        help=(
            "Requested Bu2026 prediction batch size. The first batch is "
            "checked against scalar predictions; unsupported or inconsistent "
            "batch APIs fall back automatically to scalar evaluation"
        ),
    )
    parser.add_argument(
        "--plot-mode",
        choices=["none", "saved"],
        default="none",
        help=(
            "Write PNG diagnostics for the light-curve products saved by the "
            "selected output mode"
        ),
    )
    parser.add_argument(
        "--max-plot-events",
        type=int,
        default=20,
        help="Maximum saved events plotted per product; 0 plots every saved event",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=80000,
        help="Number of independent kilonova events to inject",
    )
    parser.add_argument(
        "--pairing-mode",
        choices=["none", "create", "reuse"],
        default="none",
        help=(
            "none: draw an ordinary independent population; create: draw and "
            "save a reusable intrinsic-population catalogue; reuse: load an "
            "existing catalogue so event_id identifies the same physical KNe "
            "in several sky/latitude runs"
        ),
    )
    parser.add_argument(
        "--intrinsic-catalog",
        type=Path,
        help=(
            "CSV catalogue created or reused by paired runs. Required when "
            "--pairing-mode is create or reuse"
        ),
    )
    parser.add_argument(
        "--catalog-distance-policy",
        choices=["stored", "current-fixed"],
        default="stored",
        help=(
            "When reusing an intrinsic catalogue, either keep its stored "
            "distance/redshift or replace them by the current fixed-distance "
            "configuration while retaining all ejecta parameters and inclination"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Intrinsic-population seed. Use the same value for every member "
            "of a paired family; random when omitted"
        ),
    )
    parser.add_argument(
        "--sky-seed",
        type=int,
        help=(
            "Independent seed for OpSim field, merger epoch and photometric "
            "noise. Use different values for low/high runs. Defaults to --seed"
        ),
    )
    parser.add_argument(
        "--distance-mode",
        choices=["comoving-volume", "fixed"],
        default="comoving-volume",
        help="Draw distance uniformly in comoving volume or use one fixed luminosity distance",
    )
    parser.add_argument(
        "--fixed-distance-mpc",
        type=float,
        default=10.0,
        help="Fixed luminosity distance used when --distance-mode fixed",
    )
    parser.add_argument(
        "--distance-min-mpc",
        type=float,
        default=10.0,
        help="Minimum luminosity distance for comoving-volume draws",
    )
    parser.add_argument(
        "--distance-max-mpc",
        type=float,
        default=600.0,
        help="Maximum luminosity distance for comoving-volume draws",
    )
    parser.add_argument(
        "--numerical-floor-absolute-mag",
        type=float,
        default=0.0,
        help="Approximate absolute-magnitude floor of Bu2026",
    )
    parser.add_argument(
        "--numerical-floor-margin-mag",
        type=float,
        default=1.0,
        help="Safety margin applied before the numerical floor",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show the terminal menu even when other arguments are supplied",
    )
    return parser


def configure_runtime() -> argparse.Namespace:
    """Parse options, optionally prompt the user, and validate combinations."""
    args = build_runtime_parser().parse_args()
    # This VM-oriented generator stores only MW-extincted synthetic magnitudes.
    args.synthetic_schema = "compact-extincted"
    interactive = bool(args.interactive or len(sys.argv) == 1)
    if interactive:
        args.run_name = prompt_text("Readable run name", args.run_name)
        args.parameter_source = prompt_choice(
            "Physical-parameter population",
            [("broad", "Broad uniform parameter ranges"), ("posterior", "FIESTA posterior (.npz or AT2017gfo.tar)")],
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
        args.n_samples = int(prompt_text("Number of simulated events", str(args.n_samples)))
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
                [
                    (
                        "current-fixed",
                        "Use the fixed distance entered for this run",
                    ),
                    (
                        "stored",
                        "Keep the distance and redshift stored in the catalogue",
                    ),
                ],
                args.catalog_distance_policy,
            )
        seed_text = prompt_text(
            (
                "Intrinsic-population seed (integer)"
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
                    (
                        "Sky/noise seed (same = paired sky; different = new sky)"
                    ),
                    str(
                        args.sky_seed
                        if args.sky_seed is not None
                        else default_sky_seed
                    ),
                )
            )
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
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
        if args.pairing_mode != "reuse":
            raise ValueError(
                "--catalog-distance-policy current-fixed requires "
                "--pairing-mode reuse"
            )
        if args.distance_mode != "fixed":
            raise ValueError(
                "--catalog-distance-policy current-fixed requires "
                "--distance-mode fixed"
            )
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


RUNTIME = configure_runtime()

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

# Milky Way foreground extinction handling.
# get_opsim_baseline.py stores ebv_mw and A_band once per field in the field index.
# compact-extincted is a coherent physical mode: dust is applied to the truth
# magnitude before LSST-like SNR, noise and detection are calculated.
APPLY_MW_EXTINCTION_TO_LSST_REALISM = SYNTHETIC_SCHEMA == "compact-extincted"
# The currently available foreground-extinction coefficients below are Rubin
# coefficients. Do not apply them to PS1 filters. In compact mode, A_band is
# stored once per event in summary.csv and no redundant extincted columns are
# written. For extinction-corrected observations, compare to intrinsic columns.
SAVE_MW_EXTINCTED_SYNTHETIC_COLUMNS = (
    SYNTHETIC_SCHEMA == "compact-extincted"
)

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


# Instantiate the selected FIESTA surrogate once and reuse it for every event.
model = BullaFlux(name="Bu2026_MLP", filters=FILTERS)
print(f"Using model: {model.name}")


# Broad intrinsic population.  Distance, redshift, and inclination are sampled
# separately so their distributions can be controlled explicitly.
KN_prior = [
    # inclination_EM is sampled separately below so it can be uniform in cos(i).
    Uniform(xmin=-3.0, xmax=-1.3, naming=["log10_mej_dyn"]),
    Uniform(xmin=0.12, xmax=0.28, naming=["v_ej_dyn"]),
    Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
    Uniform(xmin=-2.0, xmax=-0.9, naming=["log10_mej_wind"]),
    Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"]),
    Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"]),
]
prior = ConstrainedPrior(KN_prior)


# Fixed intrinsic parameters. Distance/redshift are sampled separately below.

fixed_params = {}

print(f"Generating {N_SAMPLES} lightcurves; physical parameter source={PARAMETER_SOURCE}...")
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


INTRINSIC_PARAMETER_COLUMNS = [
    "log10_mej_dyn",
    "v_ej_dyn",
    "Ye_dyn",
    "log10_mej_wind",
    "v_ej_wind",
    "Ye_wind",
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
    global key
    posterior_index = np.nan
    if PARAMETER_SOURCE == "posterior":
        if POSTERIOR_ROWS is None or POSTERIOR_INDICES is None:
            raise RuntimeError(
                "Posterior rows are unavailable while drawing a new catalogue"
            )
        posterior_index = int(POSTERIOR_INDICES[event_index])
        params = dict(POSTERIOR_ROWS[event_index])
        if POSTERIOR_INCLINATION == "isotropic":
            params.pop("inclination_EM", None)
    elif VARY_ONE_PARAM:
        params = dict(base_params)
        params[VARY_PARAM] = float(vary_values[event_index])
    else:
        key, subkey = jax.random.split(key)
        sample = prior.sample(subkey, 1)
        params = {name: float(sample[name][0]) for name in prior.naming}
    params.update(fixed_params)
    if "inclination_EM" not in params:
        params["inclination_EM"] = sample_inclination_for_event(physical_rng)
    z, d_l = sample_distance_redshift_for_event(
        rng=physical_rng,
        sampler=COMOVING_VOLUME_SAMPLER,
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
            params["luminosity_distance"] = float(
                FIXED_LUMINOSITY_DISTANCE_MPC
            )
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
        print(
            "Catalogue ejecta parameters and inclinations retained; "
            f"distance/redshift overridden with D_L={FIXED_LUMINOSITY_DISTANCE_MPC:g} "
            f"Mpc and z={FIXED_REDSHIFT:.8g}"
        )


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
    "synthetic_magnitude_saved_in_mag": (
        "mw_extincted" if SAVE_MW_EXTINCTED_SYNTHETIC_COLUMNS else "intrinsic"
    ),
    "synthetic_depth_comparison_column": (
        "mag" if SYNTHETIC_DEPTH_CUT == "m5-median" else None
    ),
    "bu2026_numerical_floor_absolute_mag": NUMERICAL_FLOOR_ABSOLUTE_MAG,
    "numerical_floor_safety_margin_mag": NUMERICAL_FLOOR_MARGIN_MAG,
    "faintest_supported_absolute_mag": NUMERICAL_SUPPORT_ABSOLUTE_MAG,
    "apparent_cut_formula": "M_floor - margin + 5*log10(D_L/Mpc) + 25",
    "numerical_floor_cut_applied_before_mw_and_m5": True,
    "plot_mode": RUNTIME.plot_mode,
    "max_plot_events_per_product": MAX_PLOT_EVENTS,
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
    },
    "model": {
        "surrogate": model.name,
        "filter_system": FILTER_SYSTEM,
        "photometric_systems_saved": FILTER_SYSTEMS,
        "filters": FILTERS,
    },
    "population": population_info,
    "distance_and_orientation": distance_info,
    "output": output_info,
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
        "mw_extinction_applied": APPLY_MW_EXTINCTION_TO_LSST_REALISM,
        "window_anchor": "last_detection",
        "days_after_last_detection": LSSTLIKE_DAYS_AFTER_LAST_DETECTION,
        "rows_outside_finite_fiesta_truth_removed": True,
    }

if SAVE_MW_EXTINCTED_SYNTHETIC_COLUMNS:
    run_info_sections["foreground_extinction"] = {
        "relation": "A_band = R_band * E(B-V)",
        "rubin_coefficients": DUST_R_X,
        "ps1_coefficients_schlafly_finkbeiner_2011": (
            PS1_DUST_R_X if "ps1" in FILTER_SYSTEMS else None
        ),
        "ztf_coefficients_rv3p1": (
            ZTF_DUST_R_X if "ztf" in FILTER_SYSTEMS else None
        ),
        "summary_column_convention": "A_<system>_<band>_mw",
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
# MAIN LOOP: generate N_SAMPLES events
# =============================================================================

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
        table = ensure_mw_extinction_columns(table)
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
    Passing it to scalar ``surrogate.predict`` makes the seven-dimensional
    physical input collide with the batch dimension (the former implementation
    produced errors such as ``(1, 448)`` versus ``(7,)``).  ``vpredict`` applies
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
                    "Validated vectorized surrogate prediction against scalar "
                    f"references; batch size={len(contexts)}"
                )
            return candidates
        except Exception as exc:
            self.batch_supported = False
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            print(
                "Surrogate batch mode unavailable or inconsistent; using the "
                f"validated scalar path. Reason: {self.fallback_reason}"
            )
            return [self._scalar(context["params"]) for context in contexts]


def iter_event_predictions():
    """Yield prepared event contexts and raw predictions in bounded batches."""
    for start in range(0, N_SAMPLES, SURROGATE_BATCH_SIZE):
        stop = min(start + SURROGATE_BATCH_SIZE, N_SAMPLES)
        contexts = [
            prepare_event_parameter_context(event_index)
            for event_index in range(start, stop)
        ]
        predictions = SURROGATE_PREDICTOR.predict_contexts(contexts)
        for context, prediction in zip(contexts, predictions):
            yield context, prediction


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
    event_mw_extinction,
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
        "log10_mej_dyn": float(params["log10_mej_dyn"]),
        "v_ej_dyn": float(params["v_ej_dyn"]),
        "Ye_dyn": float(params["Ye_dyn"]),
        "log10_mej_wind": float(params["log10_mej_wind"]),
        "v_ej_wind": float(params["v_ej_wind"]),
        "Ye_wind": float(params["Ye_wind"]),
        "ebv_mw": float(event_mw_extinction.get("ebv_mw", np.nan)),
        "fieldRA": float(event_mw_extinction.get("fieldRA", np.nan)),
        "fieldDec": float(event_mw_extinction.get("fieldDec", np.nan)),
        "galactic_longitude_deg": float(
            event_mw_extinction.get("galactic_longitude_deg", np.nan)
        ),
        "galactic_latitude_deg": float(
            event_mw_extinction.get("galactic_latitude_deg", np.nan)
        ),
        "abs_galactic_latitude_deg": float(
            event_mw_extinction.get("abs_galactic_latitude_deg", np.nan)
        ),
        "target_name": str(event_mw_extinction.get("target_name", "")),
        **{
            f"R_{system}_{band}_mw": float(
                event_mw_extinction.get(f"R_{system}_{band}_mw", np.nan)
            )
            for system in FILTER_SYSTEMS
            for band in FILTER_BY_SYSTEM[system]
        },
        **{
            f"A_{system}_{band}_mw": float(
                event_mw_extinction.get(f"A_{system}_{band}_mw", np.nan)
            )
            for system in FILTER_SYSTEMS
            for band in FILTER_BY_SYSTEM[system]
        },
    }


summaries = []
n_synthetic_plots = 0
n_lsstlike_plots = 0

generation_started = time.perf_counter()

for event_context, prediction in iter_event_predictions():
    i = int(event_context["event_index"])
    params = event_context["params"]
    posterior_index = event_context["posterior_index"]
    rng_i = event_context["rng"]
    times, raw_mag = prediction
    d_l = float(params["luminosity_distance"])
    event_numerical_support_cut_mag = numerical_support_cut_apparent_mag(d_l)
    mag = mask_bands_after_numerical_floor(
        raw_mag,
        apparent_cut_mag=event_numerical_support_cut_mag,
    )

    # Per-event output/context defaults. In modes without LSST realism,
    # detection fields remain unavailable (NaN), not false detections.
    field_idx = np.nan
    t0 = np.nan
    visits = None
    event_mw_extinction = {
        "ebv_mw": np.nan,
        "fieldRA": np.nan,
        "fieldDec": np.nan,
        "target_name": "",
    }
    for _band in "ugrizy":
        event_mw_extinction[f"R_{_band}_mw"] = float(DUST_R_X[_band])
        event_mw_extinction[f"A_{_band}_mw"] = np.nan
    event_mw_extinction = add_multisystem_extinction_summary(
        event_mw_extinction
    )

    detection_evaluated = False
    n_det = 0
    t_first = np.nan
    t_last = np.nan
    t_first_nondet = np.nan
    lsstlike_window_anchor = np.nan
    lsstlike_window_end = np.nan
    lsstlike_window_anchor_found = False
    csv_selected = i in CSV_EVENT_INDICES
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
        event_mw_extinction = add_multisystem_extinction_summary(
            get_event_mw_extinction_summary(visits)
        )
        ra_event = float(event_mw_extinction.get("fieldRA", np.nan))
        dec_event = float(event_mw_extinction.get("fieldDec", np.nan))
        if np.isfinite(ra_event) and np.isfinite(dec_event):
            gal_event = SkyCoord(
                ra=ra_event * u.deg, dec=dec_event * u.deg, frame="icrs"
            ).galactic
            event_mw_extinction["galactic_longitude_deg"] = float(gal_event.l.deg)
            event_mw_extinction["galactic_latitude_deg"] = float(gal_event.b.deg)
            event_mw_extinction["abs_galactic_latitude_deg"] = abs(float(gal_event.b.deg))
        if SYNTHETIC_SCHEMA == "compact-extincted":
            missing_extinction = [
                f"{system}:{band}"
                for system in FILTER_SYSTEMS
                for band in FILTER_BY_SYSTEM[system]
                if not np.isfinite(
                    event_mw_extinction.get(
                        f"A_{system}_{band}_mw", np.nan
                    )
                )
            ]
            if missing_extinction:
                raise ValueError(
                    "compact-extincted requires finite Milky-Way extinction "
                    f"for every selected band; missing A_band for {missing_extinction} "
                    f"for {selected_opsim_file}. Use an OpSim export whose "
                    "opsim_fields_index.csv contains ebv_mw/A_band columns."
                )

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
            ra = float(event_mw_extinction.get("fieldRA", np.nan))
            dec = float(event_mw_extinction.get("fieldDec", np.nan))
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
                event_mw_extinction=event_mw_extinction,
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
                event_mw_extinction=event_mw_extinction,
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
        mw_extinction_by_band = None
        if SAVE_MW_EXTINCTED_SYNTHETIC_COLUMNS:
            mw_extinction_by_band = {
                band: float(
                    event_mw_extinction.get(
                        f"A_{photometric_system}_{band}_mw", np.nan
                    )
                )
                for band in system_filter_by_band
            }
        system_table = build_synthetic_measurement_table(
            objectid=objectid,
            times_days=t,
            magnitudes_by_filter=synthetic_mag,
            filters=system_filters,
            filter_by_band=system_filter_by_band,
            time_mjd=time_mjd,
            mw_extinction_by_band=mw_extinction_by_band,
            magnitude_limit=event_numerical_support_cut_mag,
        )
        if SAVE_MW_EXTINCTED_SYNTHETIC_COLUMNS:
            system_table = keep_only_mw_extincted_synthetic_magnitude(
                system_table
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
        )
        if plot_written:
            plot_numerical_floor_diagnostic(
                times=times,
                raw_magnitudes_by_filter=raw_mag,
                filters=FILTERS,
                filter_colors=FILTER_COLORS,
                apparent_cut_mag=event_numerical_support_cut_mag,
                outfile=pngdir / f"lightcurve_numerical_floor_{i:04d}.png",
            )
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
if SAVE_PARQUET:
    summary_df.to_parquet(
        run_dir / "summary.parquet",
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
        index=False,
    )

elapsed_seconds = time.perf_counter() - generation_started
performance = {
    "elapsed_seconds": elapsed_seconds,
    "events_requested": N_SAMPLES,
    "events_per_second": N_SAMPLES / elapsed_seconds if elapsed_seconds > 0 else None,
    "surrogate_batch_size_requested": SURROGATE_BATCH_SIZE,
    "surrogate_vectorized_batch_used": bool(SURROGATE_PREDICTOR.batch_supported),
    "surrogate_batch_fallback_reason": SURROGATE_PREDICTOR.fallback_reason or None,
    "surrogate_scalar_calls": SURROGATE_PREDICTOR.scalar_calls,
    "surrogate_batch_calls": SURROGATE_PREDICTOR.batch_calls,
    "opsim_cache_size": OPSIM_CACHE_SIZE,
    "opsim_cache_hits": OPSIM_VISIT_CACHE.hits,
    "opsim_cache_misses": OPSIM_VISIT_CACHE.misses,
}
(run_dir / "runtime_performance.json").write_text(
    json.dumps(performance, indent=2) + "\n"
)
run_info_sections["runtime_performance"] = performance
(run_dir / "run_info.txt").write_text(
    format_run_info_sections(run_info_sections)
)

print("Saved:", run_dir / "summary.csv")
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
        f"synthetic={n_synthetic_plots}, LSST-like={n_lsstlike_plots}",
        f"in {pngdir}",
    )
