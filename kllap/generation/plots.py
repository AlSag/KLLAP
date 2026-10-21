#!/usr/bin/env python3
"""Diagnostic plots for KNe light-curve generation.

The functions in this module only visualize arrays and tables produced by the
generator. They do not draw populations, evaluate models, apply detection
cuts, or mutate the saved scientific products.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "DejaVu Sans"


def plot_synthetic_only(
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


def plot_saved_synthetic_measurements(
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
    ax.set_ylabel("AB magnitude (before Milky-Way extinction)")
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


def plot_synthetic_and_lsstlike(
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


def plot_lsstlike_only(
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
