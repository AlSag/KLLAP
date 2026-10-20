#!/usr/bin/env python3
"""Plotting-only helpers for OpSim export products.

Functions in this module receive already collected survey-depth arrays. They
do not query an OpSim database, select fields, evaluate dust, or write any
scientific table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

def save_m5_violin_plot(m5_values_by_band, outfile):
    """Save the pooled per-visit m5 distributions for the exported fields."""
    band_order = list("ugrizy")
    band_colors = {
        "u": "#ff7f0e",
        "g": "#2ca02c",
        "r": "#d62728",
        "i": "#9467bd",
        "z": "#8c564b",
        "y": "#e377c2",
    }

    bands = []
    distributions = []
    for band in band_order:
        chunks = m5_values_by_band.get(band, [])
        values = np.concatenate(chunks) if chunks else np.array([], dtype=float)
        values = values[np.isfinite(values)]
        if values.size >= 2:
            bands.append(band)
            distributions.append(values)

    if not distributions:
        print("No finite m5 distributions available: violin plot not created.")
        return

    positions = np.arange(1, len(bands) + 1)
    quantiles = np.asarray(
        [np.percentile(values, [16, 25, 50, 75, 84]) for values in distributions]
    )

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    violins = ax.violinplot(
        distributions,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=200,
        bw_method="scott",
    )
    for body in violins["bodies"]:
        body.set_facecolor("#7fb3d5")
        body.set_edgecolor("#7fb3d5")
        body.set_alpha(0.62)

    # Nested percentile intervals, followed by the p50 depth-cut marker.
    ax.vlines(
        positions,
        quantiles[:, 0],
        quantiles[:, 4],
        color="#1f77b4",
        linewidth=3,
        alpha=0.78,
        zorder=3,
    )
    ax.vlines(
        positions,
        quantiles[:, 1],
        quantiles[:, 3],
        color="#1f77b4",
        linewidth=9,
        alpha=0.52,
        zorder=4,
    )
    for position, band, median in zip(positions, bands, quantiles[:, 2]):
        ax.scatter(
            position,
            median,
            marker="D",
            s=72,
            color=band_colors[band],
            edgecolor="none",
            zorder=5,
        )

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=3, label="p16-p84"),
        Line2D([0], [0], color="#1f77b4", linewidth=9, alpha=0.52, label="p25-p75"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#ff7f0e",
            markeredgecolor="none",
            markersize=8,
            label="median (p50 threshold)",
        ),
    ]

    ax.set_xticks(positions, bands)
    ax.set_xlabel("LSST band")
    ax.set_ylabel(r"$m_5$ [mag]")
    ax.set_title(r"Per-visit LSST-like $m_5$ distributions; threshold = p50")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)

    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved m5 violin plot:", outfile)


def show_revisit_histograms(
    revisit_table: pd.DataFrame,
    n_fields: int,
    *,
    linear: bool,
    logarithmic: bool,
    bins: int,
) -> None:
    """Display optional revisit-cadence diagnostics from a completed table."""
    for band in sorted(revisit_table["band"].unique()):
        all_values = revisit_table.loc[
            revisit_table["band"] == band, "delta_t_days"
        ].to_numpy()
        if linear:
            values = all_values[np.isfinite(all_values)]
            if values.size:
                fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
                ax.hist(values, bins=bins, histtype="step", linewidth=2.0)
                ax.set_xlabel(r"$\Delta t$ (days)")
                ax.set_ylabel("Counts")
                ax.set_title(
                    f"Revisit Δt (linear bins) in band {band} "
                    f"(N_FIELDS={n_fields})"
                )
                ax.grid(True, alpha=0.25)
                plt.show()
        if logarithmic:
            values = all_values[np.isfinite(all_values) & (all_values > 0)]
            if values.size:
                log_bins = np.logspace(
                    np.log10(values.min()), np.log10(values.max()), bins
                )
                fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
                ax.hist(values, bins=log_bins, histtype="step", linewidth=2.0)
                ax.set_xscale("log")
                ax.set_xlabel(r"$\Delta t$ (days)")
                ax.set_ylabel("Counts")
                ax.set_title(
                    f"Revisit Δt (log bins + log x) in band {band} "
                    f"(N_FIELDS={n_fields})"
                )
                ax.grid(True, alpha=0.25)
                plt.show()
