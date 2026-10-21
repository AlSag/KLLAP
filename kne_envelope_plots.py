"""Plotting primitives for colour-envelope products.

Functions in this module only draw values supplied by callers. They do not
select events, calculate percentiles, apply extinction, or perform m5 cuts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

def draw_population(
    ax: plt.Axes,
    time_tracks: list[np.ndarray],
    color_tracks: list[np.ndarray],
    trajectory_colors: list[str] | None = None,
) -> None:
    """Draw every event trajectory without time/colour binning."""
    segments = []
    segment_colors = []
    isolated_times = []
    isolated_colors = []
    isolated_plot_colors = []
    for track_index, (times, colors) in enumerate(zip(time_tracks, color_tracks)):
        times = np.asarray(times, dtype=float)
        colors = np.asarray(colors, dtype=float)
        plot_color = (
            trajectory_colors[track_index]
            if trajectory_colors is not None else "black"
        )
        finite = np.isfinite(times) & np.isfinite(colors)
        points = np.column_stack((times[finite], colors[finite]))
        if len(points) >= 2:
            segments.append(points)
            segment_colors.append(plot_color)
        elif len(points) == 1:
            isolated_times.append(points[0, 0])
            isolated_colors.append(points[0, 1])
            isolated_plot_colors.append(plot_color)
    if segments:
        collection = LineCollection(
            segments,
            colors=segment_colors,
            linewidths=0.16 if trajectory_colors is None else 0.75,
            alpha=0.018 if trajectory_colors is None else 0.78,
            rasterized=True,
        )
        ax.add_collection(collection)
    if isolated_times:
        ax.scatter(
            isolated_times,
            isolated_colors,
            s=0.18 if trajectory_colors is None else 3.0,
            alpha=0.025 if trajectory_colors is None else 0.78,
            color=isolated_plot_colors,
            edgecolors="none",
            rasterized=True,
        )
    if not segments and not isolated_times:
        return


def draw_envelope(ax: plt.Axes, rules: pd.DataFrame) -> None:
    for row in rules.itertuples():
        x = [row.t_min_days, row.t_max_days]
        ax.fill_between(
            x,
            [row.color_min, row.color_min],
            [row.color_max, row.color_max],
            color="#4c78a8",
            alpha=0.24,
        )
        ax.plot(x, [row.color_min] * 2, color="#2f5f98", lw=1.0)
        ax.plot(x, [row.color_max] * 2, color="#2f5f98", lw=1.0)
        ax.plot(x, [row.color_median] * 2, color="#d55e00", lw=1.25)


def format_axes(ax: plt.Axes, pair: str, t_min: float, t_max: float) -> None:
    ax.set_xlabel(r"$t-t_{\rm merger}$ [days]")
    ax.set_ylabel(f"{pair} [mag]")
    ax.set_xlim(t_min, t_max)
    ax.grid(alpha=0.18)


def format_snia_axes(
    ax: plt.Axes,
    pair: str,
    t_min: float,
    t_max: float,
    comparison: bool,
    time_reference: str,
) -> None:
    snia_reference = (
        r"$t-t_{\rm first\ observation}$"
        if time_reference == "first_observation"
        else r"$t-t_{\rm max}$"
    )
    if comparison:
        ax.set_xlabel(
            r"phase [days] (KNe: $t-t_{\rm merger}$; SNe Ia: "
            + snia_reference + ")"
        )
    else:
        ax.set_xlabel(snia_reference + " [days]")
    ax.set_ylabel(f"{pair} [mag]")
    ax.set_xlim(t_min, t_max)
    ax.grid(alpha=0.18)


def draw_population_style(
    ax: plt.Axes,
    time_tracks: list[np.ndarray],
    color_tracks: list[np.ndarray],
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    segments = []
    for times, colours in zip(time_tracks, color_tracks):
        times = np.asarray(times, dtype=float)
        colours = np.asarray(colours, dtype=float)
        finite = np.isfinite(times) & np.isfinite(colours)
        points = np.column_stack((times[finite], colours[finite]))
        if len(points) >= 2:
            segments.append(points)
        elif len(points) == 1:
            ax.scatter(
                points[0, 0], points[0, 1], s=0.2, color=color,
                alpha=alpha, edgecolors="none", rasterized=True,
            )
    if segments:
        ax.add_collection(LineCollection(
            segments, colors=color, linewidths=linewidth, alpha=alpha,
            rasterized=True,
        ))


def draw_comparison_envelope(
    ax: plt.Axes,
    rules: pd.DataFrame,
    color: str,
    linestyle: str,
    alpha: float,
) -> None:
    for row in rules.itertuples():
        x = [row.t_min_days, row.t_max_days]
        ax.fill_between(
            x, [row.color_min] * 2, [row.color_max] * 2,
            color=color, alpha=alpha,
        )
        ax.plot(x, [row.color_min] * 2, color=color, lw=0.9, linestyle=linestyle)
        ax.plot(x, [row.color_max] * 2, color=color, lw=0.9, linestyle=linestyle)
        ax.plot(x, [row.color_median] * 2, color=color, lw=1.4, linestyle=linestyle)


def comparison_ylim(
    *collections: list[np.ndarray] | pd.DataFrame,
) -> tuple[float, float] | None:
    arrays: list[np.ndarray] = []
    for collection in collections:
        if isinstance(collection, pd.DataFrame):
            if not collection.empty:
                arrays.append(
                    collection[["color_min", "color_max"]].to_numpy(float).ravel()
                )
        else:
            finite_tracks = [
                np.asarray(values, dtype=float) for values in collection
                if len(values)
            ]
            if finite_tracks:
                values = np.concatenate(finite_tracks)
                values = values[np.isfinite(values)]
                if values.size:
                    arrays.append(np.nanpercentile(values, [0.05, 99.95]))
    if not arrays:
        return None
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    low, high = float(values.min()), float(values.max())
    padding = max(0.1, 0.05 * max(high - low, 0.1))
    return low - padding, high + padding
