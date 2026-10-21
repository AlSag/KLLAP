#!/usr/bin/env python3
"""Presentation of completed colour-envelope data products.

Every function receives colours, rules, overlap tables, or classifications
that were calculated elsewhere. This module does not interpolate light curves,
apply m5 or Milky-Way cuts, bootstrap percentiles, or decide validity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from kne_envelope_core import M5_STAGES, STAGES
from kne_envelope_plots import (
    comparison_ylim,
    draw_comparison_envelope,
    draw_envelope,
    draw_population,
    draw_population_style,
    format_axes,
    format_snia_axes,
)


def event_colour_map(event_indices: np.ndarray) -> dict[int, str]:
    """Assign stable, well-separated display colours in the supplied order."""
    golden_ratio_conjugate = 0.6180339887498949
    return {
        int(event_index): to_hex(
            hsv_to_rgb(((rank * golden_ratio_conjugate) % 1.0, 0.72, 0.82))
        )
        for rank, event_index in enumerate(event_indices)
    }

def save_snia_population_plots(
    output: Path,
    pair: str,
    t_min: float,
    t_max: float,
    times: list[np.ndarray],
    colours: list[np.ndarray],
    rules_all: pd.DataFrame,
    rules_consecutive: pd.DataFrame,
    time_reference: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
    draw_population_style(ax, times, colours, "#d97706", 0.025, 0.20)
    format_snia_axes(
        ax, pair, t_min, t_max, comparison=False,
        time_reference=time_reference,
    )
    limits = comparison_ylim(colours)
    if limits is not None:
        ax.set_ylim(*limits)
    ax.legend(handles=[
        Line2D([0], [0], color="#d97706", lw=1.4, label="SN Ia population")
    ], frameon=False)
    fig.savefig(
        output / "snia_population_unbinned.png",
        dpi=220, bbox_inches="tight",
    )
    plt.close(fig)

    for envelope_name, rules in (
        ("all_valid", rules_all),
        ("consecutive", rules_consecutive),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
        draw_population_style(ax, times, colours, "#d97706", 0.025, 0.20)
        draw_comparison_envelope(ax, rules, "#d97706", "-", 0.20)
        format_snia_axes(
            ax, pair, t_min, t_max, comparison=False,
            time_reference=time_reference,
        )
        limits = comparison_ylim(colours, rules)
        if limits is not None:
            ax.set_ylim(*limits)
        ax.legend(handles=[
            Line2D([0], [0], color="#d97706", lw=1.6, label="SN Ia envelope")
        ], frameon=False)
        fig.savefig(
            output / f"snia_population_with_envelope_{envelope_name}.png",
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)


def save_kne_snia_overlay_plots(
    output: Path,
    pair: str,
    t_min: float,
    t_max: float,
    kne_times: list[np.ndarray],
    kne_colours: list[np.ndarray],
    snia_times: list[np.ndarray],
    snia_colours: list[np.ndarray],
    kne_rules: pd.DataFrame,
    snia_rules: pd.DataFrame,
    overlap: pd.DataFrame,
    envelope_name: str,
    time_reference: str,
) -> None:
    for show_population, stem in (
        (False, "KNe_SNIa_envelopes"),
        (True, "KNe_SNIa_population_and_envelopes"),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
        if show_population:
            draw_population_style(ax, kne_times, kne_colours, "#111111", 0.010, 0.14)
            draw_population_style(ax, snia_times, snia_colours, "#d97706", 0.018, 0.18)
        draw_comparison_envelope(ax, kne_rules, "#2f5f98", "-", 0.16)
        draw_comparison_envelope(ax, snia_rules, "#d97706", "--", 0.14)
        for row in overlap.dropna(
            subset=["overlap_color_min", "overlap_color_max"]
        ).itertuples():
            ax.fill_between(
                [row.t_min_days, row.t_max_days],
                [row.overlap_color_min] * 2,
                [row.overlap_color_max] * 2,
                color="#2ca02c", alpha=0.26,
            )
        format_snia_axes(
            ax, pair, t_min, t_max, comparison=True,
            time_reference=time_reference,
        )
        limits = comparison_ylim(
            kne_colours if show_population else [],
            snia_colours if show_population else [],
            kne_rules,
            snia_rules,
        )
        if limits is not None:
            ax.set_ylim(*limits)
        ax.legend(handles=[
            Line2D([0], [0], color="#2f5f98", lw=1.7, label="KNe envelope"),
            Line2D([0], [0], color="#d97706", lw=1.7, linestyle="--", label="SN Ia envelope"),
            Patch(facecolor="#2ca02c", alpha=0.26, label="Envelope intersection"),
        ], frameon=False, fontsize=8)
        fig.savefig(
            output / f"{stem}_{envelope_name}.png",
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)


def save_at2017gfo_overlay(
    results: pd.DataFrame,
    rules: pd.DataFrame,
    bestfit_curve: pd.DataFrame | None,
    path: Path,
    pair: str,
    t_min: float,
    t_max: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
    if not rules.empty:
        draw_envelope(ax, rules)
    finite = results.loc[
        np.isfinite(pd.to_numeric(results["phase_days"], errors="coerce"))
        & np.isfinite(pd.to_numeric(results["color"], errors="coerce"))
    ].sort_values("phase_days")
    if not finite.empty:
        ax.plot(
            finite["phase_days"], finite["color"], color="#222222", lw=1.1,
            alpha=0.75, zorder=4,
        )
        evaluated = finite["evaluated"].astype(bool)
        pass_values = finite["passes_filter"].astype("boolean").fillna(False).astype(bool)
        central_values = (
            finite["passes_filter_central_value"]
            .astype("boolean").fillna(False).astype(bool)
        )
        passed_central = evaluated & pass_values & central_values
        passed_via_error = evaluated & pass_values & ~central_values
        failed = evaluated & ~pass_values
        unavailable = ~evaluated
        for mask, color, marker, label, zorder in (
            (passed_central, "#009e73", "o", "AT2017gfo: central value passes", 6),
            (
                passed_via_error, "#e69f00", "D",
                "AT2017gfo: compatible within photometric error", 6,
            ),
            (failed, "#d62728", "X", "AT2017gfo: outside", 7),
            (unavailable, "#777777", "s", "AT2017gfo: no valid envelope", 5),
        ):
            subset = finite.loc[mask]
            if subset.empty:
                continue
            ax.errorbar(
                subset["phase_days"], subset["color"],
                yerr=subset["color_error"], fmt=marker, ms=5.5,
                color=color, ecolor=color, elinewidth=0.8, capsize=2,
                linestyle="none", label=label, zorder=zorder,
            )
    finite_bestfit = pd.DataFrame()
    if bestfit_curve is not None and not bestfit_curve.empty:
        finite_bestfit = bestfit_curve.loc[
            np.isfinite(pd.to_numeric(bestfit_curve["phase_days"], errors="coerce"))
            & np.isfinite(pd.to_numeric(bestfit_curve["color"], errors="coerce"))
        ].sort_values("phase_days")
        if not finite_bestfit.empty:
            ax.plot(
                finite_bestfit["phase_days"], finite_bestfit["color"],
                color="#cc33aa", lw=2.2, alpha=0.95,
                label="AT2017gfo best-fit model", zorder=5,
            )
    format_axes(ax, pair, t_min, t_max)
    arrays = []
    if not rules.empty:
        arrays.append(rules[["color_min", "color_max"]].to_numpy(float).ravel())
    if not finite.empty:
        arrays.append(finite["color"].to_numpy(float))
    if not finite_bestfit.empty:
        arrays.append(finite_bestfit["color"].to_numpy(float))
    if arrays:
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        if values.size:
            low, high = np.min(values), np.max(values)
            padding = max(0.1, 0.06 * max(high - low, 0.1))
            ax.set_ylim(low - padding, high + padding)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_at2017gfo_failed_table(results: pd.DataFrame, path: Path) -> None:
    failed = results.loc[
        results["evaluated"].astype(bool)
        & ~results["passes_filter"].astype("boolean").fillna(False).astype(bool)
    ].copy()
    figure_height = max(1.8, 0.42 * (len(failed) + 2))
    fig, ax = plt.subplots(figsize=(13.5, figure_height), constrained_layout=True)
    ax.axis("off")
    if failed.empty:
        ax.text(
            0.5, 0.5, "All evaluated AT2017gfo points pass",
            ha="center", va="center", fontsize=13,
        )
    else:
        display = pd.DataFrame({
            "date UTC": failed["date_utc"].str.replace("T", " ").str.slice(0, 19),
            "phase [d]": failed["phase_days"].map(lambda x: f"{x:.3f}"),
            "colour [mag]": failed.apply(
                lambda row: f"{row['color']:.3f} +/- {row['color_error']:.3f}", axis=1
            ),
            "allowed [mag]": failed.apply(
                lambda row: f"[{row['color_min']:.3f}, {row['color_max']:.3f}]", axis=1
            ),
            "side": failed["result"].astype(str),
            "excursion [mag]": failed["excursion_beyond_envelope_mag"].map(
                lambda x: f"{x:.3f}"
            ),
        })
        table = ax.table(
            cellText=display.to_numpy(), colLabels=display.columns,
            cellLoc="center", colLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.3)
        table.scale(1.0, 1.3)
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#dbe8f5")
                cell.set_text_props(weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f4f6f8")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_permanent_exit_table_png(
    exits: pd.DataFrame,
    path: Path,
    maximum_rows: int,
    event_colors: dict[str, str] | None = None,
) -> None:
    """Save a compact, presentation-ready physical-parameter table."""
    shown = exits.head(maximum_rows).copy()
    base_columns = [
        "event_id", "last_outside_time_days", "exit_direction",
    ]
    base_columns += [column for column in ("M_ej_dyn_Msun", "M_ej_wind_Msun")
                     if column in shown and pd.to_numeric(shown[column], errors="coerce").notna().any()]
    optional_columns = [
        column for column in (
            "v_ej_dyn_c", "Ye_dyn", "v_ej_wind_c", "Ye_wind", "phi_deg",
            "M_ej_Msun", "v_min_c", "beta", "kappa_r_cm2_g",
        )
        if column in shown
        and pd.to_numeric(shown[column], errors="coerce").notna().any()
    ]
    uses_inclination = ("inclination_used_by_model" not in shown
                        or shown["inclination_used_by_model"].fillna(True).astype(bool).any())
    display_columns = base_columns + optional_columns
    if uses_inclination:
        display_columns.append("inclination_deg")
    display_columns.append("luminosity_distance_Mpc")
    label_by_column = {
        "event_id": "event",
        "last_outside_time_days": "last out [d]",
        "exit_direction": "side",
        "M_ej_dyn_Msun": r"Mdyn [$M_\odot$]",
        "v_ej_dyn_c": "vdyn/c",
        "Ye_dyn": "Ye dyn",
        "M_ej_wind_Msun": r"Mwind [$M_\odot$]",
        "v_ej_wind_c": "vwind/c",
        "Ye_wind": "Ye wind",
        "phi_deg": "phi [deg]",
        "inclination_deg": "incl. [deg]",
        "luminosity_distance_Mpc": "DL [Mpc]",
        "M_ej_Msun": r"Mej [$M_\odot$]",
        "v_min_c": "vmin/c",
        "beta": "beta",
        "kappa_r_cm2_g": r"kappa [cm$^2$/g]",
    }
    labels = [label_by_column[column] for column in display_columns]
    figure_height = max(1.8, 0.38 * (len(shown) + 2))
    figure_width = max(11.0, 1.55 * len(display_columns))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
    ax.axis("off")
    if shown.empty:
        ax.text(0.5, 0.5, "No permanent exits", ha="center", va="center", fontsize=13)
    else:
        formatted = pd.DataFrame({
            "event_id": shown["event_id"].astype(str),
            "last_outside_time_days": shown["last_outside_time_days"].map(
                lambda x: f"{x:.2f}"
            ),
            "exit_direction": shown["exit_direction"].astype(str),
            "M_ej_dyn_Msun": shown["M_ej_dyn_Msun"].map(lambda x: f"{x:.3e}"),
            "v_ej_dyn_c": shown["v_ej_dyn_c"].map(lambda x: f"{x:.3f}"),
            "Ye_dyn": shown["Ye_dyn"].map(lambda x: f"{x:.3f}"),
            "M_ej_wind_Msun": shown["M_ej_wind_Msun"].map(lambda x: f"{x:.3e}"),
            "v_ej_wind_c": shown["v_ej_wind_c"].map(lambda x: f"{x:.3f}"),
            "Ye_wind": shown["Ye_wind"].map(lambda x: f"{x:.3f}"),
            "phi_deg": shown["phi_deg"].map(lambda x: f"{x:.1f}"),
            "inclination_deg": shown["inclination_deg"].map(lambda x: f"{x:.1f}"),
            "luminosity_distance_Mpc": shown["luminosity_distance_Mpc"].map(
                lambda x: f"{x:.1f}"
            ),
        })
        for column in ("M_ej_Msun", "v_min_c", "beta", "kappa_r_cm2_g"):
            if column in shown:
                formatted[column] = shown[column].map(lambda x: f"{x:.3g}")
        table = ax.table(
            cellText=formatted[display_columns].to_numpy(),
            colLabels=labels,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1.0, 1.25)
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#dbe8f5")
                cell.set_text_props(weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f4f6f8")
        if event_colors:
            for row_index, event_id in enumerate(shown["event_id"].astype(str), start=1):
                table[(row_index, 0)].set_text_props(
                    color=event_colors.get(event_id, "black"),
                    weight="bold",
                )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stage_plots(
    stage_dir: Path,
    pair: str,
    t_min: float,
    t_max: float,
    population_times: list[np.ndarray],
    population_colors: list[np.ndarray],
    all_valid: pd.DataFrame,
    consecutive: pd.DataFrame,
    coloured_samples: dict[int, np.ndarray],
    colour_by_event_index: dict[int, str],
    event_ids: list[str],
) -> None:
    specifications = [
        ("complete_population_unbinned.png", True, None),
        ("envelope_all_valid.png", False, all_valid),
        ("envelope_consecutive.png", False, consecutive),
        ("complete_population_unbinned_with_envelope_all_valid.png", True, all_valid),
        ("complete_population_unbinned_with_envelope_consecutive.png", True, consecutive),
    ]
    flattened_colors = (
        np.concatenate(population_colors) if population_colors else np.array([])
    )
    flattened_colors = flattened_colors[np.isfinite(flattened_colors)]
    rule_arrays = []
    for table in (all_valid, consecutive):
        if table is not None and not table.empty:
            for column in ("color_min", "color_median", "color_max"):
                if column in table:
                    rule_arrays.append(table[column].to_numpy(dtype=float))
    finite_rule_colors = (
        np.concatenate(rule_arrays) if rule_arrays else np.array([], dtype=float)
    )
    finite_rule_colors = finite_rule_colors[np.isfinite(finite_rule_colors)]
    for filename, show_population, rules in specifications:
        fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
        if show_population:
            draw_population(ax, population_times, population_colors)
        if rules is not None and not rules.empty:
            draw_envelope(ax, rules)
        format_axes(ax, pair, t_min, t_max)
        if len(flattened_colors):
            low, high = np.nanpercentile(flattened_colors, [0.05, 99.95])
            if finite_rule_colors.size:
                low = min(low, float(np.min(finite_rule_colors)))
                high = max(high, float(np.max(finite_rule_colors)))
            padding = max(0.1, 0.05 * (high - low))
            ax.set_ylim(low - padding, high + padding)
        fig.savefig(stage_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Readable event-level views.  The samples are nested and identical for
    # every photometric stage; colours are also kept fixed between stages.
    for sample_size, indices in coloured_samples.items():
        selected_times = [population_times[index] for index in indices]
        selected_colors = [population_colors[index] for index in indices]
        trajectory_colors = [colour_by_event_index[index] for index in indices]
        selected_event_ids = [event_ids[int(index)] for index in indices]
        legend_handles = [
            Line2D(
                [0], [0], color=color, lw=1.6,
                label=f"event {event_id}",
            )
            for event_id, color in zip(selected_event_ids, trajectory_colors)
        ]
        legend_columns = min(10, max(1, int(np.ceil(np.sqrt(sample_size)))))
        legend_rows = int(np.ceil(sample_size / legend_columns))
        show_legend = sample_size <= 10
        coloured_specifications = [
            (f"population_{sample_size}_random_coloured.png", None),
            (
                f"population_{sample_size}_random_coloured_with_envelope_all_valid.png",
                all_valid,
            ),
            (
                f"population_{sample_size}_random_coloured_with_envelope_consecutive.png",
                consecutive,
            ),
        ]
        for filename, rules in coloured_specifications:
            if show_legend:
                figure_height = 5.7 + 0.10 * max(0, legend_rows - 3)
                fig, ax = plt.subplots(figsize=(9.2, figure_height))
                fig.subplots_adjust(bottom=min(0.32, 0.16 + 0.023 * legend_rows))
            else:
                fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
            draw_population(
                ax,
                selected_times,
                selected_colors,
                trajectory_colors=trajectory_colors,
            )
            if rules is not None and not rules.empty:
                draw_envelope(ax, rules)
            format_axes(ax, pair, t_min, t_max)
            if len(flattened_colors):
                low, high = np.nanpercentile(flattened_colors, [0.05, 99.95])
                if finite_rule_colors.size:
                    low = min(low, float(np.min(finite_rule_colors)))
                    high = max(high, float(np.max(finite_rule_colors)))
                padding = max(0.1, 0.05 * (high - low))
                ax.set_ylim(low - padding, high + padding)
            if show_legend:
                fig.legend(
                    handles=legend_handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.015),
                    ncol=legend_columns,
                    frameon=False,
                    fontsize=7.2,
                    handlelength=1.6,
                    columnspacing=1.0,
                )
            fig.savefig(stage_dir / filename, dpi=220, bbox_inches="tight")
            plt.close(fig)


def stage_output_name(stage: str, scenario: str) -> str:
    return f"{stage}_{scenario}" if stage in M5_STAGES else stage


def save_dual_system_comparison_plots(
    output: Path,
    pair: str,
    scenario: str,
    t_min: float,
    t_max: float,
    at2017gfo_enabled: bool,
    systems: tuple[str, str] = ("lsst", "ps1"),
) -> None:
    """Overlay two survey envelopes built from identical physical events.

    AT2017gfo measurements are plotted only for the historical LSST/PS1
    comparison. They are PS1 photometry and must not be relabelled as ZTF.
    """
    system_style = {
        "lsst": ("#2f5f98", "Rubin/LSST envelope"),
        "ps1": ("#cc6677", "Pan-STARRS1 envelope"),
        "ztf": ("#dd8d29", "ZTF envelope"),
    }
    for stage in STAGES:
        stage_name = stage_output_name(stage, scenario)
        for envelope_name, rules_filename in (
            ("all_valid", "dynamic_color_cut_rules_all_valid.csv"),
            ("consecutive", "dynamic_color_cut_rules.csv"),
        ):
            rules_by_system: dict[str, pd.DataFrame] = {}
            for system in systems:
                path = output / system / stage_name / rules_filename
                if path.exists():
                    rules_by_system[system] = pd.read_csv(path)
            if not rules_by_system:
                continue

            fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
            handles: list[Line2D] = []
            color_arrays = []
            for system, rules in rules_by_system.items():
                color, label = system_style[system]
                for row in rules.itertuples():
                    x = [row.t_min_days, row.t_max_days]
                    ax.fill_between(
                        x,
                        [row.color_min, row.color_min],
                        [row.color_max, row.color_max],
                        color=color,
                        alpha=0.14,
                    )
                    ax.plot(x, [row.color_min] * 2, color=color, lw=0.9)
                    ax.plot(x, [row.color_max] * 2, color=color, lw=0.9)
                    ax.plot(
                        x, [row.color_median] * 2, color=color, lw=1.35,
                        linestyle="-" if system == systems[0] else "--",
                    )
                if not rules.empty:
                    color_arrays.append(
                        rules[["color_min", "color_max"]].to_numpy(float).ravel()
                    )
                handles.append(Line2D(
                    [0], [0], color=color, lw=1.8,
                    linestyle="-" if system == systems[0] else "--",
                    label=label,
                ))

            at_points = pd.DataFrame()
            if at2017gfo_enabled and "ps1" in systems:
                candidate = (
                    output / "ps1" / stage_name
                    / f"AT2017gfo_{envelope_name}_color_points.csv"
                )
                if candidate.exists():
                    at_points = pd.read_csv(candidate)
                    at_points = at_points.loc[
                        np.isfinite(pd.to_numeric(at_points["phase_days"], errors="coerce"))
                        & np.isfinite(pd.to_numeric(at_points["color"], errors="coerce"))
                    ].sort_values("phase_days")
                    if not at_points.empty:
                        ax.errorbar(
                            at_points["phase_days"], at_points["color"],
                            yerr=at_points["color_error"], color="#111111",
                            marker="o", ms=4.5, lw=1.0, elinewidth=0.8,
                            capsize=2, zorder=6,
                        )
                        handles.append(Line2D(
                            [0], [0], color="#111111", marker="o", lw=1.0,
                            label="AT2017gfo (PS1 photometry)",
                        ))
                        color_arrays.append(at_points["color"].to_numpy(float))

            format_axes(ax, pair, t_min, t_max)
            if color_arrays:
                values = np.concatenate(color_arrays)
                values = values[np.isfinite(values)]
                if values.size:
                    low, high = float(values.min()), float(values.max())
                    padding = max(0.1, 0.06 * max(high - low, 0.1))
                    ax.set_ylim(low - padding, high + padding)
            ax.legend(handles=handles, frameon=False, fontsize=8)
            comparison_prefix = "_".join(system.upper() for system in systems)
            fig.savefig(
                output
                / f"{comparison_prefix}_{stage}_{envelope_name}_comparison.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)
