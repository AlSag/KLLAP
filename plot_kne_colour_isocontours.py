#!/usr/bin/env python3
"""Map median KNe colour and iso-colour contours in parameter planes.

The program reuses the population loading, log-time interpolation, Milky-Way
convention, and m5 cuts from ``plot_kne_colour_parameter_space.py``.  For each
time bin and each requested physical-parameter plane, it computes p16, p50,
and p84 of the colour in rectangular parameter cells.  The heatmap shows p50;
iso-colour contours are drawn from a support-aware smoothed p50 field.
Smoothing never fills unsupported cells, so contours are not extrapolated
beyond the measured population support.

The map is a conditional population summary, not a deterministic relation:
all physical parameters not shown on the axes remain distributed according to
the simulated population.  Cells with too few available colours are masked.
Comparison modes produce matched no-m5 and m5 outputs from the same events.
Any exact OpSim m5 percentile can be read from a table or calculated from raw
``opsim_visits_field*.csv`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

from kne_m5 import load_m5_percentile
import plot_kne_colour_parameter_space as base


COMPARE_STAGES = ("compare_no_mw", "compare_with_mw")
STAGE_CHOICES = (*base.STAGES, *COMPARE_STAGES)
M5_SCENARIO_PERCENTILES = {
    "worst_p25": 25.0,
    "median_p50": 50.0,
    "best_p75": 75.0,
}
MASS_AXIS_LABELS = {
    "log10_mej_dyn": r"$M_{\mathrm{ej}}^{\mathrm{dyn}}\ [M_\odot]$",
    "log10_mej_wind": r"$M_{\mathrm{ej}}^{\mathrm{wind}}\ [M_\odot]$",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--color-pair", default="g-r")
    p.add_argument(
        "--photometric-system", choices=("lsst", "ps1", "ztf"), default="lsst"
    )
    p.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="compare_no_mw",
        help=(
            "One photometric stage, or compare_no_mw/compare_with_mw to "
            "produce matched outputs without and with the m5 cut."
        ),
    )
    p.add_argument("--t-min", type=float, default=0.4)
    p.add_argument("--t-max", type=float, default=7.2)
    p.add_argument("--bin-width", type=float, default=0.4)
    p.add_argument(
        "--time-frame", choices=("observer", "source"), default="observer"
    )
    p.add_argument(
        "--parameter-pairs",
        default="log10_mej_dyn:log10_mej_wind",
        help="Comma-separated x:y planes.",
    )
    p.add_argument(
        "--contour-mode",
        choices=("fixed", "envelope"),
        default="fixed",
        help="Use one fixed level grid or adapt integer-spaced levels to an envelope CSV.",
    )
    p.add_argument(
        "--contour-levels",
        default="-3:3:1",
        help="Inclusive start:stop:step levels in fixed mode; use --contour-levels=-3:3:1.",
    )
    p.add_argument(
        "--contour-step",
        type=float,
        default=1.0,
        help="Level spacing in envelope mode.",
    )
    p.add_argument(
        "--bounds-file",
        type=Path,
        help="Envelope rules or percentile_reference_profiles.csv for envelope mode.",
    )
    p.add_argument(
        "--percentile-pair",
        default="5,95",
        help="Envelope percentiles when bounds-file is a fixed-profile CSV.",
    )
    p.add_argument("--reference-repetition", type=int, default=0)
    p.add_argument(
        "--colour-scale",
        help="Shared heatmap min:max. Omit for robust automatic bounds.",
    )
    p.add_argument(
        "--m5-percentile",
        type=float,
        default=75.0,
        help="OpSim depth percentile used as the two-band m5 threshold.",
    )
    p.add_argument(
        "--m5-scenario",
        choices=tuple(M5_SCENARIO_PERCENTILES),
        help="Legacy alias overriding --m5-percentile (p25, p50 or p75).",
    )
    p.add_argument(
        "--threshold-scope",
        choices=("common_global", "latitude_conditioned"),
        default="common_global",
    )
    p.add_argument("--m5-quantile-table", type=Path)
    p.add_argument("--ztf-m5-g", type=float)
    p.add_argument("--ztf-m5-r", type=float)
    p.add_argument("--ztf-m5-i", type=float)
    p.add_argument("--max-events", type=int, default=0, help="0 = all events")
    p.add_argument("--grid-size", type=int, default=20, help="Cells per parameter axis")
    p.add_argument(
        "--minimum-cell-events",
        type=int,
        default=20,
        help="Mask cells containing fewer available colours.",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing for contour geometry, in cells; heatmap remains unsmoothed.",
    )
    p.add_argument("--panels-per-page", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p


def interactive(args: argparse.Namespace) -> argparse.Namespace:
    print("Isocouleurs KNe dans l’espace des paramètres")
    args.run_dir = Path(base.ask("Dossier du run (contenant summary.csv)", "."))
    args.color_pair = base.ask("Paire de couleurs", args.color_pair)
    args.photometric_system = base.choose(
        "Système photométrique",
        [("lsst", "LSST"), ("ps1", "PS1"), ("ztf", "ZTF")],
        args.photometric_system,
    )
    args.stage = base.choose(
        "Sorties photométriques",
        [
            (base.STAGES[0], "Sans m5, sans MW"),
            (base.STAGES[1], "Sans m5, avec MW"),
            (base.STAGES[2], "Avec m5, sans MW"),
            (base.STAGES[3], "Avec m5, avec MW"),
            (
                "compare_no_mw",
                "Deux sorties comparables : sans m5 et avec m5, sans MW",
            ),
            (
                "compare_with_mw",
                "Deux sorties comparables : sans m5 et avec m5, avec MW",
            ),
        ],
        args.stage,
    )
    args.t_min = float(base.ask("Début des bins temporels [jours]", args.t_min))
    args.t_max = float(base.ask("Fin des bins temporels [jours]", args.t_max))
    args.bin_width = float(base.ask("Largeur des bins [jours]", args.bin_width))
    args.time_frame = base.choose(
        "Référence temporelle",
        [("observer", "Temps observateur"), ("source", "Temps source")],
        args.time_frame,
    )
    args.parameter_pairs = base.ask(
        "Plans x:y (séparés par une virgule)", args.parameter_pairs
    )
    args.contour_mode = base.choose(
        "Niveaux des isocouleurs",
        [
            ("fixed", "Grille fixe, par exemple -3,-2,...,3"),
            ("envelope", "Pas fixe, étendue adaptée à un CSV d’enveloppe"),
        ],
        args.contour_mode,
    )
    if args.contour_mode == "fixed":
        args.contour_levels = base.ask(
            "Niveaux start:stop:step [mag]", args.contour_levels
        )
    else:
        args.bounds_file = Path(base.ask("CSV de l’enveloppe"))
        args.percentile_pair = base.ask(
            "Percentiles si CSV de profils fixes", args.percentile_pair
        )
        args.contour_step = float(base.ask("Pas des isocouleurs [mag]", args.contour_step))
    if stage_uses_m5(args.stage):
        if args.photometric_system == "ztf":
            for band in args.color_pair.split("-"):
                setattr(
                    args,
                    f"ztf_m5_{band}",
                    float(base.ask(f"Magnitude limite ZTF {band} (5 sigma)")),
                )
        else:
            args.m5_percentile = float(
                base.ask(
                    "Percentile OpSim utilisé comme seuil m5 (p90 = 90)",
                    args.m5_percentile,
                )
            )
            explicit = base.ask(
                "Table m5 ou dossier OpSim brut (vide = recherche automatique)"
            )
            args.m5_quantile_table = Path(explicit) if explicit else None
    args.max_events = int(base.ask("Événements analysés (0 = tous)", args.max_events))
    args.grid_size = int(base.ask("Cellules par axe", args.grid_size))
    args.minimum_cell_events = int(
        base.ask("Couleurs minimales par cellule", args.minimum_cell_events)
    )
    args.smooth_sigma = float(
        base.ask("Lissage des contours [cellules]", args.smooth_sigma)
    )
    args.panels_per_page = int(
        base.ask("Nombre de panneaux par PNG", args.panels_per_page)
    )
    output = base.ask("Dossier de sortie (vide = automatique)")
    args.output_dir = Path(output) if output else None
    return args


def stage_uses_m5(stage: str) -> bool:
    return stage in {"m5_no_mw", "m5_with_mw", *COMPARE_STAGES}


def stage_plan(stage: str):
    """Return the loading convention and output-stage availability masks."""
    plans = {
        "intrinsic_no_m5_no_mw": (
            "intrinsic_no_m5_no_mw",
            (("intrinsic_no_m5_no_mw", "model"),),
        ),
        "no_m5_with_mw": (
            "no_m5_with_mw",
            (("no_m5_with_mw", "model"),),
        ),
        "m5_no_mw": ("m5_no_mw", (("m5_no_mw", "depth"),)),
        "m5_with_mw": ("m5_with_mw", (("m5_with_mw", "depth"),)),
        "compare_no_mw": (
            "m5_no_mw",
            (
                ("intrinsic_no_m5_no_mw", "model"),
                ("m5_no_mw", "depth"),
            ),
        ),
        "compare_with_mw": (
            "m5_with_mw",
            (
                ("no_m5_with_mw", "model"),
                ("m5_with_mw", "depth"),
            ),
        ),
    }
    return plans[stage]


def load_depths(args, root: Path, bands: tuple[str, str]):
    if not stage_uses_m5(args.stage):
        return {}, None
    percentile = (
        M5_SCENARIO_PERCENTILES[args.m5_scenario]
        if args.m5_scenario else args.m5_percentile
    )
    if not np.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError("--m5-percentile doit être strictement entre 0 et 100")
    args.m5_percentile = float(percentile)
    if args.photometric_system == "ztf":
        manual = {band: getattr(args, f"ztf_m5_{band}") for band in bands}
        if all(value is not None for value in manual.values()):
            if any(not np.isfinite(value) or not 0 < value < 40 for value in manual.values()):
                raise ValueError("Les profondeurs ZTF doivent être entre 0 et 40")
            return manual, "manual_ztf"
    return load_m5_percentile(
        root,
        args.m5_quantile_table,
        percentile,
        bands,
        args.threshold_scope,
    )


def time_grid(t_min: float, t_max: float, width: float):
    if not np.isfinite([t_min, t_max, width]).all() or width <= 0 or t_max <= t_min:
        raise ValueError("Grille temporelle invalide")
    steps = (t_max - t_min) / width
    if not np.isclose(steps, round(steps), atol=1e-9):
        raise ValueError("(t_max - t_min) doit être un multiple de bin_width")
    edges = np.linspace(t_min, t_max, int(round(steps)) + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def fixed_levels(text: str) -> np.ndarray:
    fields = [float(value.strip()) for value in text.split(":")]
    if len(fields) != 3:
        raise ValueError("Les niveaux fixes doivent suivre start:stop:step")
    start, stop, step = fields
    if not np.isfinite(fields).all() or step <= 0 or stop < start:
        raise ValueError("Niveaux fixes invalides")
    levels = np.arange(start, stop + 0.5 * step, step)
    levels[np.isclose(levels, 0.0, atol=1e-12)] = 0.0
    if len(levels) < 1:
        raise ValueError("Aucun niveau d’isocouleur")
    return levels


def parse_colour_scale(text: str | None):
    if text is None:
        return None
    fields = [float(value.strip()) for value in text.split(":")]
    if len(fields) != 2 or not np.isfinite(fields).all() or fields[1] <= fields[0]:
        raise ValueError("--colour-scale doit suivre min:max")
    return tuple(fields)


def envelope_levels(bounds: pd.DataFrame, step: float) -> list[np.ndarray]:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("--contour-step doit être fini et positif")
    levels = []
    for row in bounds.itertuples(index=False):
        if row.bounds_status != "valid":
            levels.append(np.array([], dtype=float))
            continue
        low = math.floor(float(row.color_min) / step) * step
        high = math.ceil(float(row.color_max) / step) * step
        values = np.arange(low, high + 0.5 * step, step)
        values[np.isclose(values, 0.0, atol=1e-12)] = 0.0
        levels.append(values)
    return levels


def plotting_axis(parameter: str, edges: np.ndarray):
    """Return plotting coordinates, label, and scale for one parameter axis."""
    if parameter in MASS_AXIS_LABELS:
        return np.power(10.0, edges), MASS_AXIS_LABELS[parameter], "log"
    return edges, base.PARAMETER_LABELS.get(parameter, parameter), "linear"


def mass_tick_label(value: float, _position=None) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    if value >= 1:
        return f"{value:g}"
    decimals = max(2, int(math.ceil(-math.log10(value))) + 1)
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def configure_axis_scale(ax, x_scale: str, y_scale: str) -> None:
    for axis, scale in ((ax.xaxis, x_scale), (ax.yaxis, y_scale)):
        if scale != "log":
            continue
        if axis is ax.xaxis:
            ax.set_xscale("log")
        else:
            ax.set_yscale("log")
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(mass_tick_label))
        axis.set_minor_locator(LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
        axis.set_minor_formatter(NullFormatter())


def quantile_statistic(q: float):
    def statistic(values):
        return float(np.quantile(values, q)) if len(values) else np.nan
    return statistic


def parameter_colour_grid(
    x: np.ndarray,
    y: np.ndarray,
    colour: np.ndarray,
    available: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    minimum_cell_events: int,
    smooth_sigma: float,
):
    mask = available & np.isfinite(x) & np.isfinite(y) & np.isfinite(colour)
    sample = colour[mask]
    coordinates = (x[mask], y[mask])
    shape = (len(x_edges) - 1, len(y_edges) - 1)
    if len(sample):
        count = np.histogram2d(*coordinates, bins=(x_edges, y_edges))[0]
        p16 = binned_statistic_2d(
            *coordinates, sample, statistic=quantile_statistic(.16),
            bins=(x_edges, y_edges),
        ).statistic
        p50 = binned_statistic_2d(
            *coordinates, sample, statistic="median",
            bins=(x_edges, y_edges),
        ).statistic
        p84 = binned_statistic_2d(
            *coordinates, sample, statistic=quantile_statistic(.84),
            bins=(x_edges, y_edges),
        ).statistic
    else:
        count = np.zeros(shape, dtype=float)
        p16 = p50 = p84 = np.full(shape, np.nan)
    supported = (count >= minimum_cell_events) & np.isfinite(p50)
    raw = np.where(supported, p50, np.nan)
    if smooth_sigma > 0 and supported.any():
        numerator = gaussian_filter(
            np.where(supported, p50, 0.0), sigma=smooth_sigma, mode="nearest"
        )
        denominator = gaussian_filter(
            supported.astype(float), sigma=smooth_sigma, mode="nearest"
        )
        smoothed = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 1e-8,
        )
        # Smooth only the geometry of contours inside measured support.
        # Unsupported cells remain undefined: no spatial extrapolation is made.
        smoothed[~supported] = np.nan
    else:
        smoothed = raw.copy()
    return {
        "count": count.astype(int),
        "p16": p16,
        "p50": p50,
        "p84": p84,
        "supported": supported,
        "smoothed_p50": smoothed,
        "N_available": int(mask.sum()),
    }


def robust_colour_domain(
    colour: np.ndarray,
    available: np.ndarray,
    requested_levels: list[np.ndarray],
    explicit: tuple[float, float] | None,
):
    if explicit is not None:
        return explicit
    values = colour[available & np.isfinite(colour)]
    if not len(values):
        raise ValueError("Aucune couleur disponible pour définir l’échelle")
    low, high = np.quantile(values, [.01, .99])
    nonempty_levels = [level for level in requested_levels if len(level)]
    all_levels = (
        np.concatenate(nonempty_levels)
        if nonempty_levels else np.array([], dtype=float)
    )
    if len(all_levels):
        low, high = min(low, all_levels.min()), max(high, all_levels.max())
    if not np.isfinite(low + high) or high <= low:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


def colour_norm(vmin: float, vmax: float):
    return (
        TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax)
    )


def draw_panel(
    ax,
    grid,
    x_edges,
    y_edges,
    levels,
    norm,
    cmap,
    time_low,
    time_high,
):
    raw = np.ma.masked_invalid(grid["p50"])
    raw.mask = np.ma.getmaskarray(raw) | ~grid["supported"]
    mesh = ax.pcolormesh(
        x_edges, y_edges, raw.T, shading="flat", cmap=cmap, norm=norm,
        rasterized=True,
    )
    field = grid["smoothed_p50"]
    finite = field[np.isfinite(field)]
    drawn = np.array([], dtype=float)
    if len(finite) and len(levels):
        drawn = levels[(levels >= finite.min()) & (levels <= finite.max())]
        if len(drawn):
            x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
            contours = ax.contour(
                x_centres,
                y_centres,
                np.ma.masked_invalid(field).T,
                levels=drawn,
                colors="black",
                linewidths=1.05,
                alpha=.92,
            )
            ax.clabel(contours, inline=True, fontsize=7.5, fmt="%g")
    supported_cells = int(grid["supported"].sum())
    ax.set_title(
        f"{time_low:g}–{time_high:g} j  ·  "
        f"N={grid['N_available']:,}  ·  cellules={supported_cells:,}",
        fontsize=9.5,
    )
    ax.set_facecolor("#e4e6e8")
    ax.grid(False)
    return mesh, drawn


def time_tag(low: float, high: float) -> str:
    def one(value):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return f"t{one(low)}_{one(high)}d"


def save_individual_panels(
    output: Path,
    stem: str,
    grids: list[dict],
    levels_by_time: list[np.ndarray],
    edges: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_label: str,
    y_label: str,
    x_scale: str,
    y_scale: str,
    norm,
    cmap,
    colour_label: str,
):
    for index, grid in enumerate(grids):
        fig, ax = plt.subplots(figsize=(7.0, 5.7))
        mesh, _ = draw_panel(
            ax, grid, x_edges, y_edges, levels_by_time[index], norm, cmap,
            edges[index], edges[index + 1],
        )
        configure_axis_scale(ax, x_scale, y_scale)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        colorbar = fig.colorbar(mesh, ax=ax, pad=.02)
        colorbar.set_label(f"{colour_label} median [mag]")
        fig.tight_layout()
        fig.savefig(
            output / f"{stem}_{time_tag(edges[index], edges[index + 1])}.png",
            dpi=200,
        )
        plt.close(fig)


def save_overview_pages(
    output: Path,
    stem: str,
    grids: list[dict],
    levels_by_time: list[np.ndarray],
    edges: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_label: str,
    y_label: str,
    x_scale: str,
    y_scale: str,
    norm,
    cmap,
    colour_label: str,
    panels_per_page: int,
):
    columns = min(2, panels_per_page)
    for page, start in enumerate(range(0, len(grids), panels_per_page), 1):
        stop = min(start + panels_per_page, len(grids))
        count = stop - start
        rows = math.ceil(count / columns)
        fig = plt.figure(figsize=(5.1 * columns + .9, 4.0 * rows))
        layout = fig.add_gridspec(
            rows,
            columns + 1,
            width_ratios=[1] * columns + [.045],
            left=.075,
            right=.94,
            bottom=.075,
            top=.93,
            wspace=.20,
            hspace=.25,
        )
        axes = []
        shared = None
        for position in range(rows * columns):
            row, column = divmod(position, columns)
            ax = fig.add_subplot(
                layout[row, column], sharex=shared, sharey=shared
            )
            if shared is None:
                shared = ax
            axes.append(ax)
        mesh = None
        for local, index in enumerate(range(start, stop)):
            ax = axes[local]
            mesh, _ = draw_panel(
                ax, grids[index], x_edges, y_edges, levels_by_time[index],
                norm, cmap, edges[index], edges[index + 1],
            )
            configure_axis_scale(ax, x_scale, y_scale)
            row, column = divmod(local, columns)
            if row == rows - 1:
                ax.set_xlabel(x_label)
            if column == 0:
                ax.set_ylabel(y_label)
        for ax in axes[count:]:
            ax.set_visible(False)
        if mesh is not None:
            colorbar_axis = fig.add_subplot(layout[:rows, columns])
            colorbar = fig.colorbar(mesh, cax=colorbar_axis)
            colorbar.set_label(f"{colour_label} median [mag]")
        fig.savefig(output / f"{stem}_overview_{page:02d}.png", dpi=190)
        plt.close(fig)


def grid_table(
    pair,
    grids,
    levels_by_time,
    edges,
    x_edges,
    y_edges,
):
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    rows = []
    for index, grid in enumerate(grids):
        for ix, x_value in enumerate(x_centres):
            for iy, y_value in enumerate(y_centres):
                rows.append({
                    "t_low_days": edges[index],
                    "t_high_days": edges[index + 1],
                    "t_center_days": 0.5 * (edges[index] + edges[index + 1]),
                    "x_parameter": pair[0],
                    "y_parameter": pair[1],
                    "x_low": x_edges[ix],
                    "x_high": x_edges[ix + 1],
                    "x_center": x_value,
                    "y_low": y_edges[iy],
                    "y_high": y_edges[iy + 1],
                    "y_center": y_value,
                    "N_cell": int(grid["count"][ix, iy]),
                    "cell_supported": bool(grid["supported"][ix, iy]),
                    "colour_p16": grid["p16"][ix, iy],
                    "colour_p50": grid["p50"][ix, iy],
                    "colour_p84": grid["p84"][ix, iy],
                    "colour_p16_p84_width": (
                        grid["p84"][ix, iy] - grid["p16"][ix, iy]
                    ),
                    "smoothed_colour_p50": grid["smoothed_p50"][ix, iy],
                    "requested_contour_levels": ",".join(
                        f"{value:g}" for value in levels_by_time[index]
                    ),
                })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_dir is None:
        if not sys.stdin.isatty():
            parser.error("--run-dir requis hors terminal interactif")
        args = interactive(args)

    root = base.resolve_root(args.run_dir)
    time_edges, times = time_grid(args.t_min, args.t_max, args.bin_width)
    pairs = base.parse_parameter_pairs(args.parameter_pairs)
    bands = tuple(args.color_pair.strip().lower().split("-"))
    allowed = {"lsst": set("ugrizy"), "ps1": set("grizy"), "ztf": set("gri")}
    if len(bands) != 2 or bands[0] == bands[1] or not set(bands) <= allowed[args.photometric_system]:
        raise ValueError("Paire de couleurs invalide pour le système choisi")
    if args.max_events < 0 or args.grid_size < 5:
        raise ValueError("--max-events doit être >= 0 et --grid-size >= 5")
    if args.minimum_cell_events < 1:
        raise ValueError("--minimum-cell-events doit être >= 1")
    if not np.isfinite(args.smooth_sigma) or args.smooth_sigma < 0:
        raise ValueError("--smooth-sigma doit être fini et >= 0")
    if args.panels_per_page < 1:
        raise ValueError("--panels-per-page doit être positif")

    metadata = base.load_metadata(root, args.max_events, args.seed)
    parameter_arrays = {}
    for pair in pairs:
        for parameter in pair:
            parameter_arrays[parameter] = base.parameter_values(metadata, parameter)
            base.axis_domain(parameter_arrays[parameter])

    requested_stage = args.stage
    load_stage, output_stage_masks = stage_plan(requested_stage)
    args.stage = load_stage
    depths, depth_source = load_depths(args, root, bands)
    colour, model_supported, depth_supported = base.load_event_colours(
        args, root, metadata, bands, times, depths
    )
    args.stage = requested_stage
    availability = {
        stage: model_supported if mask_name == "model" else depth_supported
        for stage, mask_name in output_stage_masks
    }

    bounds_path = None
    if args.contour_mode == "fixed":
        level = fixed_levels(args.contour_levels)
        levels_by_stage = {
            stage: [level.copy() for _ in times] for stage in availability
        }
    else:
        if args.bounds_file is None:
            raise ValueError("--bounds-file requis en mode envelope")
        if args.time_frame != "observer":
            raise ValueError("Les CSV d’enveloppe sont en temps observateur")
        bounds_path = args.bounds_file.expanduser().resolve()
        q = tuple(float(value) for value in args.percentile_pair.split(","))
        if len(q) != 2 or not 0 < q[0] < q[1] < 100:
            raise ValueError("Percentiles requis : 0 < q_low < q_high < 100")
        levels_by_stage = {}
        for stage in availability:
            bounds = base.read_bounds(
                bounds_path, times, stage, q, args.reference_repetition
            )
            levels_by_stage[stage] = envelope_levels(bounds, args.contour_step)

    explicit_scale = parse_colour_scale(args.colour_scale)
    combined_available = np.logical_or.reduce(list(availability.values()))
    all_levels = [
        levels
        for stage_levels in levels_by_stage.values()
        for levels in stage_levels
    ]
    vmin, vmax = robust_colour_domain(
        colour, combined_available, all_levels, explicit_scale
    )
    norm = colour_norm(vmin, vmax)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e4e6e8")

    if stage_uses_m5(requested_stage):
        depth_tag = (
            "_m5manual" if depth_source == "manual_ztf"
            else f"_m5p{args.m5_percentile:g}".replace(".", "p")
        )
    else:
        depth_tag = ""
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else root / "analysis" /
        f"colour_isocontours_{args.color_pair.replace('-', '_')}_"
        f"{args.photometric_system}_{args.stage}{depth_tag}"
    )
    if output.is_dir() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Le dossier de sortie n’est pas vide; utiliser un autre dossier ou --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)

    summaries = []
    comparison = len(availability) > 1
    for stage, available in availability.items():
        stage_output = output / stage if comparison else output
        stage_output.mkdir(parents=True, exist_ok=True)
        stage_summaries = []
        levels_by_time = levels_by_stage[stage]
        for x_name, y_name in pairs:
            x = parameter_arrays[x_name]
            y = parameter_arrays[y_name]
            x_domain = base.axis_domain(x)
            y_domain = base.axis_domain(y)
            x_edges = np.linspace(*x_domain, args.grid_size + 1)
            y_edges = np.linspace(*y_domain, args.grid_size + 1)
            grids = []
            for index in range(len(times)):
                grid = parameter_colour_grid(
                    x,
                    y,
                    colour[:, index],
                    available[:, index],
                    x_edges,
                    y_edges,
                    args.minimum_cell_events,
                    args.smooth_sigma,
                )
                grids.append(grid)
                row = {
                    "stage": stage,
                    "m5_percentile": (
                        args.m5_percentile if stage.startswith("m5_") else np.nan
                    ),
                    "x_parameter": x_name,
                    "y_parameter": y_name,
                    "t_low_days": time_edges[index],
                    "t_high_days": time_edges[index + 1],
                    "t_center_days": times[index],
                    "N_total": len(metadata),
                    "N_model_supported": int(model_supported[:, index].sum()),
                    "N_stage_available": grid["N_available"],
                    "N_supported_cells": int(grid["supported"].sum()),
                    "requested_contour_levels": ",".join(
                        f"{value:g}" for value in levels_by_time[index]
                    ),
                }
                summaries.append(row)
                stage_summaries.append(row)
            stem = (
                f"isocolour_{stage}_{args.color_pair.replace('-', '_')}_"
                f"{x_name}_vs_{y_name}"
            )
            plot_x_edges, x_label, x_scale = plotting_axis(x_name, x_edges)
            plot_y_edges, y_label, y_scale = plotting_axis(y_name, y_edges)
            save_individual_panels(
                stage_output, stem, grids, levels_by_time, time_edges,
                plot_x_edges, plot_y_edges, x_label, y_label, x_scale, y_scale,
                norm, cmap, args.color_pair,
            )
            save_overview_pages(
                stage_output, stem, grids, levels_by_time, time_edges,
                plot_x_edges, plot_y_edges, x_label, y_label, x_scale, y_scale,
                norm, cmap, args.color_pair, args.panels_per_page,
            )
            table = grid_table(
                (x_name, y_name), grids, levels_by_time, time_edges,
                x_edges, y_edges,
            )
            table.insert(0, "stage", stage)
            table.to_csv(
                stage_output / f"{stem}_grid.csv.gz",
                index=False,
                compression="gzip",
            )
        pd.DataFrame(stage_summaries).to_csv(
            stage_output / "isocolour_summary.csv", index=False
        )

    pd.DataFrame(summaries).to_csv(output / "isocolour_summary.csv", index=False)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration.update({
        "run_dir": str(root),
        "requested_stage": requested_stage,
        "loading_stage": load_stage,
        "output_stages": list(availability),
        "time_edges_days": time_edges.tolist(),
        "time_centres_days": times.tolist(),
        "applied_depths": depths,
        "depth_source": depth_source,
        "N_analysed": len(metadata),
        "heatmap_statistic": "cell-wise median colour among stage-available events",
        "cell_spread": "cell-wise colour p16 and p84 saved in grid CSV",
        "contour_field": "support-aware Gaussian-smoothed cell-wise median colour",
        "marginalisation": "all unplotted parameters follow the simulated population",
        "interpolation": "saved magnitude linear in log10(observer time); no extrapolation",
        "colour_scale": [vmin, vmax],
        "summary_sha256": hashlib.sha256((root / "summary.csv").read_bytes()).hexdigest(),
        "bounds_sha256": (
            hashlib.sha256(bounds_path.read_bytes()).hexdigest()
            if bounds_path else None
        ),
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    (output / "isocolour_configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Sorties :", output)


if __name__ == "__main__":
    main()
