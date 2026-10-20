#!/usr/bin/env python3
"""Plot support and per-bin errors from a multiquantile convergence run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


GROUPS = ("central_90", "central_68", "central_50")
LABELS = {
    "central_90": "p5-p95",
    "central_68": "p16-p84",
    "central_50": "p25-p75",
}
COLOURS = {
    "central_90": "#ca6f1e",
    "central_68": "#2471a3",
    "central_50": "#229954",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace le nombre minimal de bins supportes et une heatmap des "
            "erreurs temporelles a partir des CSV de convergence."
        )
    )
    parser.add_argument(
        "--analysis-dir", type=Path, required=True,
        help="Dossier contenant les CSV de convergence.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Dossier de sortie (defaut: ANALYSIS_DIR/support_diagnostics).",
    )
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--t-max", type=float, default=4.0)
    parser.add_argument("--required-fraction", type=float, default=0.9)
    parser.add_argument("--tolerance-mag", type=float, default=0.10)
    return parser.parse_args()


def read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, float_precision="round_trip")
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonnes absentes de {path.name}: {sorted(missing)}")
    return frame


def quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q))


def support_statistics(per_bin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-repetition and across-repetition bin-support statistics."""
    by_bin = (
        per_bin.groupby(
            ["group_id", "N_requested", "repetition", "bin_index"],
            as_index=False,
        )
        .agg(bin_supported=("sample_valid", "all"))
    )
    per_repetition = (
        by_bin.groupby(["group_id", "N_requested", "repetition"], as_index=False)
        .agg(supported_shape_bins=("bin_supported", "sum"))
    )
    summary = (
        per_repetition.groupby(["group_id", "N_requested"], as_index=False)
        .agg(
            minimum_supported_shape_bins=("supported_shape_bins", "min"),
            supported_shape_bins_p16=("supported_shape_bins", lambda x: quantile(x, .16)),
            supported_shape_bins_median=("supported_shape_bins", "median"),
            supported_shape_bins_p84=("supported_shape_bins", lambda x: quantile(x, .84)),
        )
    )
    return per_repetition, summary


def plot_supported_bins(
    support: pd.DataFrame, n_bins: int, t_min: float, t_max: float,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for group in GROUPS:
        data = support.loc[support.group_id.eq(group)].sort_values("N_requested")
        if data.empty:
            continue
        x = data.N_requested.to_numpy(float)
        low = data.supported_shape_bins_p16.to_numpy(float)
        high = data.supported_shape_bins_p84.to_numpy(float)
        colour = COLOURS[group]
        ax.fill_between(x, low, high, color=colour, alpha=.10)
        ax.plot(
            x, data.minimum_supported_shape_bins, "o-", color=colour,
            label=f"{LABELS[group]} : minimum",
        )

    ax.axhline(
        n_bins, color="black", linestyle="--", linewidth=1.2,
        label=f"Fenêtre complète ({n_bins} bins)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Nombre de KNe injectées")
    ax.set_ylabel(
        "Nombre minimal de bins supportés\n"
        f"entre {t_min:g} et {t_max:g} jours"
    )
    ax.set_ylim(-.2, n_bins + .7)
    ax.set_yticks(range(n_bins + 1))
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#777777", alpha=.10, edgecolor="none"))
    labels.append("Zones transparentes : p16-p84 des répétitions")
    ax.legend(handles, labels, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def heatmap_statistics(
    per_bin: pd.DataFrame, required_fraction: float,
) -> pd.DataFrame:
    """Aggregate the worst family-member error in each temporal bin."""
    repeated = (
        per_bin.groupby(
            [
                "group_id", "N_requested", "repetition", "bin_index",
                "t_low_days", "t_high_days",
            ],
            as_index=False,
        )
        .agg(
            bin_supported=("both_supported", "all"),
            maximum_absolute_error_mag=("absolute_error_mag", "max"),
        )
    )
    repeated.loc[
        ~repeated.bin_supported, "maximum_absolute_error_mag"
    ] = np.nan

    heatmap = (
        repeated.groupby(
            ["group_id", "N_requested", "bin_index", "t_low_days", "t_high_days"],
            as_index=False,
        )
        .agg(
            support_fraction=("bin_supported", "mean"),
            median_maximum_absolute_error_mag=(
                "maximum_absolute_error_mag", "median"
            ),
        )
    )
    heatmap["displayed_as_valid"] = (
        heatmap.support_fraction + 1e-12 >= required_fraction
    )
    heatmap.loc[
        ~heatmap.displayed_as_valid, "median_maximum_absolute_error_mag"
    ] = np.nan
    return heatmap


def format_population(size: int) -> str:
    return f"{int(size):,}".replace(",", " ")


def plot_heatmaps(
    heatmap: pd.DataFrame,
    tolerance_mag: float,
    required_fraction: float,
    output: Path,
) -> None:
    present_groups = [g for g in GROUPS if heatmap.group_id.eq(g).any()]
    if not present_groups:
        raise ValueError("Aucune des familles central_90, central_68, central_50.")

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#bdbdbd")
    cmap.set_over("#d73027")
    fig, axes = plt.subplots(
        1, len(present_groups), figsize=(5.0 * len(present_groups), 8.0),
        sharey=True, squeeze=False,
    )
    image = None
    for column, group in enumerate(present_groups):
        ax = axes[0, column]
        data = heatmap.loc[heatmap.group_id.eq(group)].copy()
        sizes = sorted(data.N_requested.astype(int).unique())
        bins = (
            data[["bin_index", "t_low_days", "t_high_days"]]
            .drop_duplicates().sort_values("bin_index")
        )
        matrix = (
            data.pivot(
                index="N_requested", columns="bin_index",
                values="median_maximum_absolute_error_mag",
            )
            .reindex(index=sizes, columns=bins.bin_index)
            .to_numpy(float)
        )
        image = ax.imshow(
            np.ma.masked_invalid(matrix), origin="lower", aspect="auto",
            cmap=cmap, vmin=0.0, vmax=tolerance_mag, interpolation="nearest",
        )
        centres = (bins.t_low_days.to_numpy() + bins.t_high_days.to_numpy()) / 2
        ax.set_xticks(np.arange(len(bins)), [f"{t:g}" for t in centres], rotation=45)
        ax.set_yticks(np.arange(len(sizes)), [format_population(n) for n in sizes])
        ax.set_xlabel("Temps au centre du bin (jours)")
        ax.set_title(LABELS[group])
        ax.tick_params(length=0)

    axes[0, 0].set_ylabel("Nombre de KNe injectées")
    colourbar = fig.colorbar(
        image, ax=axes.ravel().tolist(), fraction=.025, pad=.03, extend="max",
    )
    colourbar.set_label(
        "Médiane sur les répétitions de l'écart absolu maximal (mag)"
    )
    fig.legend(
        handles=[Patch(
            facecolor="#bdbdbd", edgecolor="none",
            label=(
                f"Bin supporté dans moins de {required_fraction:.0%} "
                "des répétitions"
            ),
        )],
        loc="lower center", frameon=False,
    )
    fig.subplots_adjust(left=.10, right=.88, bottom=.14, top=.93, wspace=.08)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    source = args.analysis_dir.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else source / "support_diagnostics"
    )
    if not 0 < args.required_fraction <= 1:
        raise ValueError("--required-fraction doit appartenir a ]0, 1].")
    if args.t_max <= args.t_min:
        raise ValueError("--t-max doit etre strictement superieur a --t-min.")

    per_bin = read_csv(
        source / "percentile_value_convergence_per_bin.csv",
        {
            "repetition", "N_requested", "group_id", "bin_index",
            "t_low_days", "t_high_days", "sample_valid", "both_supported",
            "absolute_error_mag",
        },
    )
    per_bin = per_bin.loc[
        per_bin.group_id.isin(GROUPS)
        & per_bin.t_low_days.ge(args.t_min - 1e-12)
        & per_bin.t_high_days.le(args.t_max + 1e-12)
    ].copy()
    if per_bin.empty:
        raise ValueError("Aucune donnee dans la fenetre temporelle demandee.")

    output.mkdir(parents=True, exist_ok=True)
    _, support = support_statistics(per_bin)
    heatmap = heatmap_statistics(per_bin, args.required_fraction)
    n_bins = int(per_bin.bin_index.nunique())

    support.to_csv(output / "supported_shape_bins_values.csv", index=False)
    heatmap.to_csv(output / "percentile_error_heatmap_values.csv", index=False)
    plot_supported_bins(
        support, n_bins, args.t_min, args.t_max,
        output / "supported_shape_bins.png",
    )
    plot_heatmaps(
        heatmap, args.tolerance_mag, args.required_fraction,
        output / "percentile_error_heatmap.png",
    )
    print(f"Sorties : {output}")


if __name__ == "__main__":
    main()
