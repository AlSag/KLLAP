#!/usr/bin/env python3
"""Inspect an SN Ia mock population before building colour envelopes.

The input is the long-format CSV or Parquet table described in
``README_snia_mock_6band.md``. The script keeps the original luminosity-
distance diagnostics and adds LSST ``ugrizy`` light-curve figures for either
explicit SN IDs or a representative/random subset of the population.

Light-curve phase is observer-frame ``mjd - t0``. ``t0`` is the SALT2 time of
maximum light. The input mock is only inspected: no MW extinction, m5 cut,
interpolation, or format conversion is applied here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PERCENTILES = (0.0, 1.0, 5.0, 16.0, 50.0, 84.0, 95.0, 99.0, 100.0)
BAND_ORDER = ("u", "g", "r", "i", "z", "y")
BAND_COLORS = {
    "u": "#7b3294",
    "g": "#008837",
    "r": "#d73027",
    "i": "#f46d43",
    "z": "#8c510a",
    "y": "#4d4d4d",
}
LIGHT_CURVE_COLUMNS = (
    "sn_id", "mjd", "band", "mag", "magerr", "mag_perfect", "redshift",
    "t0", "x0", "x1", "c", "ra", "dec",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("snia_mock_6band.parquet"),
        help="SN Ia CSV or Parquet table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to <input parent>/snia_inspection",
    )
    parser.add_argument("--histogram-bins", type=int, default=50)
    parser.add_argument(
        "--sn-ids",
        default="",
        help=(
            "Comma-separated SN IDs to plot. When supplied, these replace "
            "the automatic representative/random selection."
        ),
    )
    parser.add_argument(
        "--n-light-curves",
        type=int,
        default=6,
        help="Number of automatically selected SNe Ia; 0 disables light curves.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["redshift_quantiles", "random"],
        default="redshift_quantiles",
        help="Automatic event-selection method when --sn-ids is empty.",
    )
    parser.add_argument(
        "--light-curve-seed",
        type=int,
        default=20260929,
        help="Random seed used only with --selection-mode random.",
    )
    parser.add_argument(
        "--bands",
        default=",".join(BAND_ORDER),
        help="Comma-separated LSST bands to display.",
    )
    parser.add_argument(
        "--light-curve-mode",
        choices=["both", "perfect", "noisy"],
        default="both",
        help=(
            "Photometry to draw: noise-free mag_perfect, noisy mag with "
            "magerr, or both."
        ),
    )
    parser.add_argument(
        "--phase-min",
        type=float,
        help="Optional lower observer-frame phase limit in days relative to t0.",
    )
    parser.add_argument(
        "--phase-max",
        type=float,
        help="Optional upper observer-frame phase limit in days relative to t0.",
    )
    parser.add_argument(
        "--individual-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write one six-band figure per selected SN Ia.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each saved figure interactively in addition to writing PNGs.",
    )
    return parser


def parse_sn_ids(text: str) -> list[int]:
    if not text.strip():
        return []
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid SN ID: {token!r}") from exc
        if value not in values:
            values.append(value)
    return values


def parse_bands(text: str) -> list[str]:
    bands: list[str] = []
    for token in text.split(","):
        band = token.strip().lower()
        if not band:
            continue
        if band not in BAND_ORDER:
            raise ValueError(
                f"Unknown band {band!r}; choose from {', '.join(BAND_ORDER)}"
            )
        if band not in bands:
            bands.append(band)
    if not bands:
        raise ValueError("At least one band must be selected")
    return bands


def read_object_redshifts(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            data = pd.read_parquet(path, columns=["sn_id", "redshift"])
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet. "
                "Install pyarrow in the environment used for this script."
            ) from exc
    elif suffix in {".csv", ".txt"}:
        data = pd.read_csv(path, usecols=["sn_id", "redshift"])
    else:
        raise ValueError("Input must be a .parquet, .pq, .csv, or .txt table")

    data["sn_id"] = pd.to_numeric(data["sn_id"], errors="coerce")
    data["redshift"] = pd.to_numeric(data["redshift"], errors="coerce")
    data = data.dropna(subset=["sn_id", "redshift"]).copy()
    data["sn_id"] = data["sn_id"].astype(np.int64)
    if (data["redshift"] < 0).any():
        raise ValueError("Negative redshifts are not supported")

    counts = data.groupby("sn_id")["redshift"].nunique(dropna=True)
    inconsistent = counts[counts > 1]
    if len(inconsistent):
        raise ValueError(
            f"{len(inconsistent)} SN Ia objects have more than one redshift"
        )
    objects = (
        data.drop_duplicates("sn_id")[["sn_id", "redshift"]]
        .sort_values("sn_id")
        .reset_index(drop=True)
    )
    if objects.empty:
        raise ValueError("No finite SN Ia redshifts were found")
    return objects


def read_band_availability(path: Path) -> pd.DataFrame:
    """Count saved rows in each LSST band for every SN Ia object."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            data = pd.read_parquet(path, columns=["sn_id", "band"])
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet."
            ) from exc
        data["sn_id"] = pd.to_numeric(data["sn_id"], errors="coerce")
        data = data.dropna(subset=["sn_id", "band"]).copy()
        data["sn_id"] = data["sn_id"].astype(np.int64)
        data["band"] = data["band"].astype(str).str.strip().str.lower()
        counts = data.groupby(["sn_id", "band"]).size().unstack(fill_value=0)
    elif suffix in {".csv", ".txt"}:
        partial_counts = []
        for chunk in pd.read_csv(
            path, usecols=["sn_id", "band"], chunksize=250_000
        ):
            chunk["sn_id"] = pd.to_numeric(chunk["sn_id"], errors="coerce")
            chunk = chunk.dropna(subset=["sn_id", "band"]).copy()
            chunk["sn_id"] = chunk["sn_id"].astype(np.int64)
            chunk["band"] = chunk["band"].astype(str).str.strip().str.lower()
            partial_counts.append(chunk.groupby(["sn_id", "band"]).size())
        if not partial_counts:
            raise ValueError("No band information was found in the input")
        counts = (
            pd.concat(partial_counts)
            .groupby(level=[0, 1])
            .sum()
            .unstack(fill_value=0)
        )
    else:
        raise ValueError("Input must be a .parquet, .pq, .csv, or .txt table")
    counts = counts.reindex(columns=BAND_ORDER, fill_value=0).astype(int)
    counts = counts.rename(columns=lambda band: f"N_rows_{band}")
    return counts.reset_index().sort_values("sn_id").reset_index(drop=True)


def build_distance_summary(objects: pd.DataFrame) -> pd.DataFrame:
    z = objects["redshift"].to_numpy(float)
    distance = objects["luminosity_distance_mpc"].to_numpy(float)
    rows = [
        {"quantity": "N_objects", "value": int(len(objects)), "unit": "count"},
        {"quantity": "redshift_min", "value": float(z.min()), "unit": ""},
        {"quantity": "redshift_max", "value": float(z.max()), "unit": ""},
        {
            "quantity": "luminosity_distance_min",
            "value": float(distance.min()),
            "unit": "Mpc",
        },
        {
            "quantity": "luminosity_distance_max",
            "value": float(distance.max()),
            "unit": "Mpc",
        },
    ]
    for percentile, z_value, distance_value in zip(
        PERCENTILES,
        np.percentile(z, PERCENTILES),
        np.percentile(distance, PERCENTILES),
    ):
        label = f"p{percentile:g}".replace(".", "p")
        rows.extend([
            {
                "quantity": f"redshift_{label}",
                "value": float(z_value),
                "unit": "",
            },
            {
                "quantity": f"luminosity_distance_{label}",
                "value": float(distance_value),
                "unit": "Mpc",
            },
        ])
    return pd.DataFrame(rows)


def select_light_curve_ids(
    objects: pd.DataFrame,
    requested_ids: list[int],
    number: int,
    mode: str,
    seed: int,
) -> list[int]:
    available = set(objects["sn_id"].astype(int))
    if requested_ids:
        missing = [sn_id for sn_id in requested_ids if sn_id not in available]
        if missing:
            raise ValueError(f"SN IDs absent from the input: {missing}")
        return requested_ids
    if number == 0:
        return []
    number = min(number, len(objects))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return sorted(
            int(value)
            for value in rng.choice(
                objects["sn_id"].to_numpy(np.int64), size=number, replace=False
            )
        )
    ranked = objects.sort_values(["redshift", "sn_id"]).reset_index(drop=True)
    indices = np.rint(np.linspace(0, len(ranked) - 1, number)).astype(int)
    return ranked.iloc[indices]["sn_id"].astype(int).tolist()


def read_selected_photometry(path: Path, sn_ids: list[int]) -> pd.DataFrame:
    if not sn_ids:
        return pd.DataFrame(columns=LIGHT_CURVE_COLUMNS)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            data = pd.read_parquet(
                path,
                columns=list(LIGHT_CURVE_COLUMNS),
                filters=[("sn_id", "in", sn_ids)],
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet."
            ) from exc
        except Exception as exc:
            raise ValueError(
                "Could not read the expected SN Ia light-curve columns "
                f"{list(LIGHT_CURVE_COLUMNS)} from {path}: {exc}"
            ) from exc
    elif suffix in {".csv", ".txt"}:
        chunks = []
        try:
            iterator = pd.read_csv(
                path, usecols=list(LIGHT_CURVE_COLUMNS), chunksize=250_000
            )
            for chunk in iterator:
                selected = chunk.loc[chunk["sn_id"].isin(sn_ids)]
                if not selected.empty:
                    chunks.append(selected)
        except ValueError as exc:
            raise ValueError(
                "The CSV does not contain all expected SN Ia columns"
            ) from exc
        data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
            columns=LIGHT_CURVE_COLUMNS
        )
    else:
        raise ValueError("Input must be a .parquet, .pq, .csv, or .txt table")

    for column in (
        "sn_id", "mjd", "mag", "magerr", "mag_perfect", "redshift", "t0",
        "x0", "x1", "c", "ra", "dec",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["sn_id", "mjd", "band", "t0"]).copy()
    data["sn_id"] = data["sn_id"].astype(np.int64)
    data["band"] = data["band"].astype(str).str.strip().str.lower()
    data["phase_days"] = data["mjd"] - data["t0"]
    data = data.loc[data["sn_id"].isin(sn_ids)].copy()
    if data.empty:
        raise ValueError("No photometric rows were found for the selected SN IDs")
    return data.sort_values(["sn_id", "band", "phase_days"]).reset_index(drop=True)


def light_curve_metadata(
    photometry: pd.DataFrame,
    objects: pd.DataFrame,
    selected_ids: list[int],
    bands: list[str],
) -> pd.DataFrame:
    first = (
        photometry.sort_values(["sn_id", "mjd"])
        .drop_duplicates("sn_id")[[
            "sn_id", "redshift", "t0", "x0", "x1", "c", "ra", "dec"
        ]]
        .set_index("sn_id")
    )
    counts = (
        photometry.loc[photometry["band"].isin(bands)]
        .groupby(["sn_id", "band"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=bands, fill_value=0)
        .rename(columns=lambda band: f"N_{band}")
    )
    phase = photometry.groupby("sn_id")["phase_days"].agg(
        phase_min_days="min", phase_max_days="max"
    )
    distance = objects.set_index("sn_id")[["luminosity_distance_mpc"]]
    result = first.join(distance).join(phase).join(counts)
    result = result.reindex(selected_ids).reset_index()
    result.insert(0, "selection_order", np.arange(1, len(result) + 1))
    return result


def style_light_curve_axis(
    ax: plt.Axes,
    phase_min: float | None,
    phase_max: float | None,
) -> None:
    ax.axvline(0.0, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.8)
    if phase_min is not None or phase_max is not None:
        left, right = ax.get_xlim()
        ax.set_xlim(
            phase_min if phase_min is not None else left,
            phase_max if phase_max is not None else right,
        )
    ax.invert_yaxis()
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.set_xlabel(r"Observer-frame phase $\mathrm{MJD}-t_0$ [days]")
    ax.set_ylabel("Apparent AB magnitude")


def draw_light_curve(
    ax: plt.Axes,
    curve: pd.DataFrame,
    bands: list[str],
    mode: str,
    *,
    alpha: float = 1.0,
    label_event: bool = True,
) -> None:
    for band in bands:
        subset = curve.loc[curve["band"] == band].sort_values("phase_days")
        if subset.empty:
            continue
        color = BAND_COLORS[band]
        if mode in {"both", "perfect"}:
            finite = np.isfinite(subset["phase_days"]) & np.isfinite(
                subset["mag_perfect"]
            )
            ax.plot(
                subset.loc[finite, "phase_days"],
                subset.loc[finite, "mag_perfect"],
                color=color,
                linewidth=1.25 if label_event else 0.75,
                alpha=0.90 * alpha,
                zorder=2,
            )
        if mode in {"both", "noisy"}:
            finite = (
                np.isfinite(subset["phase_days"])
                & np.isfinite(subset["mag"])
                & np.isfinite(subset["magerr"])
            )
            ax.errorbar(
                subset.loc[finite, "phase_days"],
                subset.loc[finite, "mag"],
                yerr=subset.loc[finite, "magerr"],
                fmt="o",
                markersize=2.3 if label_event else 1.4,
                markeredgewidth=0,
                color=color,
                ecolor=color,
                elinewidth=0.45,
                capsize=0,
                alpha=0.62 * alpha,
                zorder=3,
            )


def legend_handles(bands: list[str], mode: str) -> tuple[list[Line2D], list[Line2D]]:
    band_handles = [
        Line2D([0], [0], color=BAND_COLORS[band], lw=2.0, label=band)
        for band in bands
    ]
    style_handles: list[Line2D] = []
    if mode in {"both", "perfect"}:
        style_handles.append(
            Line2D([0], [0], color="#111827", lw=1.4, label="mag_perfect")
        )
    if mode in {"both", "noisy"}:
        style_handles.append(
            Line2D(
                [0], [0], color="#111827", marker="o", linestyle="none",
                markersize=4, label="mag ± magerr",
            )
        )
    return band_handles, style_handles


def annotate_event(ax: plt.Axes, row: pd.Series) -> None:
    ax.text(
        0.025,
        0.035,
        (
            f"SN {int(row['sn_id'])}  |  z={row['redshift']:.4f}  |  "
            f"$D_L$={row['luminosity_distance_mpc']:.1f} Mpc"
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25", "facecolor": "white",
            "edgecolor": "#d1d5db", "alpha": 0.86,
        },
    )


def finish_figure(fig: plt.Figure, path: Path, show: bool) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_distance_outputs(
    output: Path,
    path: Path,
    objects: pd.DataFrame,
    histogram_bins: int,
    show: bool,
) -> dict[str, object]:
    summary = build_distance_summary(objects)
    objects.to_csv(output / "snia_luminosity_distance_by_object.csv", index=False)
    summary.to_csv(output / "snia_luminosity_distance_summary.csv", index=False)
    summary_dict: dict[str, object] = {
        "input_file": str(path),
        "cosmology": "astropy.cosmology.Planck18",
        "N_objects": int(len(objects)),
        "redshift_min": float(objects["redshift"].min()),
        "redshift_max": float(objects["redshift"].max()),
        "luminosity_distance_min_mpc": float(
            objects["luminosity_distance_mpc"].min()
        ),
        "luminosity_distance_max_mpc": float(
            objects["luminosity_distance_mpc"].max()
        ),
        "percentiles": {
            f"p{percentile:g}": {
                "redshift": float(np.percentile(objects["redshift"], percentile)),
                "luminosity_distance_mpc": float(
                    np.percentile(objects["luminosity_distance_mpc"], percentile)
                ),
            }
            for percentile in PERCENTILES
        },
    }
    (output / "snia_luminosity_distance_summary.json").write_text(
        json.dumps(summary_dict, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    ax.hist(
        objects["luminosity_distance_mpc"],
        bins=histogram_bins,
        color="#4c78a8",
        alpha=0.84,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel("Luminosity distance [Mpc]")
    ax.set_ylabel("Number of SNe Ia")
    ax.grid(axis="y", alpha=0.18)
    finish_figure(
        fig, output / "snia_luminosity_distance_distribution.png", show
    )
    return summary_dict


def save_light_curve_outputs(
    output: Path,
    photometry: pd.DataFrame,
    metadata: pd.DataFrame,
    selected_ids: list[int],
    bands: list[str],
    mode: str,
    phase_min: float | None,
    phase_max: float | None,
    individual_plots: bool,
    show: bool,
) -> None:
    light_output = output / "light_curves"
    light_output.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(light_output / "snia_selected_light_curves.csv", index=False)
    band_handles, style_handles = legend_handles(bands, mode)

    n_events = len(selected_ids)
    n_columns = 2 if n_events > 1 else 1
    n_rows = int(np.ceil(n_events / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.4 * n_columns, 4.4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, sn_id in zip(axes.flat, selected_ids):
        curve = photometry.loc[photometry["sn_id"] == sn_id]
        row = metadata.loc[metadata["sn_id"] == sn_id].iloc[0]
        draw_light_curve(axis, curve, bands, mode)
        style_light_curve_axis(axis, phase_min, phase_max)
        annotate_event(axis, row)
    for axis in axes.flat[n_events:]:
        axis.set_visible(False)
    fig.legend(
        handles=band_handles + style_handles,
        loc="outside lower center",
        ncol=len(band_handles) + len(style_handles),
        frameon=False,
        fontsize=8,
    )
    finish_figure(
        fig, light_output / "snia_selected_light_curves_grid.png", show
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for sn_id in selected_ids:
        curve = photometry.loc[photometry["sn_id"] == sn_id]
        draw_light_curve(
            ax, curve, bands, mode, alpha=max(0.25, 1.0 / np.sqrt(n_events)),
            label_event=False,
        )
    style_light_curve_axis(ax, phase_min, phase_max)
    ax.legend(
        handles=band_handles + style_handles,
        loc="best",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    finish_figure(
        fig, light_output / "snia_selected_light_curves_overlay.png", show
    )

    if individual_plots:
        for sn_id in selected_ids:
            curve = photometry.loc[photometry["sn_id"] == sn_id]
            row = metadata.loc[metadata["sn_id"] == sn_id].iloc[0]
            fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
            draw_light_curve(ax, curve, bands, mode)
            style_light_curve_axis(ax, phase_min, phase_max)
            annotate_event(ax, row)
            ax.legend(
                handles=band_handles + style_handles,
                loc="best",
                ncol=2,
                frameon=False,
                fontsize=8,
            )
            finish_figure(
                fig,
                light_output / f"snia_{sn_id:06d}_light_curve.png",
                show,
            )


def main() -> Path:
    args = build_parser().parse_args()
    path = args.input.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if args.histogram_bins < 1:
        raise ValueError("histogram-bins must be at least 1")
    if args.n_light_curves < 0:
        raise ValueError("n-light-curves must be non-negative")
    if (
        args.phase_min is not None
        and args.phase_max is not None
        and args.phase_max <= args.phase_min
    ):
        raise ValueError("phase-max must be greater than phase-min")

    try:
        from astropy.cosmology import Planck18
    except ImportError as exc:
        raise ImportError(
            "This script uses astropy.cosmology.Planck18. Install astropy in "
            "the environment used to run it."
        ) from exc

    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else path.parent / "snia_inspection"
    )
    output.mkdir(parents=True, exist_ok=True)

    objects = read_object_redshifts(path)
    objects["luminosity_distance_mpc"] = (
        Planck18.luminosity_distance(objects["redshift"].to_numpy(float))
        .to_value("Mpc")
    )
    summary_dict = save_distance_outputs(
        output, path, objects, args.histogram_bins, args.show
    )

    bands = parse_bands(args.bands)
    requested_ids = parse_sn_ids(args.sn_ids)
    availability = read_band_availability(path)
    availability.to_csv(
        output / "snia_band_availability_by_object.csv", index=False
    )
    requested_count_columns = [f"N_rows_{band}" for band in bands]
    complete_ids = availability.loc[
        availability[requested_count_columns].gt(0).all(axis=1), "sn_id"
    ].astype(int)
    automatic_pool = objects.loc[objects["sn_id"].isin(complete_ids)].copy()
    if not requested_ids and automatic_pool.empty:
        raise ValueError(
            "No SN Ia object has saved rows in every requested display band"
        )
    selected_ids = select_light_curve_ids(
        objects if requested_ids else automatic_pool,
        requested_ids,
        args.n_light_curves,
        args.selection_mode,
        args.light_curve_seed,
    )
    if selected_ids:
        photometry = read_selected_photometry(path, selected_ids)
        metadata = light_curve_metadata(
            photometry, objects, selected_ids, bands
        )
        save_light_curve_outputs(
            output,
            photometry,
            metadata,
            selected_ids,
            bands,
            args.light_curve_mode,
            args.phase_min,
            args.phase_max,
            args.individual_plots,
            args.show,
        )
    summary_dict["light_curves"] = {
        "selected_sn_ids": selected_ids,
        "selection_mode": "explicit_ids" if requested_ids else args.selection_mode,
        "bands": bands,
        "mode": args.light_curve_mode,
        "phase_definition": "observer-frame MJD - SALT2 t0",
        "phase_min_days": args.phase_min,
        "phase_max_days": args.phase_max,
        "mw_extinction_applied": False,
        "m5_selection_applied": False,
        "automatic_selection_requires_all_display_bands": True,
        "N_objects_with_all_display_bands": int(len(automatic_pool)),
        "N_objects_with_rows_by_band": {
            band: int((availability[f"N_rows_{band}"] > 0).sum())
            for band in bands
        },
    }
    (output / "snia_inspection_configuration.json").write_text(
        json.dumps(summary_dict, indent=2) + "\n"
    )

    print(f"Objects: {len(objects):,}")
    print(
        "Redshift range: "
        f"{objects['redshift'].min():.8f} -- {objects['redshift'].max():.8f}"
    )
    print(
        "Luminosity-distance range (Planck18): "
        f"{objects['luminosity_distance_mpc'].min():.3f} -- "
        f"{objects['luminosity_distance_mpc'].max():.3f} Mpc"
    )
    if selected_ids:
        print("Light-curve SN IDs:", ", ".join(map(str, selected_ids)))
    else:
        print("Light-curve plotting disabled")
    print("Output:", output)
    return output


if __name__ == "__main__":
    main()
