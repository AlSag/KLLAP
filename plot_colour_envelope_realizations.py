#!/usr/bin/env python3
"""Compare colour-envelope quantiles across independent KNe realizations.

Each realization must contain one *maximum-size* synthetic population. Smaller
sample sizes are deterministic prefixes of the same realization
(``event_id < N``), so changes with N cannot be caused by a second draw.

The script reads the realization manifest written by
the interactive multi-realization mode of ``generate_kne_lightcurves.py`` and produces, for every
requested colour pair, one panel per sample size. At each realization and
target time the marker is p50 and the asymmetric error bar spans p16--p84.
Any nonempty bin has defined empirical percentiles. Empty bins remain in the
output table with a count of zero. Optional OpSim depth cuts apply to both bands.

Usage
-----
Run ``python plot_colour_envelope_realizations.py`` and answer the terminal questions.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from dynamic_color_cut_common import interpolate_band
from kne_m5 import load_m5_percentile


DEFAULT_TIME_COLOURS = ["#d62728", "#ff7f0e", "#1f77b4", "#111111"]


# -----------------------------------------------------------------------------
# USER-FRIENDLY TERMINAL INTERFACE AND INPUT VALIDATION
# -----------------------------------------------------------------------------


def comma_separated_values(text: str, cast):
    values = [cast(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required")
    return values


def colour_pair(text: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)\s*", text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid colour pair {text!r}; expected for example g-r"
        )
    return match.group(1), match.group(2)


def default_configuration() -> argparse.Namespace:
    """Return documented defaults before the terminal questions."""
    return argparse.Namespace(
        manifest=None, run_dir=None, sample_sizes="1000,10000",
        times="0.5,1.0,1.5,2.0", time_groups=None, colour_pairs="g-r",
        photometric_system="lsst",
        m5_cut=True, m5_file=None, m5_percentile=50.0,
        output_dir=None,
        no_connect=False,
    )




def prompt_text(label: str, default: str) -> str:
    entered = input(f"{label} [{default}]: ").strip()
    return entered or default


def prompt_choice(
    title: str,
    options: list[tuple[str, str]],
    default: str,
) -> str:
    print(f"\n{title}")
    default_index = next(
        index for index, (value, _) in enumerate(options, start=1)
        if value == default
    )
    for index, (_, label) in enumerate(options, start=1):
        suffix = " [default]" if index == default_index else ""
        print(f"  {index}. {label}{suffix}")
    entered = input(f"Choice [{default_index}]: ").strip()
    if not entered:
        return default
    try:
        selected = int(entered)
    except ValueError as exc:
        raise ValueError(f"Expected an integer choice, received {entered!r}") from exc
    if not 1 <= selected <= len(options):
        raise ValueError(f"Choice must lie between 1 and {len(options)}")
    return options[selected - 1][0]


def newest_manifest_default() -> str:
    candidates = list(Path("runs").glob("*_realizations_master_seed_*/realizations.csv"))
    if not candidates:
        return "runs/NAME_realizations_master_seed_SEED/realizations.csv"
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def configure_runtime() -> argparse.Namespace:
    args = default_configuration()
    # This public workflow is intentionally terminal-driven.
    print("Colour-envelope comparison across population realizations")
    args.manifest = Path(
        prompt_text("Realization manifest", newest_manifest_default())
    )
    args.run_dir = None
    args.sample_sizes = prompt_text(
        "Nested population sizes", args.sample_sizes
    )
    n_time_groups = int(prompt_text("Number of separate time plots", "1"))
    if n_time_groups < 1:
        raise ValueError("The number of time plots must be at least one")
    time_group_defaults = [args.times, "3.0,4.0,6.0,8.0"]
    args.time_groups = []
    for group_index in range(n_time_groups):
        default_times = (
            time_group_defaults[group_index]
            if group_index < len(time_group_defaults)
            else args.times
        )
        args.time_groups.append(prompt_text(
            f"Post-merger times for plot {group_index + 1} [days]",
            default_times,
        ))
    args.colour_pairs = prompt_text(
        "Colour pairs", args.colour_pairs
    )
    args.photometric_system = prompt_choice(
        "Photometric system",
        [("lsst", "LSST"), ("ps1", "PS1"), ("ztf", "ZTF")],
        args.photometric_system,
    )
    depth_cut = prompt_choice(
        "Synthetic depth cut",
        [
            ("yes", "Require both bands to pass a selected m5 percentile"),
            ("no", "Do not apply an m5 cut"),
        ],
        "yes" if args.m5_cut else "no",
    )
    args.m5_cut = depth_cut == "yes"
    if args.m5_cut:
        m5_path = prompt_text(
            "m5 table or raw OpSim directory (automatic search if left automatic)",
            str(args.m5_file or "automatic"),
        )
        args.m5_file = None if m5_path == "automatic" else Path(m5_path)
        percentile_choice = prompt_choice(
            "OpSim m5 percentile used as the depth threshold",
            [
                ("25", "p25: shallower/worst-depth threshold"),
                ("50", "p50: median-depth threshold"),
                ("75", "p75: deeper/best-depth threshold"),
                ("90", "p90: very deep threshold"),
                ("custom", "Enter another percentile"),
            ],
            f"{args.m5_percentile:g}"
            if args.m5_percentile in {25.0, 50.0, 75.0, 90.0}
            else "custom",
        )
        args.m5_percentile = (
            float(prompt_text("Custom m5 percentile in [0,100]", f"{args.m5_percentile:g}"))
            if percentile_choice == "custom" else float(percentile_choice)
        )
    connection = prompt_choice(
        "Connect equal-time markers across realizations",
        [("yes", "Yes"), ("no", "No")],
        "no" if args.no_connect else "yes",
    )
    args.no_connect = connection == "no"
    output = prompt_text(
        "Output directory (automatic = beside the manifest)",
        str(args.output_dir or "automatic"),
    )
    args.output_dir = None if output == "automatic" else Path(output)
    if args.manifest is None and not args.run_dir:
        raise ValueError("Select a realization manifest or at least one run directory")
    return args


# -----------------------------------------------------------------------------
# DATA LOADING AND SCIENTIFIC CALCULATIONS
# -----------------------------------------------------------------------------


def read_realizations(args: argparse.Namespace) -> pd.DataFrame:
    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        table = pd.read_csv(manifest_path)
        required = {"realization", "run_dir"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(
                f"{manifest_path} is missing columns: {sorted(missing)}"
            )
        base = manifest_path.parent
        table["run_dir"] = [
            str((base / value).resolve())
            if not Path(value).expanduser().is_absolute()
            else str(Path(value).expanduser().resolve())
            for value in table["run_dir"].astype(str)
        ]
    else:
        table = pd.DataFrame(
            {
                "realization": np.arange(1, len(args.run_dir) + 1),
                "run_dir": [
                    str(path.expanduser().resolve()) for path in args.run_dir
                ],
            }
        )
    table = table.sort_values("realization").reset_index(drop=True)
    if table["realization"].duplicated().any():
        raise ValueError("Realization identifiers must be unique")
    for value in table["run_dir"]:
        if not Path(value).is_dir():
            raise FileNotFoundError(f"Run directory not found: {value}")
    return table


def synthetic_parquet_files(run_dir: Path) -> list[Path]:
    parquet_dir = run_dir / "parquet"
    manifest_path = parquet_dir / "parquet_manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        if {"product", "file"}.issubset(manifest.columns):
            files = [
                parquet_dir / filename
                for filename in manifest.loc[
                    manifest["product"].astype(str).eq("synthetic"), "file"
                ].astype(str)
            ]
            if files:
                return files
    return sorted(parquet_dir.glob("synthetic_part_*.parquet"))


def load_synthetic_population(run_dir: Path, maximum_n: int) -> pd.DataFrame:
    columns = ["event_id", "t_days", "photometric_system", "band", "mag"]
    parquet_files = synthetic_parquet_files(run_dir)
    if parquet_files:
        frames = []
        for path in parquet_files:
            frame = pd.read_parquet(path, columns=columns)
            frame["event_id"] = pd.to_numeric(
                frame["event_id"], errors="raise"
            ).astype(int)
            frame = frame.loc[frame["event_id"] < maximum_n]
            if not frame.empty:
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=columns)

    csv_dir = run_dir / "csv"
    frames = []
    pattern = re.compile(r"lightcurve_(\d+)\.csv$")
    for path in sorted(csv_dir.glob("lightcurve_*.csv")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        event_id = int(match.group(1))
        if event_id >= maximum_n:
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "event_id", event_id)
        if "photometric_system" not in frame:
            frame["photometric_system"] = "lsst"
        frames.append(frame[columns])
    if not frames:
        raise FileNotFoundError(
            f"No synthetic Parquet shards or per-event CSVs found in {run_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def interpolate_one_band(
    frame: pd.DataFrame,
    target_times: np.ndarray,
) -> np.ndarray:
    work = frame[["t_days", "band", "mag"]].copy()
    work["t_days"] = pd.to_numeric(work["t_days"], errors="coerce")
    work["magnitude"] = pd.to_numeric(work["mag"], errors="coerce")
    work = work.loc[
        np.isfinite(work["t_days"])
        & np.isfinite(work["magnitude"])
        & work["t_days"].gt(0)
    ]
    if work.empty:
        return np.full(len(target_times), np.nan)
    band = str(work["band"].iloc[0])
    return np.asarray(interpolate_band(work, band, target_times), dtype=float)


def event_colours(
    population: pd.DataFrame,
    photometric_system: str,
    pair: tuple[str, str],
    target_times: np.ndarray,
    depths: dict[str, float] | None,
) -> pd.DataFrame:
    first_band, second_band = pair
    selection_system = "lsst" if photometric_system == "ps1" else photometric_system
    systems = {photometric_system, selection_system}
    selected = population.loc[
        population["photometric_system"].astype(str).isin(systems)
        & population["band"].astype(str).isin(pair)
    ]
    if selected.empty:
        systems = sorted(population["photometric_system"].astype(str).unique())
        bands = sorted(population["band"].astype(str).unique())
        raise ValueError(
            f"No rows for system={photometric_system!r}, pair={first_band}-{second_band}. "
            f"Available systems={systems}, bands={bands}"
        )

    rows = []
    for event_id, event in selected.groupby("event_id", sort=True):
        target = event.loc[
            event["photometric_system"].astype(str).eq(photometric_system)
        ]
        selection = event.loc[
            event["photometric_system"].astype(str).eq(selection_system)
        ]
        band_values: dict[str, np.ndarray] = {}
        selection_values: dict[str, np.ndarray] = {}
        for band in pair:
            band_values[band] = interpolate_one_band(
                target.loc[target["band"].astype(str).eq(band)], target_times
            )
            selection_values[band] = interpolate_one_band(
                selection.loc[selection["band"].astype(str).eq(band)], target_times
            )
        colours = band_values[first_band] - band_values[second_band]
        available = np.isfinite(colours)
        if depths is not None:
            available &= (
                np.isfinite(selection_values[first_band])
                & np.isfinite(selection_values[second_band])
                & (selection_values[first_band] <= depths[first_band])
                & (selection_values[second_band] <= depths[second_band])
            )
        rows.extend(
            {
                "event_id": int(event_id),
                "time_days": float(time_days),
                "colour_mag": float(value),
            }
            for time_days, value, keep in zip(target_times, colours, available)
            if keep
        )
    return pd.DataFrame(rows, columns=["event_id", "time_days", "colour_mag"])


def summarize_nested_prefixes(colours, sample_sizes, target_times):
    """Summarize each prefix of injected IDs; an empty bin stays undefined."""
    rows = []
    for size in sample_sizes:
        prefix = colours.loc[colours["event_id"] < size]
        for time in target_times:
            values = prefix.loc[np.isclose(prefix["time_days"], time), "colour_mag"]
            values = values[np.isfinite(values)]
            n = len(values)
            low, median, high = np.percentile(values, [16, 50, 84]) if n else (np.nan,) * 3
            rows.append(dict(population_size=size, time_days=time, n_valid=n,
                             valid_fraction=n / size, p16=low, p50=median, p84=high,
                             median_valid=n > 0, envelope_valid=n > 0))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# FIGURE GENERATION AND OUTPUT TABLES
# -----------------------------------------------------------------------------


def time_palette(n_times: int) -> list[object]:
    if n_times <= len(DEFAULT_TIME_COLOURS):
        return DEFAULT_TIME_COLOURS[:n_times]
    return list(plt.get_cmap("viridis")(np.linspace(0.08, 0.92, n_times)))


def time_group_filename_tag(target_times: np.ndarray) -> str:
    """Return a compact, filesystem-safe description of one time group."""
    encoded = []
    for value in target_times:
        token = f"{float(value):g}".replace("-", "m").replace(".", "p")
        encoded.append(token)
    return "times_" + "_".join(encoded)


def plot_pair(
    values: pd.DataFrame,
    pair: tuple[str, str],
    sample_sizes: list[int],
    target_times: np.ndarray,
    output_dir: Path,
    connect: bool,
    output_tag: str | None = None,
) -> None:
    realizations = sorted(values["realization"].unique())
    n_panels = len(sample_sizes)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(7.2, 5.2 * n_panels), 6.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    palette = time_palette(len(target_times))
    offsets = (
        np.linspace(-0.24, 0.24, len(target_times))
        if len(target_times) > 1 else np.array([0.0])
    )

    for axis, sample_size in zip(axes, sample_sizes):
        panel = values.loc[values["population_size"].eq(sample_size)]
        for time_index, (time_days, colour, offset) in enumerate(
            zip(target_times, palette, offsets)
        ):
            series = panel.loc[np.isclose(panel["time_days"], time_days)].copy()
            series = series.set_index("realization").reindex(realizations)
            x = np.asarray(realizations, dtype=float) + offset
            y = series["p50"].to_numpy(float)
            p16 = series["p16"].to_numpy(float)
            p84 = series["p84"].to_numpy(float)
            lower = y - p16
            upper = p84 - y
            median_valid = (
                np.isfinite(y)
                & series["median_valid"].fillna(False).to_numpy(bool)
            )
            envelope_valid = (
                np.isfinite(y)
                & np.isfinite(lower)
                & np.isfinite(upper)
                & series["envelope_valid"].fillna(False).to_numpy(bool)
            )
            if connect:
                axis.plot(
                    x, np.where(median_valid, y, np.nan),
                    color=colour, alpha=0.45,
                    linewidth=1.0, zorder=1,
                )

            # The p16--p84 extent and p50 marker have independent validity.
            # The p50 value only centres the asymmetric error-bar geometry;
            # no median marker is drawn unless p50 itself is valid.
            axis.errorbar(
                x[envelope_valid], y[envelope_valid],
                yerr=np.vstack([
                    lower[envelope_valid], upper[envelope_valid]
                ]),
                fmt="none",
                capsize=3,
                elinewidth=1.25,
                color=colour,
                zorder=2 + time_index,
            )
            axis.plot(
                x[median_valid], y[median_valid],
                linestyle="none", marker="o",
                markersize=5.5, color=colour, zorder=4 + time_index,
            )
        axis.set_title(f"N = {sample_size:,}")
        axis.set_xlabel("Population realization")
        axis.set_xticks(realizations)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(f"{pair[0]}−{pair[1]} [mag]")
    time_handles = [
        Line2D([], [], color=colour, marker="o", linestyle="-", label=f"{time:g} d")
        for time, colour in zip(target_times, palette)
    ]
    time_legend = fig.legend(
        handles=time_handles, title=r"$t-t_{\rm merger}$",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(4, len(target_times)),
        frameon=False,
    )
    fig.add_artist(time_legend)
    status_handles = [
        Line2D([], [], color="0.25", marker="o", linestyle="none",
               label="median"),
        Line2D([], [], color="0.25", linestyle="-", linewidth=1.4,
               marker="_", markersize=9,
               label="16th–84th percentiles"),
    ]
    fig.legend(
        handles=status_handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
        ncol=len(status_handles), frameon=False,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    stem = f"{pair[0]}_minus_{pair[1]}_realization_envelopes"
    if output_tag:
        stem += f"_{output_tag}"
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    args = configure_runtime()
    sample_sizes = sorted(set(comma_separated_values(args.sample_sizes, int)))
    raw_time_groups = args.time_groups or [args.times]
    time_groups = [
        np.asarray(comma_separated_values(group, float), dtype=float)
        for group in raw_time_groups
    ]
    target_times = np.asarray(sorted({
        float(time)
        for group in time_groups
        for time in group
    }), dtype=float)
    pairs = [colour_pair(token) for token in args.colour_pairs.split(",")]
    if any(size <= 0 for size in sample_sizes):
        raise ValueError("Sample sizes must be positive")
    for group_index, group in enumerate(time_groups, start=1):
        if np.any(~np.isfinite(group)) or np.any(group <= 0):
            raise ValueError(
                f"Times in plot {group_index} must be finite and positive"
            )
        if len(np.unique(group)) != len(group):
            raise ValueError(
                f"Times in plot {group_index} must not contain duplicates"
            )
    if not 0.0 <= args.m5_percentile <= 100.0:
        raise ValueError("--m5-percentile must lie in [0, 100]")
    realizations = read_realizations(args)
    depth_tag = (
        "m5_p" + f"{args.m5_percentile:g}".replace(".", "p")
        if args.m5_cut else "no_m5"
    )
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif args.manifest is not None:
        output_dir = (
            args.manifest.expanduser().resolve().parent
            / f"colour_envelope_comparison_{depth_tag}"
        )
    else:
        output_dir = Path.cwd() / f"colour_envelope_comparison_{depth_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    maximum_n = max(sample_sizes)
    for record in realizations.itertuples(index=False):
        run_dir = Path(record.run_dir)
        summary_path = run_dir / "summary.csv"
        if summary_path.is_file() and len(pd.read_csv(summary_path, usecols=[0])) < maximum_n:
            raise ValueError(
                f"{run_dir} contains fewer than {maximum_n:,} injected events"
            )
        print(f"Reading realization {record.realization}: {run_dir}")
        population = load_synthetic_population(run_dir, maximum_n)
        for pair_index, pair in enumerate(pairs):
            if args.m5_cut:
                pair_depths, m5_source = load_m5_percentile(
                    run_dir, args.m5_file, args.m5_percentile, pair
                )
            else:
                pair_depths = None
                m5_source = "not_applied"
            colours = event_colours(
                population,
                args.photometric_system,
                pair,
                target_times,
                pair_depths,
            )
            values = summarize_nested_prefixes(colours, sample_sizes, target_times)
            values.insert(0, "colour_pair", f"{pair[0]}-{pair[1]}")
            values.insert(1, "photometric_system", args.photometric_system)
            values.insert(2, "m5_cut", bool(args.m5_cut))
            values.insert(3, "m5_source", m5_source)
            values.insert(4, "m5_percentile", args.m5_percentile if args.m5_cut else np.nan)
            values.insert(5, "m5_band1", pair_depths[pair[0]] if pair_depths else np.nan)
            values.insert(6, "m5_band2", pair_depths[pair[1]] if pair_depths else np.nan)
            values.insert(7, "realization", int(record.realization))
            if hasattr(record, "intrinsic_seed"):
                values.insert(8, "intrinsic_seed", int(record.intrinsic_seed))
            values["run_dir"] = str(run_dir)
            all_rows.append(values)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(output_dir / "colour_envelope_realization_values.csv", index=False)
    multiple_time_plots = len(time_groups) > 1
    for pair in pairs:
        pair_name = f"{pair[0]}-{pair[1]}"
        pair_values = result.loc[result["colour_pair"].eq(pair_name)]
        for group_index, group_times in enumerate(time_groups, start=1):
            output_tag = (
                time_group_filename_tag(group_times)
                if multiple_time_plots else None
            )
            plot_pair(
                pair_values,
                pair,
                sample_sizes,
                group_times,
                output_dir,
                connect=not args.no_connect,
                output_tag=output_tag,
            )
            if multiple_time_plots:
                print(
                    f"Saved time plot {group_index}/{len(time_groups)}: "
                    f"{pair[0]}-{pair[1]} at "
                    + ", ".join(f"{time:g} d" for time in group_times)
                )
    print(f"Saved comparison outputs: {output_dir}")


if __name__ == "__main__":
    main()
