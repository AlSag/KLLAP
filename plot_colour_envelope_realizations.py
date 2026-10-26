#!/usr/bin/env python3
"""Compare colour-envelope quantiles across independent KNe realizations.

Each realization must contain one *maximum-size* synthetic population. Smaller
sample sizes are deterministic prefixes of the same realization
(``event_id < N``), so changes with N cannot be caused by a second draw.

The script reads the realization manifest written by
the interactive multi-realization mode of ``generate_kne_lightcurves.py`` and produces, for every
requested colour pair, one panel per sample size. At each realization and
target time the marker is p50 and the asymmetric error bar spans p16--p84.
The median and the p16--p84 extent are validated independently. A valid p50
can therefore be displayed without an interval, and a valid p16--p84 interval
can be displayed without a median marker. Invalid components are omitted from
the figure but remain fully documented in the output CSV.

Optionally, both bands must pass a selected percentile of the OpSim limiting-
depth distribution. Statistical support is checked from the expected number
of events in the p16/p84 tails.
An optional event bootstrap also checks that the 68% half-width of each
percentile estimator (p16, p50 and p84) is below a requested threshold.

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
        times="0.5,1.0,1.5,2.0", colour_pairs="g-r",
        photometric_system="lsst", mw_treatment="corrected",
        minimum_valid_events=30, minimum_tail_events=20.0,
        m5_cut=True, m5_file=None, m5_percentile=50.0,
        bootstrap=True, n_bootstrap=200, bootstrap_threshold_mag=0.10,
        bootstrap_seed=12345, plot_bootstrap_distributions=True,
        bootstrap_plot_realization=1,
        bootstrap_plot_sample_sizes="100,1000",
        bootstrap_plot_times="0.5,2.0", output_dir=None,
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
    if True:
        print("Colour-envelope comparison across population realizations")
        args.manifest = Path(
            prompt_text("Realization manifest", newest_manifest_default())
        )
        args.run_dir = None
        args.sample_sizes = prompt_text(
            "Nested population sizes", args.sample_sizes
        )
        args.times = prompt_text(
            "Post-merger times [days]", args.times
        )
        args.colour_pairs = prompt_text(
            "Colour pairs", args.colour_pairs
        )
        args.photometric_system = prompt_choice(
            "Photometric system",
            [("lsst", "LSST"), ("ps1", "PS1"), ("ztf", "ZTF")],
            args.photometric_system,
        )
        args.mw_treatment = prompt_choice(
            "Milky-Way extinction",
            [
                ("corrected", "Remove A_band event by event"),
                ("present", "Keep extinction in the saved magnitudes"),
            ],
            args.mw_treatment,
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
        args.minimum_valid_events = int(
            prompt_text(
                "Minimum finite colours required",
                str(args.minimum_valid_events),
            )
        )
        args.minimum_tail_events = float(
            prompt_text(
                "Minimum expected events in each p16/p84 tail",
                str(args.minimum_tail_events),
            )
        )
        bootstrap = prompt_choice(
            "Bootstrap validation of p16, p50 and p84",
            [("yes", "Enable bootstrap"), ("no", "Do not bootstrap")],
            "yes" if args.bootstrap else "no",
        )
        args.bootstrap = bootstrap == "yes"
        if args.bootstrap:
            args.n_bootstrap = int(
                prompt_text("Bootstrap repetitions", str(args.n_bootstrap))
            )
            args.bootstrap_threshold_mag = float(
                prompt_text(
                    "Maximum 68% bootstrap half-width [mag]",
                    str(args.bootstrap_threshold_mag),
                )
            )
            args.bootstrap_seed = int(
                prompt_text("Bootstrap master seed", str(args.bootstrap_seed))
            )
            plot_bootstrap = prompt_choice(
                "Bootstrap-distribution diagnostic",
                [
                    ("yes", "Plot selected bootstrap distributions"),
                    ("no", "Do not plot bootstrap distributions"),
                ],
                "yes" if args.plot_bootstrap_distributions else "no",
            )
            args.plot_bootstrap_distributions = plot_bootstrap == "yes"
            if args.plot_bootstrap_distributions:
                args.bootstrap_plot_realization = int(
                    prompt_text(
                        "Population realization to illustrate",
                        str(args.bootstrap_plot_realization),
                    )
                )
                args.bootstrap_plot_sample_sizes = prompt_text(
                    "Population sizes in the bootstrap diagnostic",
                    args.bootstrap_plot_sample_sizes,
                )
                args.bootstrap_plot_times = prompt_text(
                    "Times in the bootstrap diagnostic [days]",
                    args.bootstrap_plot_times,
                )
        else:
            args.plot_bootstrap_distributions = False
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
    target_extinction: pd.DataFrame | None,
    selection_extinction: pd.DataFrame | None,
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
            if target_extinction is not None:
                band_values[band] = (
                    band_values[band]
                    - target_extinction.at[int(event_id), band]
                )
            if selection_extinction is not None:
                selection_values[band] = (
                    selection_values[band]
                    - selection_extinction.at[int(event_id), band]
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


def event_extinction(
    run_dir: Path,
    photometric_system: str,
    pair: tuple[str, str],
) -> pd.DataFrame:
    summary_path = run_dir / "summary.csv"
    summary = pd.read_csv(summary_path)
    id_column = "event_index" if "event_index" in summary else "event_id"
    event_ids = pd.to_numeric(summary[id_column], errors="raise").astype(int)
    result = pd.DataFrame(index=event_ids)
    for band in pair:
        candidates = [f"A_{photometric_system}_{band}_mw", f"A_{band}_mw"]
        column = next((name for name in candidates if name in summary), None)
        if column is None:
            raise ValueError(
                f"Cannot remove MW extinction: {summary_path} has none of {candidates}"
            )
        result[band] = pd.to_numeric(summary[column], errors="coerce").to_numpy(float)
    if result[list(pair)].isna().any().any():
        raise ValueError(f"Non-finite A_band values in {summary_path}")
    return result


def bootstrap_percentiles_with_draws(
    values: np.ndarray,
    percentiles: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ordinary event bootstrap, including the individual estimates."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    percentiles = np.asarray(percentiles, dtype=float)
    empty = np.full(len(percentiles), np.nan)
    if not len(values):
        return empty, empty, empty, empty, np.empty((0, len(percentiles)))

    estimate = np.percentile(values, percentiles)
    if n_bootstrap <= 0 or len(values) < 2:
        return estimate, empty, empty, empty, np.empty((0, len(percentiles)))

    draws = np.empty((n_bootstrap, len(percentiles)), dtype=np.float32)
    cursor = 0
    while cursor < n_bootstrap:
        count = min(batch_size, n_bootstrap - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples = values[indices]
        draws[cursor:cursor + count] = np.percentile(
            samples, percentiles, axis=1
        ).T
        cursor += count
    low, high = np.percentile(draws, [16.0, 84.0], axis=0)
    return estimate, low, high, 0.5 * (high - low), draws


def summarize_nested_prefixes(
    colours: pd.DataFrame,
    sample_sizes: list[int],
    target_times: np.ndarray,
    minimum_valid_events: int,
    minimum_tail_events: float,
    bootstrap: bool,
    n_bootstrap: int,
    bootstrap_threshold_mag: float,
    bootstrap_seed: int,
    realization: int,
    pair_index: int,
    capture_bootstrap: bool = False,
    capture_sample_sizes: set[int] | None = None,
    capture_times: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    percentiles = np.array([16.0, 50.0, 84.0])
    percentile_names = ("p16", "p50", "p84")
    rows = []
    draw_rows = []
    for sample_size in sample_sizes:
        prefix = colours.loc[colours["event_id"] < sample_size]
        for time_index, time_days in enumerate(target_times):
            values = prefix.loc[
                np.isclose(prefix["time_days"], time_days), "colour_mag"
            ].to_numpy(float)
            values = values[np.isfinite(values)]
            rng = np.random.default_rng(np.random.SeedSequence([
                int(bootstrap_seed), int(realization), int(sample_size),
                int(time_index), int(pair_index),
            ]))
            if bootstrap:
                quantiles, boot_low, boot_high, boot_half, bootstrap_draws = (
                    bootstrap_percentiles_with_draws(
                        values, percentiles, n_bootstrap, rng
                    )
                )
            elif len(values):
                quantiles = np.percentile(values, percentiles)
                boot_low = boot_high = boot_half = np.full(3, np.nan)
                bootstrap_draws = np.empty((0, 3))
            else:
                quantiles = boot_low = boot_high = boot_half = np.full(3, np.nan)
                bootstrap_draws = np.empty((0, 3))

            enough_events = len(values) >= minimum_valid_events
            expected_tails = len(values) * np.minimum(
                percentiles, 100.0 - percentiles
            ) / 100.0
            tail_components = expected_tails >= minimum_tail_events
            tail_valid = bool(enough_events and np.all(tail_components))
            bootstrap_components = (
                np.isfinite(boot_half)
                & (boot_half <= bootstrap_threshold_mag)
            )
            tail_component_valid = enough_events & tail_components
            bootstrap_component_valid = (
                enough_events & bootstrap_components
                if bootstrap else np.ones(3, dtype=bool)
            )
            percentile_valid = tail_component_valid & bootstrap_component_valid
            envelope_valid = bool(percentile_valid[0] and percentile_valid[2])
            median_valid = bool(percentile_valid[1])
            bootstrap_valid = bool(
                enough_events and np.all(bootstrap_components)
            ) if bootstrap else True
            valid = bool(np.all(percentile_valid))
            if median_valid and envelope_valid:
                failure_scope = "valid"
            elif median_valid:
                failure_scope = "envelope"
            elif envelope_valid:
                failure_scope = "median"
            else:
                failure_scope = "median_and_envelope"

            component_status = []
            for tail_ok, boot_ok in zip(
                tail_component_valid, bootstrap_component_valid
            ):
                if not enough_events:
                    component_status.append("insufficient_events")
                elif tail_ok and boot_ok:
                    component_status.append("valid")
                elif not tail_ok and bootstrap and not boot_ok:
                    component_status.append("tail_and_bootstrap_invalid")
                elif not tail_ok:
                    component_status.append("tail_support_invalid")
                else:
                    component_status.append("bootstrap_width_invalid")
            if valid:
                status = "valid"
            elif not tail_valid and bootstrap and not bootstrap_valid:
                status = "tail_and_bootstrap_invalid"
            elif not tail_valid:
                status = (
                    "insufficient_events"
                    if not enough_events else "tail_support_invalid"
                )
            else:
                status = "bootstrap_width_invalid"

            selected_for_diagnostic = (
                capture_bootstrap
                and sample_size in (capture_sample_sizes or set())
                and capture_times is not None
                and np.any(np.isclose(capture_times, time_days))
            )
            if selected_for_diagnostic and len(bootstrap_draws):
                for bootstrap_index, draw in enumerate(bootstrap_draws, start=1):
                    for percentile_index, percentile_name in enumerate(percentile_names):
                        draw_rows.append({
                            "realization": int(realization),
                            "population_size": int(sample_size),
                            "time_days": float(time_days),
                            "n_valid": int(len(values)),
                            "bootstrap_index": int(bootstrap_index),
                            "percentile": percentile_name,
                            "bootstrap_value_mag": float(draw[percentile_index]),
                            "sample_estimate_mag": float(quantiles[percentile_index]),
                            "bootstrap_q16_mag": float(boot_low[percentile_index]),
                            "bootstrap_q84_mag": float(boot_high[percentile_index]),
                            "bootstrap_half_width_mag": float(boot_half[percentile_index]),
                            "bootstrap_precision_valid": bool(
                                bootstrap_components[percentile_index]
                            ),
                            "tail_support_valid": bool(
                                tail_component_valid[percentile_index]
                            ),
                            "statistically_valid": bool(
                                percentile_valid[percentile_index]
                            ),
                        })
            rows.append(
                {
                    "population_size": sample_size,
                    "time_days": float(time_days),
                    "n_valid": len(values),
                    "valid_fraction": len(values) / sample_size,
                    "p16": quantiles[0],
                    "p50": quantiles[1],
                    "p84": quantiles[2],
                    "minimum_events_valid": enough_events,
                    "expected_tail_events_p16": expected_tails[0],
                    "expected_tail_events_p50": expected_tails[1],
                    "expected_tail_events_p84": expected_tails[2],
                    "tail_valid_p16": bool(tail_components[0]),
                    "tail_valid_p50": bool(tail_components[1]),
                    "tail_valid_p84": bool(tail_components[2]),
                    "tail_valid": tail_valid,
                    "p16_bootstrap_p16": boot_low[0],
                    "p16_bootstrap_p84": boot_high[0],
                    "p16_bootstrap_half_width": boot_half[0],
                    "p50_bootstrap_p16": boot_low[1],
                    "p50_bootstrap_p84": boot_high[1],
                    "p50_bootstrap_half_width": boot_half[1],
                    "p84_bootstrap_p16": boot_low[2],
                    "p84_bootstrap_p84": boot_high[2],
                    "p84_bootstrap_half_width": boot_half[2],
                    "bootstrap_valid_p16": bool(bootstrap_components[0]),
                    "bootstrap_valid_p50": bool(bootstrap_components[1]),
                    "bootstrap_valid_p84": bool(bootstrap_components[2]),
                    "bootstrap_valid": bootstrap_valid,
                    "p16_valid": bool(percentile_valid[0]),
                    "p50_valid": median_valid,
                    "p84_valid": bool(percentile_valid[2]),
                    "p16_status": component_status[0],
                    "p50_status": component_status[1],
                    "p84_status": component_status[2],
                    "envelope_valid": envelope_valid,
                    "median_valid": median_valid,
                    "failure_scope": failure_scope,
                    "valid": valid,
                    "status": status,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(draw_rows)


# -----------------------------------------------------------------------------
# FIGURE GENERATION AND OUTPUT TABLES
# -----------------------------------------------------------------------------


def time_palette(n_times: int) -> list[object]:
    if n_times <= len(DEFAULT_TIME_COLOURS):
        return DEFAULT_TIME_COLOURS[:n_times]
    return list(plt.get_cmap("viridis")(np.linspace(0.08, 0.92, n_times)))


def plot_pair(
    values: pd.DataFrame,
    pair: tuple[str, str],
    sample_sizes: list[int],
    target_times: np.ndarray,
    output_dir: Path,
    connect: bool,
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
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_distributions(
    draws: pd.DataFrame,
    pair: tuple[str, str],
    realization: int,
    sample_sizes: list[int],
    selected_times: np.ndarray,
    all_times: np.ndarray,
    threshold_mag: float,
    output_dir: Path,
) -> None:
    """Plot selected bootstrap sampling distributions in three panels."""
    percentile_names = ("p16", "p50", "p84")
    percentile_labels = (r"$p_{16}$", r"$p_{50}$", r"$p_{84}$")
    palette = time_palette(len(all_times))
    time_colours = {
        float(time): colour for time, colour in zip(all_times, palette)
    }
    fallback_colours = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, max(1, len(selected_times)))
    )
    for index, time in enumerate(selected_times):
        time_colours.setdefault(float(time), fallback_colours[index])
    line_styles = ["--", "-", ":", "-."]
    size_styles = {
        size: line_styles[index % len(line_styles)]
        for index, size in enumerate(sample_sizes)
    }
    combinations = [
        (sample_size, float(time))
        for sample_size in sample_sizes
        for time in selected_times
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 7.5), squeeze=False)
    axes = axes[0]
    for axis, percentile_name, percentile_label in zip(
        axes, percentile_names, percentile_labels
    ):
        diagnostic_labels = []
        percentile_draws = draws.loc[draws["percentile"].eq(percentile_name)]
        finite_groups = []
        for sample_size, time in combinations:
            group = percentile_draws.loc[
                percentile_draws["population_size"].eq(sample_size)
                & np.isclose(percentile_draws["time_days"], time)
            ]
            values = group["bootstrap_value_mag"].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values):
                finite_groups.append(values)
        if not finite_groups:
            axis.text(
                0.5, 0.5, "No bootstrap draws", transform=axis.transAxes,
                ha="center", va="center",
            )
            continue
        combined = np.concatenate(finite_groups)
        bins = np.histogram_bin_edges(combined, bins="auto")
        if len(bins) < 8:
            low, high = float(combined.min()), float(combined.max())
            padding = max(0.01, 0.08 * max(high - low, 0.01))
            bins = np.linspace(low - padding, high + padding, 16)
        elif len(bins) > 36:
            bins = np.linspace(float(combined.min()), float(combined.max()), 37)

        interval_y = np.linspace(0.95, 0.76, len(combinations))
        for combo_index, ((sample_size, time), y_position) in enumerate(
            zip(combinations, interval_y)
        ):
            group = percentile_draws.loc[
                percentile_draws["population_size"].eq(sample_size)
                & np.isclose(percentile_draws["time_days"], time)
            ]
            values = group["bootstrap_value_mag"].to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            first = group.iloc[0]
            colour = time_colours[float(time)]
            style = size_styles[sample_size]
            half_width = float(first["bootstrap_half_width_mag"])
            bootstrap_ok = bool(first["bootstrap_precision_valid"])
            tail_ok = bool(first["tail_support_valid"])
            label = f"N={sample_size:,}, t={time:g} d"
            density, _ = np.histogram(values, bins=bins, density=True)
            axis.stairs(
                density, bins, color=colour, linestyle=style,
                linewidth=1.8, label=label,
            )
            axis.fill_between(
                bins[:-1], density, step="post", color=colour, alpha=0.055,
            )

            q16 = float(first["bootstrap_q16_mag"])
            q84 = float(first["bootstrap_q84_mag"])
            estimate = float(first["sample_estimate_mag"])
            axis.plot(
                [q16, q84], [y_position, y_position],
                transform=axis.get_xaxis_transform(), color=colour,
                linestyle=style, linewidth=2.2, solid_capstyle="round",
            )
            axis.plot(
                estimate, y_position, transform=axis.get_xaxis_transform(),
                marker="o", markersize=4.2, color=colour, linestyle="none",
            )
            diagnostic_labels.append(
                (
                    colour,
                    style,
                    f"N={sample_size:,}, t={time:g} d: h={half_width:.3f} mag; "
                    f"B {'pass' if bootstrap_ok else 'fail'}; "
                    f"T {'pass' if tail_ok else 'fail'}",
                )
            )

        axis.set_title(percentile_label)
        axis.set_xlabel(
            f"Bootstrap estimate of {percentile_name}({pair[0]}−{pair[1]}) [mag]"
        )
        axis.grid(axis="y", alpha=0.20)
        for label_index, (colour, style, label) in enumerate(diagnostic_labels):
            y_label = -0.27 - 0.085 * label_index
            axis.plot(
                [0.01, 0.06], [y_label, y_label], transform=axis.transAxes,
                color=colour, linestyle=style, linewidth=2.0,
                clip_on=False,
            )
            axis.text(
                0.075, y_label, label, transform=axis.transAxes,
                color=colour, fontsize=7.6, ha="left", va="center",
            )
    axes[0].set_ylabel("Bootstrap density")
    fig.suptitle(
        f"Bootstrap distributions — realization {realization} — "
        f"{pair[0]}−{pair[1]}"
    )
    fig.text(
        0.5, 0.015,
        "Horizontal segments: central bootstrap 68% interval; dot: estimate "
        f"from the original sample. Precision criterion: h68 ≤ {threshold_mag:g} mag. "
        "B: bootstrap; T: tail support.",
        ha="center", va="bottom", fontsize=9, color="0.35",
    )
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.84, bottom=0.42, wspace=0.11
    )
    stem = (
        f"{pair[0]}_minus_{pair[1]}_bootstrap_distributions_"
        f"realization_{realization:02d}"
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = configure_runtime()
    if not args.bootstrap:
        args.plot_bootstrap_distributions = False
    sample_sizes = sorted(set(comma_separated_values(args.sample_sizes, int)))
    target_times = np.asarray(
        comma_separated_values(args.times, float), dtype=float
    )
    diagnostic_sizes = sorted(set(comma_separated_values(
        args.bootstrap_plot_sample_sizes, int
    )))
    diagnostic_times = np.asarray(comma_separated_values(
        args.bootstrap_plot_times, float
    ), dtype=float)
    pairs = [colour_pair(token) for token in args.colour_pairs.split(",")]
    if any(size <= 0 for size in sample_sizes):
        raise ValueError("Sample sizes must be positive")
    if np.any(~np.isfinite(target_times)) or np.any(target_times <= 0):
        raise ValueError("Target times must be finite and positive")
    if args.minimum_valid_events < 1:
        raise ValueError("--minimum-valid-events must be positive")
    if args.minimum_tail_events < 0:
        raise ValueError("--minimum-tail-events cannot be negative")
    if not 0.0 <= args.m5_percentile <= 100.0:
        raise ValueError("--m5-percentile must lie in [0, 100]")
    if args.bootstrap and args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive when bootstrap is enabled")
    if args.bootstrap_threshold_mag < 0:
        raise ValueError("--bootstrap-threshold-mag cannot be negative")
    if args.plot_bootstrap_distributions:
        missing_sizes = sorted(set(diagnostic_sizes) - set(sample_sizes))
        missing_times = [
            float(time) for time in diagnostic_times
            if not np.any(np.isclose(target_times, time))
        ]
        if missing_sizes:
            raise ValueError(
                "Bootstrap diagnostic sizes must also appear in --sample-sizes; "
                f"missing: {missing_sizes}"
            )
        if missing_times:
            raise ValueError(
                "Bootstrap diagnostic times must also appear in --times; "
                f"missing: {missing_times}"
            )

    realizations = read_realizations(args)
    realization_ids = set(
        pd.to_numeric(realizations["realization"], errors="raise").astype(int)
    )
    if (
        args.plot_bootstrap_distributions
        and args.bootstrap_plot_realization not in realization_ids
    ):
        raise ValueError(
            f"Realization {args.bootstrap_plot_realization} is absent; "
            f"available realizations: {sorted(realization_ids)}"
        )
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
    all_bootstrap_draws = []
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
            selection_system = (
                "lsst" if args.photometric_system == "ps1"
                else args.photometric_system
            )
            target_extinction = (
                event_extinction(run_dir, args.photometric_system, pair)
                if args.mw_treatment == "corrected" else None
            )
            selection_extinction = (
                target_extinction
                if selection_system == args.photometric_system
                else event_extinction(run_dir, selection_system, pair)
            ) if args.mw_treatment == "corrected" else None
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
                target_extinction,
                selection_extinction,
                pair_depths,
            )
            values, bootstrap_draws = summarize_nested_prefixes(
                colours,
                sample_sizes,
                target_times,
                args.minimum_valid_events,
                args.minimum_tail_events,
                args.bootstrap,
                args.n_bootstrap,
                args.bootstrap_threshold_mag,
                args.bootstrap_seed,
                int(record.realization),
                pair_index,
                capture_bootstrap=(
                    args.plot_bootstrap_distributions
                    and int(record.realization) == args.bootstrap_plot_realization
                ),
                capture_sample_sizes=set(diagnostic_sizes),
                capture_times=diagnostic_times,
            )
            values.insert(0, "colour_pair", f"{pair[0]}-{pair[1]}")
            values.insert(1, "photometric_system", args.photometric_system)
            values.insert(2, "mw_treatment", args.mw_treatment)
            values.insert(3, "m5_cut", bool(args.m5_cut))
            values.insert(4, "m5_source", m5_source)
            values.insert(5, "m5_percentile", args.m5_percentile if args.m5_cut else np.nan)
            values.insert(6, "m5_band1", pair_depths[pair[0]] if pair_depths else np.nan)
            values.insert(7, "m5_band2", pair_depths[pair[1]] if pair_depths else np.nan)
            values.insert(8, "bootstrap_enabled", bool(args.bootstrap))
            values.insert(9, "n_bootstrap", args.n_bootstrap if args.bootstrap else 0)
            values.insert(10, "bootstrap_threshold_mag", args.bootstrap_threshold_mag)
            values.insert(11, "minimum_tail_events", args.minimum_tail_events)
            values.insert(12, "realization", int(record.realization))
            if hasattr(record, "intrinsic_seed"):
                values.insert(13, "intrinsic_seed", int(record.intrinsic_seed))
            values["run_dir"] = str(run_dir)
            all_rows.append(values)
            if not bootstrap_draws.empty:
                bootstrap_draws.insert(0, "colour_pair", f"{pair[0]}-{pair[1]}")
                bootstrap_draws.insert(1, "photometric_system", args.photometric_system)
                bootstrap_draws.insert(2, "mw_treatment", args.mw_treatment)
                bootstrap_draws.insert(3, "m5_cut", bool(args.m5_cut))
                bootstrap_draws.insert(
                    4, "m5_percentile",
                    args.m5_percentile if args.m5_cut else np.nan,
                )
                bootstrap_draws.insert(
                    5, "bootstrap_threshold_mag", args.bootstrap_threshold_mag
                )
                bootstrap_draws["run_dir"] = str(run_dir)
                all_bootstrap_draws.append(bootstrap_draws)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(output_dir / "colour_envelope_realization_values.csv", index=False)
    for pair in pairs:
        pair_name = f"{pair[0]}-{pair[1]}"
        plot_pair(
            result.loc[result["colour_pair"].eq(pair_name)],
            pair,
            sample_sizes,
            target_times,
            output_dir,
            connect=not args.no_connect,
        )
    if args.plot_bootstrap_distributions:
        if not all_bootstrap_draws:
            raise RuntimeError("No bootstrap draws were captured for the diagnostic")
        diagnostic = pd.concat(all_bootstrap_draws, ignore_index=True)
        diagnostic.to_csv(
            output_dir / "bootstrap_percentile_draws.csv.gz",
            index=False,
            compression="gzip",
        )
        for pair in pairs:
            pair_name = f"{pair[0]}-{pair[1]}"
            plot_bootstrap_distributions(
                diagnostic.loc[diagnostic["colour_pair"].eq(pair_name)],
                pair,
                args.bootstrap_plot_realization,
                diagnostic_sizes,
                diagnostic_times,
                target_times,
                args.bootstrap_threshold_mag,
                output_dir,
            )
    print(f"Saved comparison outputs: {output_dir}")


if __name__ == "__main__":
    main()
