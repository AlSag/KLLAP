#!/usr/bin/env python3
"""Fast, single-load convergence analysis for KNe colour envelopes.

Unlike ``analyze_envelope_population_convergence.py``, this program reads the
saved synthetic light curves once and evaluates all nested sample sizes from
in-memory arrays.  The default analysis uses 100% of every selected subset and
separates fixed-percentile value convergence on a fixed time window from
convergence of the last statistically valid temporal bin.  Legacy adaptive-
bound and last-bin-only modes remain available.  Bootstrap validation is
delegated to ``build_color_envelopes.py`` so the scientific validity
definitions stay shared with the envelope builder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import kne_convergence_common as convergence
import build_color_envelopes as builder
from dynamic_color_cut_common import (
    assign_event_splits,
    canonical_event_id,
    interpolate_band,
    parse_floats,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast nested-subsample convergence analysis (single Parquet read).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--color-pair", default="g-r")
    p.add_argument("--photometric-system", choices=["lsst", "ps1", "ztf"], default="lsst")
    p.add_argument("--stage", choices=convergence.STAGES, default="m5_with_mw")
    p.add_argument("--sample-sizes", default="1000,3000,10000,30000,50000,75000,100000,150000,200000")
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--subsample-seed", type=int, default=20261002)
    p.add_argument("--builder-seed", type=int, default=12345)
    p.add_argument("--t-min", type=float, default=0.4)
    p.add_argument("--t-max", type=float, default=16.0)
    p.add_argument("--bin-width", type=float, default=0.4)
    p.add_argument(
        "--n-boot", type=int, default=200,
        help=("Bootstrap resamples per temporal bin and percentile. Two "
              "hundred is the recommended speed/precision compromise."),
    )
    p.add_argument("--minimum-events", type=int, default=30)
    p.add_argument("--minimum-tail-events", type=float, default=20.0)
    p.add_argument("--bootstrap-threshold-mag", type=float, default=0.1)
    p.add_argument(
        "--train-fraction", type=float, default=0.70,
        help=("Used only by legacy-colour and last-edge. The fixed 'colour' "
              "analysis always uses 100%% of the selected subset."),
    )
    p.add_argument(
        "--validation-fraction", type=float, default=0.15,
        help=("Used only by legacy-colour and last-edge. The fixed 'colour' "
              "analysis performs no train/validation split."),
    )
    p.add_argument("--tail-percentiles")
    p.add_argument("--m5-scenario", choices=convergence.M5_SCENARIOS, default="median_p50")
    p.add_argument("--threshold-scope", choices=["common_global", "latitude_conditioned"], default="common_global")
    p.add_argument("--m5-quantile-table", type=Path)
    p.add_argument("--ztf-depth-source", choices=["manual", "table"], default="manual")
    p.add_argument("--ztf-m5-g", type=float)
    p.add_argument("--ztf-m5-r", type=float)
    p.add_argument("--ztf-m5-i", type=float)
    p.add_argument("--ztf-depth-quantile-table", type=Path)
    p.add_argument("--tolerance-days", type=float, default=0.4)
    p.add_argument("--required-fraction", type=float, default=0.9)
    p.add_argument(
        "--stability-metric",
        choices=["colour", "legacy-colour", "last-edge"],
        default="colour",
        help=(
            "'colour' runs the fixed multi-percentile shape/extent analysis; "
            "'legacy-colour' restores the previous adaptive three-bound test; "
            "'last-edge' uses the original last-bin-only analysis."
        ),
    )
    p.add_argument(
        "--shape-t-max",
        type=float,
        default=4.0,
        help=(
            "Upper edge in days of the fixed window used only for percentile-"
            "value (shape) convergence. Temporal-extent convergence is "
            "calculated separately up to --t-max."
        ),
    )
    p.add_argument(
        "--shape-horizon-mode",
        choices=["fixed", "reference-minus-tolerance"],
        default="fixed",
        help=(
            "Use --shape-t-max, or derive a separate fixed comparison "
            "horizon for each percentile family from the reference's last "
            "valid edge minus --tolerance-days. The latter tests percentile "
            "values over the full interval that a temporally compatible "
            "candidate is required to reproduce."
        ),
    )
    p.add_argument(
        "--tolerance-mode",
        choices=["absolute", "relative"],
        default="absolute",
        help=(
            "Metric used to decide convergence. 'absolute' compares colour "
            "differences directly in magnitudes; 'relative' reproduces the "
            "older fractional criterion."
        ),
    )
    p.add_argument(
        "--absolute-tolerance-mag",
        type=float,
        default=0.10,
        help="Maximum absolute colour difference in mag when --tolerance-mode=absolute.",
    )
    p.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.10,
        help="Fractional tolerance used only when --tolerance-mode=relative.",
    )
    p.add_argument("--bound-mode", choices=["adaptive", "reference-percentiles"], default="adaptive",
                   help="Compare each sample's actual adaptive bounds, or keep the reference's percentile pair in each bin.")
    p.add_argument("--allow-degenerate-envelope", action="store_true",
                   help="Count p50-p50 as a valid envelope in colour mode (excluded by default).")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fit-exponential", action=argparse.BooleanOptionalAction, default=True)
    return p


def validate(a: argparse.Namespace) -> tuple[Path, list[int], tuple[str, str], np.ndarray, np.ndarray]:
    root = convergence.resolve_run_root(a.run_dir)
    sizes = convergence.parse_sample_sizes(a.sample_sizes)
    bands = convergence.split_colour_pair(a.color_pair)
    if a.photometric_system == "ztf" and not set(bands) <= {"g", "r", "i"}:
        raise ValueError("ZTF supports g, r and i only")
    if sizes[-1] > convergence.count_summary_events(root):
        raise ValueError("Largest sample size exceeds the events in summary.csv")
    if a.repetitions < 1 or a.n_boot < 1:
        raise ValueError("repetitions and n-boot must be positive")
    if a.bin_width <= 0 or a.t_max <= a.t_min or a.tolerance_days < 0:
        raise ValueError("Invalid time grid or tolerance")
    if not a.t_min < a.shape_t_max <= a.t_max:
        raise ValueError("--shape-t-max must lie in (t-min, t-max]")
    shape_steps = (a.shape_t_max - a.t_min) / a.bin_width
    if not np.isclose(shape_steps, round(shape_steps), atol=1e-8):
        raise ValueError("(shape-t-max - t-min) must be an integer multiple of bin-width")
    if not 0 < a.required_fraction <= 1:
        raise ValueError("required-fraction must lie in (0, 1]")
    edges = np.arange(a.t_min, a.t_max + 0.5 * a.bin_width, a.bin_width)
    if not np.isclose(edges[-1], a.t_max, atol=1e-8):
        raise ValueError("(t-max - t-min) must be an integer multiple of bin-width")
    return root, sizes, bands, edges, (edges[:-1] + edges[1:]) / 2


def make_builder_args(a: argparse.Namespace, root: Path) -> argparse.Namespace:
    """Create the builder's default metadata-selection namespace."""
    args = builder.build_parser().parse_args([
        "--run-dir", str(root),
        "--color-pair", a.color_pair,
        "--photometric-system", a.photometric_system,
        "--t-min", str(a.t_min), "--t-max", str(a.t_max),
        "--bin-width", str(a.bin_width), "--seed", str(a.builder_seed),
        "--max-events", "0",
    ])
    args.threshold_scope = a.threshold_scope
    args.m5_scenario = a.m5_scenario
    args.m5_quantile_table = a.m5_quantile_table
    # The convergence CLI intentionally uses the all-latitude parent sample,
    # matching the original script's defaults.
    return args


def get_depths(a: argparse.Namespace, bargs: argparse.Namespace, root: Path, bands: tuple[str, str]) -> dict[str, float]:
    if a.photometric_system == "ztf":
        if a.ztf_depth_source == "manual":
            manual = {"g": a.ztf_m5_g, "r": a.ztf_m5_r, "i": a.ztf_m5_i}
            missing = [band for band in set(bands) if manual[band] is None]
            if missing:
                raise ValueError("Missing manual ZTF depth(s): " + ", ".join(missing))
            return {band: float(manual[band]) for band in bands}
        if a.ztf_depth_quantile_table is None:
            raise ValueError("Table mode needs --ztf-depth-quantile-table")
        depths, _ = builder.load_ztf_depth_thresholds(
            directory=None, quantile_table=a.ztf_depth_quantile_table,
            scenario=a.m5_scenario, bands=bands, n_samples=0, seed=0,
            depth_source="table", manual_thresholds=None,
        )
        return depths
    table = builder.discover_m5_table(root, a.m5_quantile_table)
    return builder.load_m5(table, a.threshold_scope, a.m5_scenario, bands)


def load_population_once(a: argparse.Namespace, root: Path, bands: tuple[str, str], centres: np.ndarray, sizes: list[int]):
    """Read relevant event photometry once and return compact numerical arrays."""
    source = builder.SyntheticPopulationSource(root)
    if source.kind == "missing":
        raise FileNotFoundError(f"No saved synthetic population under {root}")
    metadata = builder.prepare_metadata(make_builder_args(a, root), source.available_event_ids())
    maximum = sizes[-1]
    # Form the union of event IDs that can occur in any requested repetition.
    ids_needed: set[str] = set()
    rank_maps: dict[int, dict[str, int]] = {}
    for rep in range(a.repetitions):
        seed = a.subsample_seed + rep
        permutation = np.random.default_rng(seed).permutation(len(metadata))
        selected = permutation[:min(maximum, len(metadata))]
        selected_ids = metadata.iloc[selected].event_id.astype(str).tolist()
        rank_maps[seed] = {event_id: rank for rank, event_id in enumerate(selected_ids)}
        ids_needed.update(selected_ids)
    metadata = metadata.loc[metadata.event_id.astype(str).isin(ids_needed)].copy().reset_index(drop=True)
    # Preserve each permutation's original rank.  Re-drawing from the union
    # would create different, non-nested subsets when repetitions overlap.
    for seed, ranks in rank_maps.items():
        metadata[f"_subsample_rank_{seed}"] = (
            metadata.event_id.astype(str).map(ranks).fillna(np.inf)
        )
    event_ids = metadata.event_id.astype(str).tolist()
    pos = {canonical_event_id(eid): i for i, eid in enumerate(event_ids)}
    n, nt = len(metadata), len(centres)
    band1, band2 = bands
    system = a.photometric_system
    selection_system = "lsst" if system == "ps1" else system
    systems = tuple(dict.fromkeys((system, selection_system)))
    arrays = {
        "colour_intrinsic": np.full((n, nt), np.nan, np.float32),
        "colour_extincted": np.full((n, nt), np.nan, np.float32),
        "mag_sel_ext_1": np.full((n, nt), np.nan, np.float32),
        "mag_sel_ext_2": np.full((n, nt), np.nan, np.float32),
        "mag_sel_int_1": np.full((n, nt), np.nan, np.float32),
        "mag_sel_int_2": np.full((n, nt), np.nan, np.float32),
    }
    meta_by_id = metadata.set_index("event_id", drop=False)
    seen: set[str] = set()
    interval = max(500, n // 20)
    print(f"Reading {n:,} events once for all sample sizes ({source.kind})")
    for eid, curve in source.iter_events(event_ids, bands, systems):
        eid = canonical_event_id(eid)
        if eid in seen:
            raise RuntimeError(f"Duplicate saved event rows for {eid}")
        seen.add(eid)
        i = pos[eid]
        row = meta_by_id.loc[eid]
        curve = curve.copy()
        curve["t_days"] = pd.to_numeric(curve.t_days, errors="coerce")
        curve["band"] = curve.band.astype(str).str.strip().str.lower()
        curve["magnitude"] = pd.to_numeric(curve.mag, errors="coerce")
        if "photometric_system" not in curve:
            curve["photometric_system"] = "lsst"
        curve["photometric_system"] = curve.photometric_system.astype(str).str.lower()
        curve = curve.loc[
            curve.band.isin(bands) & curve.photometric_system.isin(systems)
            & np.isfinite(curve.t_days) & (curve.t_days > 0)
            & np.isfinite(curve.magnitude)
        ]
        target = curve.loc[curve.photometric_system == system]
        select = curve.loc[curve.photometric_system == selection_system]
        m1 = interpolate_band(target, band1, centres)
        m2 = interpolate_band(target, band2, centres)
        s1 = interpolate_band(select, band1, centres)
        s2 = interpolate_band(select, band2, centres)
        a1 = builder.event_extinction_mag(row, system, band1)
        a2 = builder.event_extinction_mag(row, system, band2)
        sa1 = builder.event_extinction_mag(row, selection_system, band1)
        sa2 = builder.event_extinction_mag(row, selection_system, band2)
        arrays["colour_extincted"][i] = m1 - m2
        arrays["colour_intrinsic"][i] = (m1 - a1) - (m2 - a2)
        arrays["mag_sel_ext_1"][i] = s1
        arrays["mag_sel_ext_2"][i] = s2
        arrays["mag_sel_int_1"][i] = s1 - sa1
        arrays["mag_sel_int_2"][i] = s2 - sa2
        if (i + 1) % interval == 0:
            print(f"  loaded rows for {i + 1:,}/{n:,} events")
    print(f"Finished loading: {len(seen):,}/{n:,} events with saved rows")
    return metadata, event_ids, arrays


def select_loaded_metadata_subset(metadata: pd.DataFrame, maximum_events: int,
                                  subsample_seed: int) -> pd.DataFrame:
    """Recover the exact nested subset recorded by ``load_population_once``."""
    rank_column = f"_subsample_rank_{subsample_seed}"
    if rank_column not in metadata:
        # Compatibility fallback for callers supplying a conventional table.
        return builder.select_metadata_subset(
            metadata, maximum_events, subsample_seed
        )
    subset = metadata.loc[metadata[rank_column] < int(maximum_events)].copy()
    expected = min(
        int(maximum_events), int(np.isfinite(metadata[rank_column]).sum())
    )
    if len(subset) != expected:
        # Keep the explicit error rather than silently analysing the wrong
        # population.
        raise RuntimeError(
            f"Stored subset rank {rank_column} produced {len(subset)} events; "
            f"expected {expected}"
        )
    return subset.sort_values(
        "event_id", key=lambda series: pd.to_numeric(series, errors="coerce")
    ).reset_index(drop=True)


def last_consecutive_edge(a: argparse.Namespace, sample: pd.DataFrame, arrays: dict[str, np.ndarray], indices: np.ndarray, depths: dict[str, float], bands: tuple[str, str], edges: np.ndarray, centres: np.ndarray) -> tuple[float, int, float, float, float]:
    ids = sample.event_id.astype(str).tolist()
    split = assign_event_splits(ids, a.train_fraction, a.validation_fraction, a.builder_seed)
    train = split.split.to_numpy() == "train"
    if a.stage in builder.MW_EXTINCTED_STAGES:
        colours = arrays["colour_extincted"][indices]
    else:
        colours = arrays["colour_intrinsic"][indices]
    available = np.isfinite(arrays["colour_intrinsic"][indices]) & np.isfinite(arrays["colour_extincted"][indices])
    if a.stage in builder.M5_STAGES:
        if a.stage == "m5_no_mw":
            one, two = "mag_sel_int_1", "mag_sel_int_2"
        else:
            one, two = "mag_sel_ext_1", "mag_sel_ext_2"
        available &= np.isfinite(arrays[one][indices]) & np.isfinite(arrays[two][indices])
        available &= (arrays[one][indices] <= depths[bands[0]]) & (arrays[two][indices] <= depths[bands[1]])
    tails = parse_floats(a.tail_percentiles or builder.DEFAULT_TAIL_GRID)
    for fallback in (25, 30, 35, 40, 42, 44, 45, 46, 47, 48, 49, 50):
        if fallback not in tails:
            tails.append(float(fallback))
    seed_offset = builder.STAGE_BOOTSTRAP_SEED_OFFSETS[a.stage]
    rng = np.random.default_rng(a.builder_seed + 1000 * seed_offset)
    count = 0
    low_q = high_q = np.nan
    n_train_last = np.nan
    for b, _ in enumerate(centres):
        values = colours[train & available[:, b], b].astype(float)
        if len(values) < a.minimum_events:
            break
        _, selected = builder.percentile_diagnostics(
            values, tails, a.n_boot, a.bootstrap_threshold_mag,
            a.minimum_tail_events, rng,
        )
        if selected is None:
            break
        count += 1
        n_train_last = float(len(values))
        low_q = float(selected["low_percentile"])
        high_q = float(selected["high_percentile"])
    if count == 0:
        return np.nan, 0, np.nan, np.nan, np.nan
    return float(edges[count]), count, low_q, high_q, n_train_last


def configuration(a: argparse.Namespace, root: Path, sizes: list[int]) -> dict[str, object]:
    return {
        "method": "single_load_in_memory_exact_builder_bootstrap_early_stop",
        "run_dir": str(root), "color_pair": a.color_pair,
        "photometric_system": a.photometric_system, "stage": a.stage,
        "sample_sizes": sizes, "repetitions": a.repetitions,
        "subsample_seed": a.subsample_seed, "builder_seed": a.builder_seed,
        "t_min": a.t_min, "t_max": a.t_max, "bin_width": a.bin_width,
        "n_boot": a.n_boot, "minimum_events": a.minimum_events,
        "minimum_tail_events": a.minimum_tail_events,
        "bootstrap_threshold_mag": a.bootstrap_threshold_mag,
        "train_fraction": a.train_fraction,
        "validation_fraction": a.validation_fraction,
        "tail_percentiles": a.tail_percentiles,
        "m5_scenario": a.m5_scenario, "threshold_scope": a.threshold_scope,
        "tolerance_days": a.tolerance_days,
        "required_fraction": a.required_fraction,
    }


def main() -> None:
    a = parser().parse_args()
    root, sizes, bands, edges, centres = validate(a)
    if a.stability_metric == "colour":
        from envelope_multiquantile_convergence import run_multi_percentile_convergence
        run_multi_percentile_convergence(a, root, sizes, bands, edges, centres)
        return
    if a.stability_metric == "legacy-colour":
        from envelope_colour_convergence import run_colour_convergence
        run_colour_convergence(a, root, sizes, bands, edges, centres)
        return
    output = (a.output_dir.expanduser().resolve() if a.output_dir else root / "analysis" / f"convergence_fast_{a.color_pair.replace('-', '_')}_{a.photometric_system}_{a.stage}")
    output.mkdir(parents=True, exist_ok=True)
    conf = configuration(a, root, sizes)
    conf_path = output / "fast_configuration.json"
    if a.resume and conf_path.exists() and json.loads(conf_path.read_text()) != conf:
        raise ValueError(f"Output directory has a different fast-run configuration: {output}")
    conf_path.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")

    bargs = make_builder_args(a, root)
    depths = get_depths(a, bargs, root, bands)
    metadata, event_ids, arrays = load_population_once(a, root, bands, centres, sizes)
    position = {event_id: i for i, event_id in enumerate(event_ids)}
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    total = len(sizes) * a.repetitions
    counter = 0
    for rep in range(a.repetitions):
        subset_seed = a.subsample_seed + rep
        for size in sizes:
            counter += 1
            checkpoint = checkpoint_dir / f"rep_{rep:03d}_N_{size:09d}.json"
            if a.resume and checkpoint.is_file():
                row = json.loads(checkpoint.read_text(encoding="utf-8"))
                records.append(row)
                print(f"[{counter}/{total}] reusing repetition={rep + 1}, N={size:,}")
                continue
            print(f"[{counter}/{total}] repetition={rep + 1}/{a.repetitions}, N={size:,}")
            sample = select_loaded_metadata_subset(metadata, size, subset_seed)
            indices = np.asarray([position[eid] for eid in sample.event_id.astype(str)], dtype=int)
            edge, n_bins, low_q, high_q, n_train = last_consecutive_edge(
                a, sample, arrays, indices, depths, bands, edges, centres
            )
            row = {
                "repetition": rep, "subsample_seed": subset_seed,
                "N_requested": size, "N_input_events": len(sample),
                "N_train_last_bin": n_train, "stage": a.stage,
                "last_valid_time_days": edge,
                "N_consecutive_valid_bins": n_bins,
                "last_low_percentile": low_q, "last_high_percentile": high_q,
                "build_output": "in_memory_fast_path",
            }
            checkpoint.write_text(json.dumps(row, allow_nan=True) + "\n", encoding="utf-8")
            records.append(row)

    raw = pd.DataFrame(records)
    raw, summary, reference, threshold = convergence.aggregate_results(
        raw, a.tolerance_days, a.required_fraction
    )
    raw_path = output / "envelope_population_convergence_raw.csv"
    summary_path = output / "envelope_population_convergence_summary.csv"
    plot_path = output / "envelope_population_convergence.png"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    convergence.save_plot(raw, summary, reference, threshold, a, plot_path)
    result = dict(conf)
    result.update({
        "last_bin_definition": "last initial consecutive valid envelope-bin upper edge",
        "reference": "last-bin time at largest tested population, paired by repetition",
        "stable_population_threshold": threshold,
        "largest_N_reference_median_days": reference,
        "stability_rule": "first tested N where median and required fraction of repetitions are within tolerance, and all larger tested N also pass",
    })
    (output / "envelope_population_convergence_configuration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("Reference last edge:", reference, "days")
    print("Stable population threshold:", f"{threshold:,} KNe" if threshold is not None else "not determined")
    print("Plot:", plot_path)
    print("Summary:", summary_path)
    print("Raw repetitions:", raw_path)


if __name__ == "__main__":
    main()
