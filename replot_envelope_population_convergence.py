#!/usr/bin/env python3
"""Rebuild convergence plots/tables from saved profiles, without bootstrap.

The expensive light-curve read and event bootstrap are not repeated.  This
program uses ``percentile_profiles.csv`` and the saved reference profiles to
re-evaluate the percentile-value horizon, then restyles all plots and tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import envelope_multiquantile_convergence as multi


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replot saved fixed-percentile convergence results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument(
        "--shape-horizon-mode",
        choices=["fixed", "reference-minus-tolerance"],
        default="reference-minus-tolerance",
    )
    p.add_argument("--shape-t-max", type=float)
    p.add_argument("--absolute-tolerance-mag", type=float)
    p.add_argument("--tolerance-days", type=float)
    p.add_argument("--required-fraction", type=float)
    return p


def _configuration(source: Path, cli: argparse.Namespace) -> SimpleNamespace:
    candidates = (
        source / "multi_percentile_results.json",
        source / "multi_percentile_configuration.json",
    )
    config = {}
    for path in candidates:
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8"))
            break
    required = {
        "t_min": .4,
        "shape_t_max": 4.,
        "absolute_tolerance_mag": .10,
        "relative_tolerance": .10,
        "tolerance_mode": "absolute",
        "tolerance_days": .4,
        "required_fraction": .9,
        "subsample_seed": 20261002,
    }
    for key, default in required.items():
        config.setdefault(key, default)
    for key in (
        "shape_t_max", "absolute_tolerance_mag", "tolerance_days",
        "required_fraction",
    ):
        override = getattr(cli, key)
        if override is not None:
            config[key] = override
    config["shape_horizon_mode"] = cli.shape_horizon_mode
    return SimpleNamespace(**config)


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, float_precision="round_trip")


def rebuild(source: Path, output: Path, a: SimpleNamespace) -> None:
    profiles = _load(source / "percentile_profiles.csv")
    references = _load(source / "percentile_reference_profiles.csv")
    extent_raw_path = source / "temporal_extent_convergence_raw.csv"

    comparisons = []
    records = []
    extent_records = []
    sizes = sorted(profiles.N_requested.astype(int).unique())
    repetitions = sorted(profiles.repetition.astype(int).unique())

    for repetition in repetitions:
        reference = references.loc[
            references.repetition.eq(repetition)
        ].copy()
        reference = reference.drop(columns="repetition")
        for size in sizes:
            profile = profiles.loc[
                profiles.repetition.eq(repetition)
                & profiles.N_requested.eq(size)
            ].copy()
            profile = profile.drop(columns=["repetition", "N_requested"])
            seed = int(a.subsample_seed) + int(repetition)
            for group_id, members in multi.ANALYSIS_GROUPS.items():
                horizon = multi.shape_horizon(reference, members, a)
                comparison = multi.compare_shape_group(
                    profile, reference, group_id, members, horizon,
                    tolerance_mode=a.tolerance_mode,
                    relative_tolerance=a.relative_tolerance,
                    absolute_tolerance_mag=a.absolute_tolerance_mag,
                )
                comparison.insert(0, "N_requested", int(size))
                comparison.insert(0, "repetition", int(repetition))
                comparisons.append(comparison)
                records.append(multi.shape_repetition_record(
                    comparison, repetition, size, seed,
                    tolerance_mode=a.tolerance_mode,
                ))
                if not extent_raw_path.is_file():
                    extent_records.append(multi.extent_repetition_record(
                        profile, reference, group_id, members, repetition,
                        size, seed, a.tolerance_days,
                    ))

    shape_per_bin = pd.concat(comparisons, ignore_index=True)
    shape_raw = pd.DataFrame(records).sort_values(
        ["group_id", "repetition", "N_requested"]
    )
    shape_summary, shape_thresholds = multi.aggregate_shape(
        shape_raw, a.required_fraction
    )
    extent_raw = (
        _load(extent_raw_path) if extent_raw_path.is_file()
        else pd.DataFrame(extent_records)
    )
    extent_summary, extent_thresholds = multi.aggregate_extent(
        extent_raw, a.required_fraction
    )
    thresholds = multi.threshold_table(
        shape_thresholds, extent_thresholds, max(sizes)
    )

    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "percentile_value_convergence_per_bin.csv": shape_per_bin,
        "percentile_value_convergence_raw.csv": shape_raw,
        "percentile_value_convergence_summary.csv": shape_summary,
        "temporal_extent_convergence_raw.csv": extent_raw,
        "temporal_extent_convergence_summary.csv": extent_summary,
        "convergence_thresholds.csv": thresholds,
    }
    for name, frame in frames.items():
        multi.atomic_csv(frame, output / name)

    multi.save_shape_plot(
        shape_raw, shape_summary, a,
        output / "percentile_value_convergence.png",
    )
    multi.save_extent_plot(
        extent_summary, a,
        output / "temporal_extent_convergence.png",
    )
    multi.save_shape_values_table(
        shape_summary,
        output / "percentile_value_convergence_values_table.png",
    )
    multi.save_extent_values_table(
        extent_summary, a,
        output / "temporal_extent_values_table.png",
    )
    settings = {
        "source_analysis_dir": str(source),
        "shape_horizon_mode": a.shape_horizon_mode,
        "fixed_shape_t_max_days": a.shape_t_max,
        "absolute_tolerance_mag": a.absolute_tolerance_mag,
        "tolerance_days": a.tolerance_days,
        "required_fraction": a.required_fraction,
        "bootstrap_recomputed": False,
    }
    (output / "replot_configuration.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    print("Replot output:", output)
    print(thresholds.to_string(index=False))


def main() -> None:
    cli = parser().parse_args()
    source = cli.analysis_dir.expanduser().resolve()
    a = _configuration(source, cli)
    suffix = (
        "dynamic_reference_horizon"
        if cli.shape_horizon_mode == "reference-minus-tolerance"
        else f"fixed_{a.shape_t_max:g}d"
    )
    output = (
        cli.output_dir.expanduser().resolve()
        if cli.output_dir else source / f"replot_{suffix}"
    )
    rebuild(source, output, a)


if __name__ == "__main__":
    main()
