"""Saved-population discovery and metadata selection for KNe analyses.

This module owns storage access only. It does not calculate colours, apply
depth thresholds, bootstrap percentiles, or create figures. Keeping it neutral
prevents analysis commands from importing the envelope-builder command merely
to read the same generated population.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from dynamic_color_cut_common import (
    canonical_event_id,
    discover_synthetic_files,
    event_id_from_path,
    load_summary,
    run_root,
)
from kne_envelope_core import select_metadata_subset


LATITUDE_COLUMNS = (
    "galactic_b_deg",
    "galactic_latitude_deg",
    "b_gal_deg",
    "b_deg",
)


def population_latitude_label(args: argparse.Namespace) -> str:
    """Return a human-readable KNe population latitude selection."""
    mode = args.population_latitude_selection
    if mode == "all":
        return "all generated Galactic latitudes (no latitude cut)"
    operator = ">=" if mode == "high" else "<"
    return f"|b| {operator} {args.latitude_threshold_deg:g} deg"


class SyntheticPopulationSource:
    """Read saved curves from preferred Parquet shards or legacy CSV files."""

    def __init__(self, run_dir: Path):
        self.root = run_root(run_dir)
        self.parquet_paths = sorted(
            (self.root / "parquet").glob("synthetic_part_*.parquet")
        )
        self.csv_paths = discover_synthetic_files(run_dir)
        self.csv_by_event = {
            event_id_from_path(path): path for path in self.csv_paths
        }
        if self.parquet_paths:
            self.kind = "parquet_shards"
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Reading the synthetic Parquet population requires pyarrow. "
                    "Install it with `python -m pip install pyarrow`."
                ) from exc
        elif self.csv_paths:
            self.kind = "legacy_csv_files"
        else:
            self.kind = "missing"

    def available_event_ids(self) -> set[str]:
        if self.kind == "legacy_csv_files":
            return set(self.csv_by_event)
        if self.kind != "parquet_shards":
            return set()
        summary = load_summary(self.root)
        if not summary.empty and "synthetic_parquet_saved" in summary:
            saved = (
                summary["synthetic_parquet_saved"]
                .astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            )
            return set(summary.loc[saved, "event_id"].map(canonical_event_id))
        available: set[str] = set()
        for path in self.parquet_paths:
            identifiers = pd.read_parquet(path, columns=["event_id"])
            available.update(
                identifiers["event_id"].map(canonical_event_id).unique()
            )
        return available

    def iter_events(
        self,
        requested_event_ids: list[str],
        bands: tuple[str, str],
        photometric_systems: tuple[str, ...] = ("lsst",),
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        requested = {canonical_event_id(value) for value in requested_event_ids}
        if self.kind == "legacy_csv_files":
            for event_id in requested_event_ids:
                canonical = canonical_event_id(event_id)
                lightcurve = pd.read_csv(self.csv_by_event[canonical])
                if "photometric_system" not in lightcurve:
                    lightcurve["photometric_system"] = "lsst"
                yield canonical, lightcurve
            return

        for path in self.parquet_paths:
            import pyarrow.parquet as pq

            schema_names = set(pq.read_schema(path).names)
            has_system = "photometric_system" in schema_names
            columns = ["event_id", "t_days", "band", "mag"]
            filters = [("band", "in", list(bands))]
            if has_system:
                columns.append("photometric_system")
                filters.append(
                    ("photometric_system", "in", list(photometric_systems))
                )
            elif any(system != "lsst" for system in photometric_systems):
                raise ValueError(
                    f"{path} has no photometric_system column. It is a legacy "
                    "LSST-only shard and cannot build the requested non-LSST envelope."
                )
            shard = pd.read_parquet(path, columns=columns, filters=filters)
            if shard.empty:
                continue
            if not has_system:
                shard["photometric_system"] = "lsst"
            shard["event_id"] = shard["event_id"].map(canonical_event_id)
            shard = shard.loc[shard["event_id"].isin(requested)]
            for event_id, lightcurve in shard.groupby("event_id", sort=False):
                yield canonical_event_id(event_id), lightcurve.drop(
                    columns=["event_id"]
                ).reset_index(drop=True)


def prepare_metadata(
    args: argparse.Namespace,
    available_event_ids: set[str],
) -> pd.DataFrame:
    """Select usable metadata without interpreting photometric measurements."""
    summary = load_summary(args.run_dir)
    if summary.empty:
        raise FileNotFoundError(f"Missing usable {run_root(args.run_dir) / 'summary.csv'}")
    required_columns = (
        "ebv_mw", "inclination_EM", "luminosity_distance", "redshift",
    )
    optional_physical_columns = (
        "log10_mej_dyn", "log10_mej_wind", "v_ej_dyn", "Ye_dyn",
        "v_ej_wind", "Ye_wind", "phi", "log10_mej", "log10_vej",
        "beta", "log10_kappa_r",
    )
    missing = [column for column in required_columns if column not in summary]
    if missing:
        raise ValueError(f"summary.csv is missing {missing}")
    metadata = summary.drop_duplicates("event_id").copy()

    latitude_column = next(
        (column for column in LATITUDE_COLUMNS if column in metadata), None
    )
    latitude_mode = args.population_latitude_selection
    before_latitude = len(metadata)
    if latitude_mode == "all":
        print(
            "No Galactic-latitude cut applied to the KNe population: "
            f"{len(metadata):,}/{before_latitude:,} events retained"
        )
    elif latitude_column is not None:
        latitude = pd.to_numeric(metadata[latitude_column], errors="coerce")
        finite = latitude.notna()
        if latitude_mode == "high":
            keep = finite & (latitude.abs() >= args.latitude_threshold_deg)
        else:
            keep = finite & (latitude.abs() < args.latitude_threshold_deg)
        metadata = metadata.loc[keep].copy()
        print(
            f"Applied {population_latitude_label(args)} using {latitude_column}: "
            f"{len(metadata):,}/{before_latitude:,} events retained"
        )
    else:
        selection_assumed = bool(
            args.assume_latitude_selection_already_applied
            or (args.assume_high_latitude_run and latitude_mode == "high")
        )
        if not selection_assumed:
            raise ValueError(
                "The requested population selection "
                f"({population_latitude_label(args)}) cannot be applied because "
                "summary.csv has no Galactic-latitude column. Supply such a "
                "column, choose --population-latitude-selection all, or use "
                "--assume-latitude-selection-already-applied only for an input "
                "run already generated with this exact selection."
            )
        print(
            "No Galactic-latitude column: accepting the explicit confirmation "
            "that the input run already satisfies "
            f"{population_latitude_label(args)}."
        )

    for column in (*required_columns, *optional_physical_columns):
        if column not in metadata:
            metadata[column] = np.nan
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    metadata = metadata.dropna(subset=list(required_columns))
    metadata["event_id"] = metadata["event_id"].map(canonical_event_id)
    before_files = len(metadata)
    metadata = metadata.loc[metadata["event_id"].isin(available_event_ids)].copy()
    print(
        f"Matched saved synthetic events: {len(metadata):,}/{before_files:,} "
        "selected summary events"
    )
    metadata = metadata.sort_values(
        "event_id", key=lambda values: pd.to_numeric(values, errors="coerce")
    ).reset_index(drop=True)
    metadata = select_metadata_subset(
        metadata, args.max_events, args.event_subsample_seed
    )
    if metadata.empty:
        raise ValueError(
            "No events satisfying the requested Galactic-latitude selection "
            "have a saved synthetic light curve"
        )
    return metadata
