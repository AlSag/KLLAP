"""Interpolate saved KNe photometry for four-stage envelope construction.

The functions here transform already generated light curves into magnitude
matrices at fixed bin centres and retain the native colour tracks used for
figures. They never extrapolate beyond the saved support and do not apply m5
thresholds or bootstrap statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kllap.colour.common import canonical_event_id, interpolate_band
from kllap.envelope.core import MW_EXTINCTED_STAGES, STAGES, event_extinction_mag
from kllap.population_io import SyntheticPopulationSource


@dataclass
class InterpolatedPopulation:
    intrinsic_band1: np.ndarray
    intrinsic_band2: np.ndarray
    extincted_band1: np.ndarray
    extincted_band2: np.ndarray
    selection_intrinsic_band1: np.ndarray
    selection_intrinsic_band2: np.ndarray
    selection_extincted_band1: np.ndarray
    selection_extincted_band2: np.ndarray
    native_times: dict[str, list[np.ndarray]]
    native_colours: dict[str, list[np.ndarray]]
    selection_system: str
    missing_event_ids: list[str]


def _normalise_lightcurve(lightcurve: pd.DataFrame) -> pd.DataFrame:
    required = {"t_days", "band", "mag"}
    missing = required.difference(lightcurve.columns)
    if missing:
        raise ValueError(f"Synthetic light curve is missing {sorted(missing)}")
    return pd.DataFrame({
        "t_days": pd.to_numeric(lightcurve["t_days"], errors="coerce"),
        "band": lightcurve["band"].astype(str).str.strip().str.lower(),
        "magnitude": pd.to_numeric(lightcurve["mag"], errors="coerce"),
        "photometric_system": (
            lightcurve["photometric_system"].astype(str).str.strip().str.lower()
            if "photometric_system" in lightcurve else "lsst"
        ),
    })


def interpolate_saved_population(
    source: SyntheticPopulationSource,
    metadata: pd.DataFrame,
    band1: str,
    band2: str,
    centres: np.ndarray,
    t_min: float,
    t_max: float,
    photometric_system: str,
) -> InterpolatedPopulation:
    """Read each selected event once and construct fixed-centre matrices."""
    event_ids = metadata["event_id"].tolist()
    n_events, n_times = len(metadata), len(centres)
    intrinsic_band1 = np.full((n_events, n_times), np.nan, dtype=np.float32)
    intrinsic_band2 = np.full_like(intrinsic_band1, np.nan)
    extincted_band1 = np.full_like(intrinsic_band1, np.nan)
    extincted_band2 = np.full_like(intrinsic_band1, np.nan)
    selection_intrinsic_band1 = np.full_like(intrinsic_band1, np.nan)
    selection_intrinsic_band2 = np.full_like(intrinsic_band1, np.nan)
    selection_extincted_band1 = np.full_like(intrinsic_band1, np.nan)
    selection_extincted_band2 = np.full_like(intrinsic_band1, np.nan)
    native_times = {stage: [None] * n_events for stage in STAGES}
    native_colours = {stage: [None] * n_events for stage in STAGES}

    selection_system = "lsst" if photometric_system == "ps1" else photometric_system
    requested_systems = tuple(dict.fromkeys((photometric_system, selection_system)))
    positions = {
        canonical_event_id(event_id): index
        for index, event_id in enumerate(event_ids)
    }
    metadata_by_event = metadata.set_index("event_id", drop=False)
    processed: set[str] = set()
    progress_interval = max(500, n_events // 20)
    print(
        f"Reading {n_events:,} saved synthetic light curves; numerical support "
        "is inherited from generate_kne_lightcurves.py"
    )

    for event_id, raw_lightcurve in source.iter_events(
        event_ids, (band1, band2), requested_systems
    ):
        event_id = canonical_event_id(event_id)
        if event_id in processed:
            raise RuntimeError(
                f"Synthetic event {event_id} occurs in more than one input group/shard"
            )
        processed.add(event_id)
        matrix_index = positions[event_id]
        metadata_row = metadata_by_event.loc[event_id]
        lightcurve = _normalise_lightcurve(raw_lightcurve)
        lightcurve = lightcurve.loc[
            np.isfinite(lightcurve["t_days"])
            & (lightcurve["t_days"] > 0)
            & np.isfinite(lightcurve["magnitude"])
            & lightcurve["band"].isin((band1, band2))
            & lightcurve["photometric_system"].isin(requested_systems)
        ].copy()
        target_curve = lightcurve.loc[
            lightcurve["photometric_system"] == photometric_system
        ].copy()
        selection_curve = lightcurve.loc[
            lightcurve["photometric_system"] == selection_system
        ].copy()

        mw1 = event_extinction_mag(metadata_row, photometric_system, band1)
        mw2 = event_extinction_mag(metadata_row, photometric_system, band2)
        selection_mw1 = event_extinction_mag(metadata_row, selection_system, band1)
        selection_mw2 = event_extinction_mag(metadata_row, selection_system, band2)

        centre_ext1 = interpolate_band(target_curve, band1, centres)
        centre_ext2 = interpolate_band(target_curve, band2, centres)
        extincted_band1[matrix_index] = centre_ext1
        extincted_band2[matrix_index] = centre_ext2
        intrinsic_band1[matrix_index] = centre_ext1 - mw1
        intrinsic_band2[matrix_index] = centre_ext2 - mw2
        centre_selection_ext1 = interpolate_band(selection_curve, band1, centres)
        centre_selection_ext2 = interpolate_band(selection_curve, band2, centres)
        selection_extincted_band1[matrix_index] = centre_selection_ext1
        selection_extincted_band2[matrix_index] = centre_selection_ext2
        selection_intrinsic_band1[matrix_index] = centre_selection_ext1 - selection_mw1
        selection_intrinsic_band2[matrix_index] = centre_selection_ext2 - selection_mw2

        paired = (
            target_curve.groupby(["t_days", "band"], as_index=False)["magnitude"]
            .median().pivot(index="t_days", columns="band", values="magnitude")
        )
        if band1 not in paired.columns or band2 not in paired.columns:
            times = np.array([], dtype=float)
            saved1 = np.array([], dtype=float)
            saved2 = np.array([], dtype=float)
        else:
            paired = paired.dropna(subset=[band1, band2]).sort_index()
            times = paired.index.to_numpy(float)
            saved1 = paired[band1].to_numpy(float)
            saved2 = paired[band2].to_numpy(float)
        finite_native = (
            np.isfinite(times) & np.isfinite(saved1) & np.isfinite(saved2)
            & (times >= t_min) & (times <= t_max)
        )
        intrinsic_colour = (saved1 - mw1) - (saved2 - mw2)
        extincted_colour = saved1 - saved2
        for stage in STAGES:
            native_times[stage][matrix_index] = times[finite_native].astype(np.float32)
            colours = extincted_colour if stage in MW_EXTINCTED_STAGES else intrinsic_colour
            native_colours[stage][matrix_index] = colours[finite_native].astype(np.float32)

        if len(processed) % progress_interval == 0:
            print(f"  {len(processed):,}/{n_events:,}")

    missing_event_ids = [
        canonical_event_id(event_id) for event_id in event_ids
        if canonical_event_id(event_id) not in processed
    ]
    if missing_event_ids:
        empty = np.array([], dtype=np.float32)
        for event_id in missing_event_ids:
            matrix_index = positions[event_id]
            for stage in STAGES:
                native_times[stage][matrix_index] = empty.copy()
                native_colours[stage][matrix_index] = empty.copy()
    print(
        f"Finished reading: {len(processed):,} events with saved {band1}-{band2} "
        f"rows; {len(missing_event_ids):,} colour-unavailable events."
    )
    return InterpolatedPopulation(
        intrinsic_band1, intrinsic_band2, extincted_band1, extincted_band2,
        selection_intrinsic_band1, selection_intrinsic_band2,
        selection_extincted_band1, selection_extincted_band2,
        native_times, native_colours, selection_system, missing_event_ids,
    )
