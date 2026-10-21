"""Numerical conventions shared by colour-envelope analyses.

This module contains deterministic table/statistical transformations only. It
does not parse terminal input, read a run, create figures, or execute a
pipeline when imported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kllap.colour.common import bootstrap_percentiles_batched


RUBIN_MW_R = {
    "u": 4.757217815396922,
    "g": 3.6605664439892616,
    "r": 2.70136780871597,
    "i": 2.0536599130965882,
    "z": 1.5900964472616756,
    "y": 1.3077049588254708,
}

PS1_MW_R = {
    "g": 3.172,
    "r": 2.271,
    "i": 1.682,
    "z": 1.322,
    "y": 1.087,
}

ZTF_MW_R = {
    "g": 3.303,
    "r": 2.285,
    "i": 1.698,
}

MW_R_BY_SYSTEM = {
    "lsst": RUBIN_MW_R,
    "ps1": PS1_MW_R,
    "ztf": ZTF_MW_R,
}

# q is the lower percentile of the symmetric p_q--p_(100-q) envelope.
DEFAULT_TAIL_GRID = (
    "0.02,0.05,0.1,0.25,0.5,0.53,1,2.5,5,10,16,20,25,30,35,"
    "40,42,44,45,46,47,48,49,50"
)

STAGES = (
    "intrinsic_no_m5_no_mw",
    "no_m5_with_mw",
    "m5_no_mw",
    "m5_with_mw",
)

MW_EXTINCTED_STAGES = frozenset({"no_m5_with_mw", "m5_with_mw"})
M5_STAGES = frozenset({"m5_no_mw", "m5_with_mw"})

# Preserve the historical bootstrap streams of the original three stages.
STAGE_BOOTSTRAP_SEED_OFFSETS = {
    "intrinsic_no_m5_no_mw": 1,
    "m5_no_mw": 2,
    "m5_with_mw": 3,
    "no_m5_with_mw": 4,
}


def event_extinction_mag(
    metadata_row: pd.Series,
    photometric_system: str,
    band: str,
) -> float:
    """Return the stored A_band, or reconstruct it from E(B-V)."""
    column = f"A_{photometric_system}_{band}_mw"
    if column in metadata_row.index:
        value = pd.to_numeric(
            pd.Series([metadata_row[column]]), errors="coerce"
        ).iloc[0]
        if np.isfinite(value):
            return float(value)
    ebv = float(metadata_row["ebv_mw"])
    return float(MW_R_BY_SYSTEM[photometric_system][band] * ebv)


def select_metadata_subset(
    metadata: pd.DataFrame,
    maximum_events: int,
    subsample_seed: int | None,
) -> pd.DataFrame:
    """Select a reproducible subset nested across sizes for one fixed seed."""
    if maximum_events <= 0 or maximum_events >= len(metadata):
        return metadata.copy().reset_index(drop=True)
    maximum = min(int(maximum_events), len(metadata))
    if subsample_seed is None:
        return metadata.iloc[:maximum].copy().reset_index(drop=True)
    subset_rng = np.random.default_rng(subsample_seed)
    selected = subset_rng.permutation(len(metadata))[:maximum]
    subset = metadata.iloc[selected].copy()
    return subset.sort_values(
        "event_id", key=lambda s: pd.to_numeric(s, errors="coerce")
    ).reset_index(drop=True)


def percentile_diagnostics(
    values: np.ndarray,
    lower_grid: list[float],
    n_boot: int,
    bootstrap_threshold: float,
    minimum_tail_events: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    """Validate percentiles and retain the widest valid symmetric envelope."""
    lower = sorted(set(float(q) for q in lower_grid if 0.0 < q <= 50.0))
    percentiles = sorted(set(lower + [100.0 - q for q in lower] + [50.0]))
    estimate, boot16, boot84, half = bootstrap_percentiles_batched(
        values, percentiles, n_boot, rng
    )
    n_values = len(values)
    rows = []
    for q, value, low, high, width in zip(
        percentiles, estimate, boot16, boot84, half
    ):
        expected_tail = n_values * min(q, 100.0 - q) / 100.0
        width_valid = bool(np.isfinite(width) and width <= bootstrap_threshold)
        tail_valid = bool(expected_tail >= minimum_tail_events)
        rows.append(
            {
                "percentile": q,
                "value": value,
                "bootstrap_p16": low,
                "bootstrap_p84": high,
                "bootstrap_half_width_68": width,
                "expected_nearest_tail_events": expected_tail,
                "bootstrap_width_valid": width_valid,
                "tail_support_valid": tail_valid,
                "percentile_valid": bool(width_valid and tail_valid),
            }
        )
    diagnostics = pd.DataFrame(rows)
    by_q = diagnostics.set_index("percentile")
    median_valid = bool(by_q.loc[50.0, "percentile_valid"])
    valid_pairs = [
        q
        for q in lower
        if median_valid
        and bool(by_q.loc[q, "percentile_valid"])
        and bool(by_q.loc[100.0 - q, "percentile_valid"])
    ]
    if not valid_pairs:
        return diagnostics, None

    q_low = min(valid_pairs)
    q_high = 100.0 - q_low
    selected = {
        "low_percentile": q_low,
        "high_percentile": q_high,
        "color_min": float(by_q.loc[q_low, "value"]),
        "color_median": float(by_q.loc[50.0, "value"]),
        "color_max": float(by_q.loc[q_high, "value"]),
        "low_bootstrap_half_width_68": float(
            by_q.loc[q_low, "bootstrap_half_width_68"]
        ),
        "median_bootstrap_half_width_68": float(
            by_q.loc[50.0, "bootstrap_half_width_68"]
        ),
        "high_bootstrap_half_width_68": float(
            by_q.loc[q_high, "bootstrap_half_width_68"]
        ),
        "expected_lower_tail_events": float(n_values * q_low / 100.0),
        "expected_upper_tail_events": float(n_values * q_low / 100.0),
        "nominal_coverage_percent": float(q_high - q_low),
    }
    return diagnostics, selected
