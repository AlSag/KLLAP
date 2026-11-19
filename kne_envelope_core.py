"""Numerical conventions shared by colour-envelope analyses.

This module contains deterministic table/statistical transformations only. It
does not parse terminal input, read a run, create figures, or execute a
pipeline when imported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STAGES = ("no_m5", "m5")
M5_STAGES = frozenset({"m5"})

DEFAULT_TAIL_GRID = (
    "0.02,0.05,0.1,0.25,0.5,0.53,1,2.5,5,10,16,20,25,30,35,"
    "40,42,44,45,46,47,48,49,50"
)

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
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    """Compute one fixed empirical percentile interval without resampling."""

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    lower = list(dict.fromkeys(
        float(q) for q in lower_grid if 0.0 <= q <= 50.0
    ))
    percentiles = sorted(set(lower + [100.0 - q for q in lower] + [50.0]))
    if not len(finite_values) or not percentiles:
        return pd.DataFrame(), None

    estimates = np.percentile(finite_values, percentiles, method="linear")
    n_values = len(finite_values)
    rows: list[dict[str, object]] = []
    for percentile, estimate in zip(percentiles, estimates):
        expected_tail = (
            n_values * min(percentile, 100.0 - percentile) / 100.0
        )
        rows.append({
            "percentile": percentile,
            "value": float(estimate),
            "expected_nearest_tail_events": expected_tail,
            "percentile_valid": True,
            "estimator": "numpy.percentile(method='linear')",
        })

    diagnostics = pd.DataFrame(rows)
    return diagnostics, select_fixed_percentile_interval(
        diagnostics, n_values, lower
    )


def select_fixed_percentile_interval(
    diagnostics: pd.DataFrame,
    n_values: int,
    lower_sequence: list[float],
) -> dict[str, float | str | bool] | None:
    """Return the single fixed interval requested by the user."""
    if diagnostics.empty:
        return None

    by_percentile = diagnostics.set_index("percentile")
    if 50.0 not in by_percentile.index:
        return None
    median = by_percentile.loc[50.0]
    if not bool(median["percentile_valid"]):
        return None

    sequence = list(dict.fromkeys(
        float(q) for q in lower_sequence if 0.0 <= float(q) <= 50.0
    ))
    for rank, q_low in enumerate(sequence, start=1):
        q_high = 100.0 - q_low
        if q_low == 50.0:
            low = high = median
            definition = "median_only"
        else:
            if q_low not in by_percentile.index or q_high not in by_percentile.index:
                continue
            low = by_percentile.loc[q_low]
            high = by_percentile.loc[q_high]
            if not (
                bool(low["percentile_valid"])
                and bool(high["percentile_valid"])
            ):
                continue
            definition = f"p{q_low:g}-p{q_high:g}"

        return {
            "low_percentile": q_low,
            "high_percentile": q_high,
            "color_min": float(low["value"]),
            "color_median": float(median["value"]),
            "color_max": float(high["value"]),
            "expected_lower_tail_events": float(n_values * q_low / 100.0),
            "expected_upper_tail_events": float(n_values * q_low / 100.0),
            "nominal_coverage_percent": float(q_high - q_low),
            "envelope_definition": definition,
            "selection_rank": rank,
            "median_only": bool(q_low == 50.0),
        }
    return None
