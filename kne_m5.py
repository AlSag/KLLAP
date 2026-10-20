#!/usr/bin/env python3
"""Resolve OpSim limiting-depth percentiles for analysis scripts.

This module owns only m5 input discovery and parsing.  It supports the two
conventions already used by the repository:

* an unscoped lookup that prefers the common/global, both-latitude rows when
  they are present;
* an explicitly scoped lookup for the isocontour analysis, including the
  existing latitude-conditioned ``high`` selection.

If an exact percentile is not available in a table, a common/global lookup
may calculate it directly from the raw OpSim visit files.  A
latitude-conditioned lookup never falls back to unconditioned raw visits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_M5_SOURCE_CACHE: dict[
    tuple[str, float, tuple[str, ...], str | None],
    tuple[dict[str, float], str],
] = {}


def recorded_paths(run_dir: Path) -> list[Path]:
    """Return path-like values recorded in ``run_info.txt``."""
    path = run_dir / "run_info.txt"
    if not path.is_file():
        return []
    found = []
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        value = line.split(":", 1)[1].strip().strip("'\"")
        if value and ("/" in value or value.endswith(".csv")):
            found.append(Path(value).expanduser())
    return found


def m5_source_candidates(run_dir: Path, explicit: Path | None) -> list[Path]:
    """Return unique candidate tables and raw OpSim directories in priority order."""
    candidates = []
    if explicit is not None:
        explicit = explicit.expanduser()
        if explicit.is_dir():
            candidates.extend([
                explicit / "m5_depth_quantiles_by_band.csv",
                explicit / "opsim_m5_medians_by_band.csv",
                explicit,
            ])
        else:
            candidates.append(explicit)

    for base in (run_dir, run_dir / "products", run_dir.parent, Path.cwd()):
        candidates.extend([
            base / "m5_depth_quantiles_by_band.csv",
            base / "analysis" / "latitude_depth_comparison"
            / "m5_depth_quantiles_by_band.csv",
            base / "opsim_m5_medians_by_band.csv",
        ])

    for recorded in recorded_paths(run_dir):
        base = recorded if recorded.is_dir() else recorded.parent
        candidates.extend([
            base / "m5_depth_quantiles_by_band.csv",
            base / "opsim_m5_medians_by_band.csv",
            recorded,
        ])

    unique = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def normalised_quantiles(series: pd.Series) -> np.ndarray:
    """Express fractional or percentage quantile columns on a 0--100 scale."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return np.where(values <= 1.0, 100.0 * values, values)


def _value_column(table: pd.DataFrame, percentile: float) -> str | None:
    labels = [
        f"m5_p{percentile:g}",
        f"p{percentile:g}",
        f"m5_q{percentile:g}",
    ]
    if np.isclose(percentile, 50.0):
        labels.extend(["m5_median", "median_m5"])
    return next((label for label in labels if label in table), None)


def _select_scope(
    table: pd.DataFrame,
    threshold_scope: str | None,
) -> pd.DataFrame:
    """Apply the repository's existing global or latitude scope convention."""
    use = table
    if threshold_scope is None:
        if (
            "threshold_scope" in use
            and use["threshold_scope"].astype(str).eq("common_global").any()
        ):
            use = use.loc[
                use["threshold_scope"].astype(str).eq("common_global")
            ]
        if (
            "threshold_latitude_group" in use
            and use["threshold_latitude_group"].astype(str).eq("both").any()
        ):
            use = use.loc[
                use["threshold_latitude_group"].astype(str).eq("both")
            ]
        return use

    if "threshold_scope" in use:
        use = use.loc[
            use["threshold_scope"].astype(str).eq(threshold_scope)
        ]
    if "threshold_latitude_group" in use:
        latitude_group = "both" if threshold_scope == "common_global" else "high"
        use = use.loc[
            use["threshold_latitude_group"].astype(str).eq(latitude_group)
        ]
    return use


def depths_from_table(
    path: Path,
    percentile: float,
    bands: tuple[str, ...],
    threshold_scope: str | None = None,
) -> dict[str, float] | None:
    """Read an exact percentile from long, wide, or legacy-median m5 tables."""
    try:
        table = pd.read_csv(path)
    except Exception:
        return None
    if "band" not in table:
        return None

    table = table.copy()
    table["band"] = table["band"].astype(str).str.strip().str.lower()
    use = _select_scope(table, threshold_scope)

    if {"quantile", "m5"}.issubset(use.columns):
        quantiles = normalised_quantiles(use["quantile"])
        use = use.loc[np.isclose(quantiles, percentile, atol=1e-9)]
        value_column = "m5"
    else:
        value_column = _value_column(use, percentile)
        if value_column is None:
            return None

    if threshold_scope is None:
        lookup = dict(zip(
            use["band"],
            pd.to_numeric(use[value_column], errors="coerce"),
        ))
        result = {
            band: float(lookup[band])
            for band in bands
            if band in lookup and np.isfinite(lookup[band])
        }
        return result if len(result) == len(bands) else None

    result = {}
    for band in bands:
        rows = use.loc[use["band"].eq(band)]
        values = pd.to_numeric(rows[value_column], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) != 1:
            return None
        result[band] = float(values.iloc[0])
    return result


def depths_from_raw_opsim(
    source: Path,
    percentile: float,
    bands: tuple[str, ...],
) -> dict[str, float] | None:
    """Compute an arbitrary visit-weighted percentile from raw OpSim CSVs."""
    root = source if source.is_dir() else source.parent
    files = sorted(root.rglob("opsim_visits_field*.csv"))
    if not files:
        return None

    values = {band: [] for band in bands}
    accepted = {"band", "filter", "m5", "fiveSigmaDepth"}
    print(
        f"Computing m5 p{percentile:g} from {len(files):,} raw OpSim field "
        f"files in {root}"
    )
    for path in files:
        frame = pd.read_csv(path, usecols=lambda name: name in accepted)
        band_column = (
            "band" if "band" in frame
            else "filter" if "filter" in frame else None
        )
        depth_column = (
            "m5" if "m5" in frame
            else "fiveSigmaDepth" if "fiveSigmaDepth" in frame else None
        )
        if band_column is None or depth_column is None:
            continue
        normalised_bands = (
            frame[band_column].astype(str).str.strip().str.lower()
            .str.replace(r"^lsst", "", regex=True).str[-1:]
        )
        depths = pd.to_numeric(frame[depth_column], errors="coerce")
        for band in bands:
            selected = depths.loc[normalised_bands.eq(band)].to_numpy(float)
            selected = selected[np.isfinite(selected)]
            if selected.size:
                values[band].append(selected)

    if any(not values[band] for band in bands):
        return None
    return {
        band: float(np.percentile(np.concatenate(values[band]), percentile))
        for band in bands
    }


def load_m5_percentile(
    run_dir: Path,
    explicit: Path | None,
    percentile: float,
    bands: tuple[str, ...],
    threshold_scope: str | None = None,
) -> tuple[dict[str, float], str]:
    """Resolve an m5 percentile from a table or permitted raw OpSim visits."""
    for candidate in m5_source_candidates(run_dir, explicit):
        if not candidate.exists():
            continue
        key = (
            str(candidate),
            float(percentile),
            tuple(bands),
            threshold_scope,
        )
        if key in _M5_SOURCE_CACHE:
            return _M5_SOURCE_CACHE[key]

        if candidate.is_file():
            depths = depths_from_table(
                candidate,
                percentile,
                bands,
                threshold_scope,
            )
            if depths is not None:
                result = (depths, f"p{percentile:g} table: {candidate}")
                _M5_SOURCE_CACHE[key] = result
                return result

        if threshold_scope in (None, "common_global"):
            depths = depths_from_raw_opsim(candidate, percentile, bands)
            if depths is not None:
                root = candidate if candidate.is_dir() else candidate.parent
                result = (depths, f"p{percentile:g} raw OpSim visits: {root}")
                _M5_SOURCE_CACHE[key] = result
                return result

    conditioned = (
        " Latitude-conditioned thresholds require the exact percentile in "
        "the table; unconditioned raw visits are not a valid fallback."
        if threshold_scope == "latitude_conditioned" else ""
    )
    raise FileNotFoundError(
        f"Cannot determine m5 p{percentile:g} for bands {bands}.{conditioned} "
        "Provide an m5 quantile table or a raw OpSim export directory."
    )
