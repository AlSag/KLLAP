#!/usr/bin/env python3
"""Shared utilities for the kilonova simulation and analysis pipeline.

Purpose
-------
This module centralizes small operations that are needed by more than one
pipeline stage, or that would otherwise obscure the main scientific workflow.
Importing it does not execute a simulation and does not define scientific run
configuration.  Physical priors, distance limits, detection thresholds, and
output modes remain in the scripts that own those choices.

How to use this file
--------------------
Keep ``kne_pipeline_common.py`` in the same directory as the scripts that
import it.  Public helpers have names without a leading underscore.  Functions
whose names start with ``_`` are implementation details and should not normally
be imported by another script.

The file is organized by responsibility rather than by calling script.  Each
section has a ``Primary users`` note to make navigation easier, but those notes
are informative rather than restrictive: a shared helper may acquire new
callers without changing its behavior.

Main consumers
--------------
``generate_kne_lightcurves.py``
    Uses terminal prompts, output-table conversion, interpolation, candidate
    identifiers, posterior sampling, OpSim metadata, and run-info formatting.

``get_opsim_baseline.py``
    Uses terminal prompts and filesystem-safe names.  Its exported metadata is
    subsequently read by the generator through this module.

Product-extraction and plotting scripts
    Use run discovery, light-curve file discovery, CSV parsing, OpSim/m5 source
    resolution, m5 summaries, and bootstrap statistics as needed.

Navigation
----------
1. Pipeline constants
2. Naming and CSV parsing
3. Terminal prompts
4. Generator output and interpolation
5. OpSim export metadata
6. Run and product discovery
7. OpSim/m5 source reading
8. FIESTA posterior loading and sampling
9. m5 summary products
10. Statistical helpers
"""

from __future__ import annotations

import io
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence, TextIO

import numpy as np
import pandas as pd


# =============================================================================
# 1. PIPELINE CONSTANTS
# Primary users: generator, posterior tools, and band-based analysis scripts.
# =============================================================================

BANDS = tuple("ugrizy")
PHYSICAL_PARAMETERS = (
    "inclination_EM",
    "log10_mej_dyn",
    "v_ej_dyn",
    "Ye_dyn",
    "log10_mej_wind",
    "v_ej_wind",
    "Ye_wind",
)


# =============================================================================
# 2. GENERAL NAMING AND CSV PARSING HELPERS
# Primary users: all command-line pipeline scripts and downstream analyses.
# =============================================================================

def slugify(value: str) -> str:
    """Convert a user-facing label into a filesystem-safe run name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._-") or "run"


def event_id_from_path(path: Path) -> int | None:
    """Extract the trailing integer event ID from a light-curve CSV name."""
    match = re.search(r"_(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else None


def parse_bool_series(values: pd.Series) -> np.ndarray:
    """Normalize common CSV boolean representations into a NumPy mask."""
    if values.dtype == bool:
        return values.fillna(False).to_numpy(bool)
    if np.issubdtype(values.dtype, np.number):
        return (pd.to_numeric(values, errors="coerce").fillna(0) != 0).to_numpy(bool)
    return (
        values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})
    ).to_numpy(bool)


def parse_csv_strings(value: str) -> list[str]:
    """Split a comma-separated option and discard empty entries."""
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    """Parse a comma-separated option as floating-point values."""
    return [float(item) for item in parse_csv_strings(value)]


def parse_pairs(value: str) -> list[tuple[str, str, str]]:
    """Parse requested color pairs such as ``g_i,r_z`` or ``all``."""
    if str(value).strip().lower() == "all":
        return [(b1, b2, f"{b1}_{b2}") for i, b1 in enumerate(BANDS) for b2 in BANDS[i + 1 :]]
    result: list[tuple[str, str, str]] = []
    for item in parse_csv_strings(value):
        clean = item.lower().replace("-", "_")
        parts = clean.split("_")
        if len(parts) != 2 or parts[0] not in BANDS or parts[1] not in BANDS:
            raise ValueError(f"Invalid color pair {item!r}; expected e.g. g_i,r_z")
        result.append((parts[0], parts[1], clean))
    if not result:
        raise ValueError("At least one color pair is required")
    return result


# =============================================================================
# 3. TERMINAL PROMPT HELPERS
# Primary users: generate_kne_lightcurves.py and get_opsim_baseline.py.
# =============================================================================

def prompt_choice(question: str, choices: Sequence[tuple[str, str]], default: str) -> str:
    """Display a numbered terminal menu and return the selected key."""
    print(f"\n{question}")
    for index, (key, label) in enumerate(choices, start=1):
        marker = " [default]" if key == default else ""
        print(f"  {index}. {label}{marker}")
    raw = input("Choice: ").strip()
    if not raw:
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1][0]
    keys = {key for key, _ in choices}
    if raw in keys:
        return raw
    raise ValueError(f"Unknown choice {raw!r}; valid keys: {sorted(keys)}")


def prompt_text(question: str, default: str | None = None) -> str:
    """Prompt for text, returning the displayed default for an empty answer."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    raw = input(f"{question}{suffix}: ").strip()
    if raw:
        return raw
    if default is None:
        raise ValueError(f"A value is required for {question}")
    return default


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for a boolean answer with an explicit default."""
    marker = "Y/n" if default else "y/N"
    raw = input(f"{question} [{marker}]: ").strip().lower()
    if not raw:
        return default
    if raw in {"y", "yes", "o", "oui", "1", "true"}:
        return True
    if raw in {"n", "no", "non", "0", "false"}:
        return False
    raise ValueError(f"Expected yes/no, got {raw!r}")


# =============================================================================
# 4. GENERATOR OUTPUT AND INTERPOLATION HELPERS
# Primary user: generate_kne_lightcurves.py.
#
# These functions contain format mechanics, not scientific population choices:
# log-time interpolation, optional output resampling, long measurement-table
# construction, unique identifiers, prior-bound extraction, and run-info formatting.
# =============================================================================


def interpolate_magnitude_log_time(
    model_times: Sequence[float] | np.ndarray,
    model_magnitudes: Sequence[float] | np.ndarray,
    query_times: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Interpolate magnitudes linearly as a function of log10(time).

    Parameters
    ----------
    model_times
        Strictly increasing, finite, positive observer-frame times in days.
    model_magnitudes
        Magnitudes at ``model_times``.  The array must have the same length as
        ``model_times``.  Non-finite magnitude samples are ignored.
    query_times
        Times in days at which the magnitudes are requested.

    Returns
    -------
    numpy.ndarray
        Interpolated magnitudes with the same shape as ``query_times``.  Query
        times outside the model interval, or non-finite query times, are
        returned as NaN; the function never extrapolates.

    Notes
    -----
    Only the time coordinate is transformed.  Magnitudes remain linear.  This
    convention is useful when the light curve is approximately a power law in
    flux, because magnitude is then approximately linear in log10(time).
    """
    times = np.asarray(model_times, dtype=float)
    magnitudes = np.asarray(model_magnitudes, dtype=float)
    query = np.asarray(query_times, dtype=float)

    if times.ndim != 1 or magnitudes.ndim != 1:
        raise ValueError("model_times and model_magnitudes must be one-dimensional")
    if len(times) != len(magnitudes):
        raise ValueError("model_times and model_magnitudes must have the same length")
    if len(times) < 2:
        raise ValueError("At least two model samples are required for interpolation")
    if np.any(~np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("model_times must contain only finite, strictly positive values")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("model_times must be strictly increasing")
    finite_magnitudes = np.isfinite(magnitudes)
    if np.count_nonzero(finite_magnitudes) < 2:
        return np.full(query.shape, np.nan, dtype=float)
    interpolation_times = times[finite_magnitudes]
    interpolation_magnitudes = magnitudes[finite_magnitudes]

    result = np.full(query.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(query)
        & (query >= interpolation_times[0])
        & (query <= interpolation_times[-1])
    )
    result[valid] = np.interp(
        np.log10(query[valid]),
        np.log10(interpolation_times),
        interpolation_magnitudes,
    )
    return result


def resample_synthetic_output_grid(
    times_model: Sequence[float] | np.ndarray,
    mag_model_dict: Mapping[str, Sequence[float] | np.ndarray],
    filters: Sequence[str],
    cadence_days: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the time grid and magnitudes written to synthetic CSV files.

    A non-positive cadence preserves the native FIESTA grid.  A positive
    cadence creates a grid that is regular in observer-frame days and evaluates
    every requested filter with :func:`interpolate_magnitude_log_time`.  This
    resampling affects only saved synthetic products; LSST-like calculations
    should continue to use the native model grid.
    """
    native_times = np.asarray(times_model, dtype=float)
    cadence = float(cadence_days)
    if cadence <= 0.0:
        return native_times, {
            filter_name: np.asarray(mag_model_dict[filter_name], dtype=float)
            for filter_name in filters
        }

    if np.any(~np.isfinite(native_times)) or np.any(native_times <= 0.0):
        raise ValueError("The FIESTA time grid must be finite and strictly positive")
    if np.any(np.diff(native_times) <= 0.0):
        raise ValueError("The FIESTA time grid must be strictly increasing")

    time_min = float(native_times[0])
    time_max = float(native_times[-1])
    n_steps = int(np.floor((time_max - time_min) / cadence))
    output_times = time_min + cadence * np.arange(n_steps + 1, dtype=float)

    tolerance = max(1e-12, 1e-9 * cadence)
    if time_max - output_times[-1] > tolerance:
        output_times = np.append(output_times, time_max)

    output_magnitudes = {
        filter_name: interpolate_magnitude_log_time(
            native_times,
            mag_model_dict[filter_name],
            output_times,
        )
        for filter_name in filters
    }
    return output_times, output_magnitudes


def build_synthetic_measurement_table(
    objectid: str,
    times_days: Sequence[float] | np.ndarray,
    magnitudes_by_filter: Mapping[str, Sequence[float] | np.ndarray],
    filters: Sequence[str],
    filter_by_band: Mapping[str, str],
    *,
    time_mjd: Sequence[float] | np.ndarray | None = None,
    mw_extinction_by_band: Mapping[str, float] | None = None,
    magnitude_limit: float | None = None,
) -> pd.DataFrame:
    """Build the long synthetic table written for one event.

    The returned layout has one row per finite ``(time, band)`` measurement,
    rather than one row per time with a separate magnitude column for every
    filter.  This is often called *long format* in data analysis.

    Parameters
    ----------
    objectid
        Event identifier, for example ``"0007"``.
    times_days
        Observer-frame times since merger, in days.
    magnitudes_by_filter
        Intrinsic magnitude array for each FIESTA filter.
    filters
        Filter order.  It determines row order within every time step.
    filter_by_band
        Mapping from short band names such as ``"g"`` to FIESTA filter names
        such as ``"lsstg"``.
    time_mjd
        Optional absolute MJD array with the same length as ``times_days``.
    mw_extinction_by_band
        Optional mapping from short band name to ``A_band``.  When supplied,
        the table includes ``mag_mw_extincted = mag + A_band``.
    magnitude_limit
        Optional faint-magnitude cap.  Intrinsic magnitudes numerically larger
        than this value are discarded before candidate identifiers are made.

    Returns
    -------
    pandas.DataFrame
        Columns are ``objectid``, ``candidate_id``, optional ``time_mjd``,
        ``t_days``, ``band``, ``mag``, and optional ``mag_mw_extincted``.
        Rows are ordered by time and then by ``filters``.  ``objectid`` is
        written only in the first row because event metadata live once in
        ``summary.csv``.  Every retained measurement receives a unique
        ``candidate_id``.

    Notes
    -----
    The function constructs the long table directly.  It intentionally avoids
    creating a temporary wide DataFrame, reducing allocations without changing
    the saved CSV schema or row order.
    """
    times = np.asarray(times_days, dtype=float)
    if times.ndim != 1:
        raise ValueError("times_days must be one-dimensional")

    n_times = len(times)
    if time_mjd is not None:
        absolute_times = np.asarray(time_mjd, dtype=float)
        if absolute_times.shape != times.shape:
            raise ValueError("time_mjd must have the same shape as times_days")
    else:
        absolute_times = None

    filter_to_band = {
        filter_name: band for band, filter_name in filter_by_band.items()
    }
    missing_filters = [name for name in filters if name not in filter_to_band]
    if missing_filters:
        raise KeyError(f"No short-band mapping for filters: {missing_filters}")

    magnitude_columns = []
    for filter_name in filters:
        values = np.asarray(magnitudes_by_filter[filter_name], dtype=float).copy()
        if values.shape != times.shape:
            raise ValueError(
                f"Magnitude array for {filter_name!r} must match times_days"
            )
        if magnitude_limit is not None:
            values[values > float(magnitude_limit)] = np.nan
        magnitude_columns.append(values)

    n_filters = len(filters)
    bands = [filter_to_band[name] for name in filters]
    measurement_data: dict[str, Any] = {}
    if absolute_times is not None:
        measurement_data["time_mjd"] = np.repeat(absolute_times, n_filters)
    measurement_data["t_days"] = np.repeat(times, n_filters)
    measurement_data["band"] = np.tile(bands, n_times)
    measurement_data["mag"] = np.column_stack(magnitude_columns).reshape(-1)

    if mw_extinction_by_band is not None:
        extinction = np.tile(
            [float(mw_extinction_by_band.get(band, np.nan)) for band in bands],
            n_times,
        )
        measurement_data["mag_mw_extincted"] = (
            measurement_data["mag"] + extinction
        )

    measurements = pd.DataFrame(measurement_data)
    measurements = measurements[np.isfinite(measurements["mag"])].reset_index(
        drop=True
    )
    objectid = str(objectid)
    candidate_ids = [
        f"{objectid}_{measurement_index}"
        for measurement_index in range(1, len(measurements) + 1)
    ]
    objectids = np.full(len(measurements), "", dtype=object)
    if len(measurements):
        objectids[0] = objectid
    measurements.insert(0, "candidate_id", candidate_ids)
    measurements.insert(0, "objectid", objectids)
    return measurements


def add_lsstlike_candidate_ids(
    observations: pd.DataFrame,
    event_index: int,
) -> pd.DataFrame:
    """Add event and unique per-visit identifiers to an LSST-like table.

    Every retained post-trigger OpSim visit receives an identifier, including
    non-detections and visits outside the FIESTA model interval.  The ``LSST``
    marker keeps these identifiers distinct from synthetic point identifiers.
    """
    identified = observations.copy()
    if "objectid" in identified.columns or "candidate_id" in identified.columns:
        raise ValueError("LSST-like observations already contain identifiers")

    objectid = f"{int(event_index):04d}"
    candidate_ids = [
        f"{objectid}_LSST_{visit_index:06d}"
        for visit_index in range(1, len(identified) + 1)
    ]
    objectids = np.full(len(identified), "", dtype=object)
    if len(identified):
        objectids[0] = objectid
    identified.insert(0, "candidate_id", candidate_ids)
    identified.insert(0, "objectid", objectids)
    return identified


def prior_bounds_dict(prior_components: Iterable[Any]) -> dict[str, tuple[float, float]]:
    """Return ``parameter -> (minimum, maximum)`` for FIESTA prior components."""
    return {
        component.naming[0]: (float(component.xmin), float(component.xmax))
        for component in prior_components
    }


def format_run_info_sections(
    sections: Mapping[str, Mapping[str, Any]],
) -> str:
    """Format a concise, human-readable ``run_info.txt``.

    Empty sections and entries whose value is ``None`` are omitted.  Lists and
    mappings remain on one line so the file records essential configuration
    without duplicating detailed metadata stored in dedicated CSV files.
    Section and entry order follow the insertion order of the supplied
    mappings.
    """

    def format_value(value: Any) -> str:
        """Render one compact configuration value without multiline output."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return ", ".join(f"{key}={item}" for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return ", ".join(map(str, value))
        return str(value)

    lines = [
        "# Kilonova simulation run configuration",
        "# Detailed OpSim metadata remain in the selected OpSim export directory.",
    ]
    for title, values in sections.items():
        entries = [(key, value) for key, value in values.items() if value is not None]
        if not entries:
            continue
        lines.extend(("", f"[{title}]"))
        lines.extend(f"{key}: {format_value(value)}" for key, value in entries)
    return "\n".join(lines) + "\n"


# =============================================================================
# 5. OPSIM EXPORT METADATA HELPERS
# Primary user: generate_kne_lightcurves.py.
#
# These helpers connect get_opsim_baseline.py exports to the generator while
# retaining compatibility with older exports that stored coordinates only in
# opsim_fields_index.csv.
# =============================================================================

def load_opsim_run_metadata(opsim_dir: Path) -> dict[str, Any]:
    """Read reproducibility metadata written by ``get_opsim_baseline.py``.

    The structured ``opsim_survey_summary.csv`` is preferred.  Selected values
    from the legacy ``run.txt`` file are retained as a backward-compatible
    supplement.  Missing metadata files do not prevent simulation; the returned
    mapping records which sources were available.
    """
    directory = Path(opsim_dir)
    metadata: dict[str, Any] = {
        "opsim_dir": str(directory),
        "opsim_dir_resolved": str(directory.resolve()),
        "opsim_survey_summary_found": False,
        "opsim_run_txt_found": False,
    }

    summary_file = directory / "opsim_survey_summary.csv"
    if summary_file.exists():
        try:
            survey_summary = pd.read_csv(summary_file)
        except Exception as exc:
            metadata["opsim_survey_summary_read_error"] = repr(exc)
            survey_summary = pd.DataFrame()
        if len(survey_summary):
            row = survey_summary.iloc[0]
            for column in survey_summary.columns:
                value = row[column]
                if pd.isna(value):
                    continue
                metadata[f"opsim_{column}"] = value.item() if hasattr(value, "item") else value
            metadata["opsim_survey_summary_found"] = True
            metadata["opsim_survey_summary_file"] = str(summary_file)

    run_file = directory / "run.txt"
    if run_file.exists():
        metadata["opsim_run_txt_found"] = True
        metadata["opsim_run_txt_file"] = str(run_file)
        try:
            for line in run_file.read_text(errors="replace").splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                clean_key = slugify(key).lower()
                if clean_key:
                    metadata[f"opsim_export_{clean_key}"] = value.strip()
        except Exception as exc:
            metadata["opsim_run_txt_read_error"] = repr(exc)
    return metadata


def normalize_opsim_visit_columns(visits: pd.DataFrame) -> pd.DataFrame:
    """Normalize compact and legacy OpSim visit schemas for internal use.

    New exports contain exactly ``MJD``, ``band`` and ``m5``.  Legacy exports
    used ``observationStartMJD`` and ``fiveSigmaDepth`` and may also contain
    repeated field metadata.  Downstream code uses the legacy canonical names
    internally so both file generations remain readable.
    """
    result = visits.copy()
    aliases = {
        "MJD": "observationStartMJD",
        "mjd": "observationStartMJD",
        "fiveSigmaDepth": "fiveSigmaDepth",
        "m5": "fiveSigmaDepth",
        "filter": "band",
    }
    for source, destination in aliases.items():
        if destination not in result.columns and source in result.columns:
            result = result.rename(columns={source: destination})

    required = {"observationStartMJD", "band", "fiveSigmaDepth"}
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(
            "OpSim visit table is missing required columns "
            f"{missing}; accepted compact schema is MJD,band,m5"
        )
    result["observationStartMJD"] = pd.to_numeric(
        result["observationStartMJD"], errors="coerce"
    )
    result["fiveSigmaDepth"] = pd.to_numeric(result["fiveSigmaDepth"], errors="coerce")
    result["band"] = result["band"].astype(str).str.strip().str.lower()
    return result


def enrich_visits_with_field_index_metadata(
    visits: pd.DataFrame,
    visits_file: Path,
    fields_index: pd.DataFrame,
) -> pd.DataFrame:
    """Restore per-field metadata from ``opsim_fields_index.csv``.

    Compact visit files contain only MJD, band and m5.  The field number is
    recovered from names such as ``opsim_visits_field0123.csv`` and joined to
    the index.  Existing usable metadata in legacy visit files is preserved.
    """
    result = visits.copy()
    if fields_index.empty or "field_index" not in fields_index.columns:
        return result

    match = re.search(r"field(\d+)", Path(visits_file).stem, flags=re.IGNORECASE)
    if match is None:
        return result
    field_number = int(match.group(1))
    index_values = pd.to_numeric(fields_index["field_index"], errors="coerce")
    selected = fields_index.loc[index_values == field_number]
    if selected.empty:
        return result
    row = selected.iloc[0]

    def first_finite(names: Sequence[str]) -> float:
        """Return the first finite coordinate found among compatible names."""
        for name in names:
            if name in selected.columns:
                value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
                if np.isfinite(value):
                    return float(value)
        return np.nan

    numeric_metadata = {
        "fieldRA": ("ra0_deg", "fieldRA", "ra_deg"),
        "fieldDec": ("dec0_deg", "fieldDec", "dec_deg"),
        "ebv_mw": ("ebv_mw", "ebv", "E_BV", "EBV"),
    }
    for band in "ugrizy":
        numeric_metadata[f"R_{band}_mw"] = (f"R_{band}_mw",)
        numeric_metadata[f"A_{band}_mw"] = (f"A_{band}_mw",)

    for destination, candidates in numeric_metadata.items():
        existing = pd.to_numeric(
            result.get(destination, pd.Series(np.nan, index=result.index)),
            errors="coerce",
        )
        if np.any(np.isfinite(existing)):
            continue
        value = first_finite(candidates)
        if np.isfinite(value):
            result[destination] = value

    existing_targets = result.get("target_name", pd.Series("", index=result.index))
    has_target = existing_targets.fillna("").astype(str).str.strip().ne("").any()
    if not has_target:
        for name in ("target_name", "sampled_target_name"):
            if name in selected.columns and pd.notna(row[name]):
                result["target_name"] = str(row[name])
                break
    return result


# =============================================================================
# 6. RUN AND PRODUCT DISCOVERY HELPERS
# Primary users: product-extraction, color-evolution, and plotting scripts.
#
# A run is recognized from summary.csv or its per-event light-curve CSV files.
# These helpers do not modify a run directory.
# =============================================================================

def looks_like_run_dir(path: Path) -> bool:
    """Return whether a directory contains recognizable pipeline products."""
    csv_dir = path / "csv"
    return (path / "summary.csv").exists() or (
        csv_dir.is_dir()
        and any(csv_dir.glob("lightcurve_*.csv"))
    )


def discover_run_dirs(root: Path) -> list[Path]:
    """Discover simulation runs at ``root`` and up to two levels below it."""
    root = root.expanduser().resolve()
    candidates: list[Path] = []
    if looks_like_run_dir(root):
        candidates.append(root)
    if root.is_dir():
        first_level = [path for path in root.iterdir() if path.is_dir()]
        candidates.extend(path for path in first_level if looks_like_run_dir(path))
        # The cleaned generator writes below runs/<readable_name>_seed_<seed>.
        # Search one additional level so the no-argument menus find those runs.
        for parent in first_level:
            try:
                candidates.extend(
                    path for path in parent.iterdir() if path.is_dir() and looks_like_run_dir(path)
                )
            except PermissionError:
                continue
    unique = {path.resolve(): path.resolve() for path in candidates}
    return sorted(unique.values(), key=lambda path: path.name.lower())


def choose_run_dir(run_dir: Path | None, runs_root: Path, interactive: bool) -> Path:
    """Resolve an explicit run or select one of the discovered runs."""
    if run_dir is not None:
        result = run_dir.expanduser().resolve()
        if not looks_like_run_dir(result):
            raise FileNotFoundError(f"Not a simulation run directory: {result}")
        return result
    choices = discover_run_dirs(runs_root)
    if not choices:
        raise FileNotFoundError(f"No simulation run found below {runs_root.expanduser().resolve()}")
    if len(choices) == 1 or not interactive:
        return choices[0]
    print("\nAvailable simulation runs:")
    for index, path in enumerate(choices, start=1):
        print(f"  {index}. {path.name}  ({path})")
    raw = input("Run number: ").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= len(choices):
        raise ValueError("Invalid run number")
    return choices[int(raw) - 1]


def find_synthetic_files(run_dir: Path) -> list[Path]:
    """List per-event synthetic CSV files while excluding LSST-like files."""
    csv_dir = run_dir / "csv"
    files = []
    for path in sorted(csv_dir.glob("lightcurve_*.csv")):
        if path.name.startswith("lightcurve_LSSTlike_"):
            continue
        if event_id_from_path(path) is not None:
            files.append(path)
    return files


def find_lsstlike_files(run_dir: Path) -> list[Path]:
    """List per-event LSST-like CSV files in supported legacy locations."""
    locations = [run_dir / "csv", run_dir]
    seen: dict[Path, Path] = {}
    for location in locations:
        for path in location.glob("lightcurve_LSSTlike_*.csv"):
            seen[path.resolve()] = path.resolve()
    return sorted(seen.values())


# =============================================================================
# 7. OPSIM/M5 SOURCE DISCOVERY AND READING
# Primary users: m5 product extraction and downstream diagnostics.
#
# Public entry points are resolve_opsim_source() and describe_m5_source().
# Functions beginning with an underscore implement the supported CSV,
# directory, and ZIP input formats.
# =============================================================================

def _opsim_paths_from_run_info(run_dir: Path) -> list[Path]:
    """Return OpSim paths recorded by the generator, most specific first."""
    run_info = run_dir / "run_info.txt"
    if not run_info.exists():
        return []
    wanted = ("opsim_dir_resolved", "opsim_dir")
    found: dict[str, str] = {}
    section = ""
    for line in run_info.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        # Accept both current "key: value" lines and legacy "- key: value".
        clean_key = key.strip().lstrip("-").strip().lower()
        if clean_key in wanted and value.strip():
            found[clean_key] = value.strip()
        # Runs produced before opsim_dir was standardized stored the path as
        # ``directory`` inside the opsim_context section.
        elif section == "opsim_context" and clean_key == "directory" and value.strip():
            found.setdefault("opsim_dir", value.strip())
    paths: list[Path] = []
    for key in wanted:
        value = found.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (run_dir / path).resolve()
        if path not in paths:
            paths.append(path)
    return paths


def _contains_opsim_csv(source: Path) -> bool:
    """Return whether a path contains at least one OpSim visit CSV."""
    if source.is_file() and source.suffix.lower() == ".csv":
        return True
    if source.is_file() and source.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(source) as archive:
                return any(
                    Path(name).name.startswith("opsim_visits_field") and name.lower().endswith(".csv")
                    for name in archive.namelist()
                )
        except zipfile.BadZipFile:
            return False
    if source.is_dir():
        return any(source.rglob("opsim_visits_field*.csv"))
    return False


def resolve_opsim_source(run_dir: Path, opsim_source: Path | None = None) -> Path | None:
    """Resolve an OpSim export CSV, directory, or ZIP.

    An explicit source is authoritative. Otherwise, paths stored in run_info.txt
    are tried. ``None`` means that the caller may use its legacy LSST-like
    fallback when available.
    """
    run_dir = run_dir.expanduser().resolve()
    if opsim_source is not None:
        explicit = opsim_source.expanduser().resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit OpSim source does not exist: {explicit}")
        if not _contains_opsim_csv(explicit):
            raise FileNotFoundError(
                f"No opsim_visits_field*.csv found in explicit OpSim source: {explicit}"
            )
        return explicit

    recorded = _opsim_paths_from_run_info(run_dir)
    for candidate in recorded:
        if candidate.exists() and _contains_opsim_csv(candidate):
            return candidate
    return None


def describe_m5_source(run_dir: Path, opsim_source: Path | None = None) -> tuple[str, Path | None]:
    """Describe the source that compute_m5_products will use without reading it."""
    resolved = resolve_opsim_source(run_dir, opsim_source)
    if resolved is not None:
        kind = "opsim_zip" if resolved.suffix.lower() == ".zip" else (
            "opsim_csv" if resolved.is_file() else "opsim_directory"
        )
        return kind, resolved
    if find_lsstlike_files(run_dir):
        return "lsstlike_fallback", None
    recorded = _opsim_paths_from_run_info(run_dir)
    detail = ""
    if recorded:
        detail = " Recorded but unavailable paths: " + ", ".join(map(str, recorded)) + "."
    raise FileNotFoundError(
        "No usable OpSim export and no lightcurve_LSSTlike_*.csv were found. "
        "Provide --opsim-source with an OpSim export directory, ZIP, or CSV."
        + detail
    )


def _read_m5_chunk(source: str, handle: str | Path | BinaryIO | TextIO) -> pd.DataFrame:
    """Read and normalize only the band and m5 columns from one CSV."""
    accepted = {"band", "filter", "m5", "fiveSigmaDepth"}
    frame = pd.read_csv(handle, usecols=lambda column: column in accepted)
    band_column = "band" if "band" in frame.columns else "filter" if "filter" in frame.columns else None
    m5_column = "m5" if "m5" in frame.columns else (
        "fiveSigmaDepth" if "fiveSigmaDepth" in frame.columns else None
    )
    if band_column is None or m5_column is None:
        raise ValueError(
            f"{source} must contain band/filter and m5/fiveSigmaDepth columns; "
            f"found {list(frame.columns)}"
        )
    bands = frame[band_column].astype(str).str.strip().str.lower()
    bands = bands.str.replace(r"^lsst", "", regex=True).str[-1:]
    return pd.DataFrame(
        {
            "band": bands,
            "m5": pd.to_numeric(frame[m5_column], errors="coerce"),
        }
    )


def _read_opsim_m5(source: Path) -> tuple[list[pd.DataFrame], int]:
    """Read normalized m5 chunks from a CSV, directory, or ZIP export."""
    chunks: list[pd.DataFrame] = []
    if source.is_dir():
        files = sorted(source.rglob("opsim_visits_field*.csv"))
        for path in files:
            chunks.append(_read_m5_chunk(str(path), path))
        return chunks, len(files)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if Path(name).name.startswith("opsim_visits_field") and name.lower().endswith(".csv")
            )
            for name in members:
                with archive.open(name) as handle:
                    chunks.append(_read_m5_chunk(f"{source}:{name}", handle))
        return chunks, len(members)
    chunks.append(_read_m5_chunk(str(source), source))
    return chunks, 1


# =============================================================================
# 8. FIESTA POSTERIOR LOADING AND SAMPLING
# Primary user: generate_kne_lightcurves.py in posterior-population mode.
#
# Complete rows are sampled with one common index across all physical
# parameters so posterior correlations are preserved.
# =============================================================================

def _read_posterior_npz(source: Path) -> dict[str, np.ndarray]:
    """Load every named array from an uncompressed posterior NPZ file."""
    with np.load(source, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_posterior_samples(source: Path) -> dict[str, np.ndarray]:
    """Load named FIESTA posterior arrays from .npz or a tar archive containing it."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".npz":
        arrays = _read_posterior_npz(source)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source, mode="r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith("posterior.npz")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one posterior.npz in {source}; found {len(members)}")
            extracted = tar.extractfile(members[0])
            if extracted is None:
                raise OSError(f"Could not read {members[0].name} from {source}")
            payload = io.BytesIO(extracted.read())
            with np.load(payload, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
    else:
        raise ValueError("Posterior input must be posterior.npz or a tar/tar.gz archive containing it")

    missing = [name for name in PHYSICAL_PARAMETERS if name not in arrays]
    if missing:
        raise KeyError(f"Posterior is missing physical parameters: {missing}")
    lengths = {len(np.ravel(arrays[name])) for name in PHYSICAL_PARAMETERS}
    if len(lengths) != 1:
        raise ValueError(f"Posterior physical arrays do not have a common length: {sorted(lengths)}")
    return {name: np.ravel(arrays[name]).astype(float, copy=False) for name in PHYSICAL_PARAMETERS}


def draw_posterior_rows(
    arrays: dict[str, np.ndarray], n_samples: int, rng: np.random.Generator
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Draw complete posterior rows while preserving parameter correlations.

    Sampling uses one common index for every physical parameter.  Rows are
    drawn without replacement when possible and with replacement only when the
    requested population exceeds the number of available posterior samples.
    """
    n_available = len(next(iter(arrays.values())))
    replace = n_samples > n_available
    indices = rng.choice(n_available, size=int(n_samples), replace=replace)
    rows = [
        {name: float(arrays[name][index]) for name in PHYSICAL_PARAMETERS}
        for index in indices
    ]
    return rows, np.asarray(indices, dtype=np.int64)


# =============================================================================
# 9. M5 SUMMARY PRODUCT HELPERS
# Primary users: product-extraction and survey-depth diagnostic scripts.
#
# Statistics are unweighted per visit.  OpSim exports are preferred; existing
# LSST-like event files are retained only as a backward-compatible fallback.
# =============================================================================

def compute_m5_products(
    run_dir: Path,
    percentile: float = 50.0,
    bands: Iterable[str] = BANDS,
    max_events: int | None = None,
    opsim_source: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute unweighted per-visit m5 summaries and band thresholds.

    OpSim exports are the preferred source and each exported visit row is read
    once. LSST-like event files remain a backward-compatible fallback only.
    ``max_events`` applies only to that fallback, never to OpSim field files.
    """
    if not 0.0 <= float(percentile) <= 100.0:
        raise ValueError(f"m5 percentile must be in [0, 100], got {percentile}")
    source_kind, resolved_source = describe_m5_source(run_dir, opsim_source)
    selected = set(bands)
    if resolved_source is not None:
        chunks, n_source_files = _read_opsim_m5(resolved_source)
        source_label = str(resolved_source)
    else:
        files = find_lsstlike_files(run_dir)
        if max_events is not None:
            files = files[: int(max_events)]
        chunks = [_read_m5_chunk(str(path), path) for path in files]
        n_source_files = len(files)
        source_label = str(run_dir / "csv" / "lightcurve_LSSTlike_*.csv")
    chunks = [
        frame.loc[frame["band"].isin(selected) & np.isfinite(frame["m5"]), ["band", "m5"]]
        for frame in chunks
    ]
    chunks = [frame for frame in chunks if len(frame)]
    if not chunks:
        raise RuntimeError("No finite m5 values found")
    visits = pd.concat(chunks, ignore_index=True)
    summary_rows = []
    threshold_rows = []
    for band in bands:
        values = visits.loc[visits["band"] == band, "m5"].to_numpy(float)
        if len(values) == 0:
            continue
        row = {
            "band": band,
            "N_visits": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "source_kind": source_kind,
            "source": source_label,
            "N_source_files": int(n_source_files),
        }
        for q in (1, 5, 16, 50, 84, 95, 99):
            row[f"p{q}"] = float(np.percentile(values, q))
        summary_rows.append(row)
        threshold_rows.append(
            {
                "band": band,
                "m5_threshold_percentile": float(percentile),
                "m5_threshold": float(np.percentile(values, percentile)),
                "N_visits": int(len(values)),
                "weighting": "none_per_visit",
                "source_kind": source_kind,
                "source": source_label,
                "N_source_files": int(n_source_files),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(threshold_rows)


def save_m5_products(run_dir: Path, summary: pd.DataFrame, thresholds: pd.DataFrame) -> Path:
    """Write the standard m5 summary tables and return the threshold path."""
    outdir = run_dir / "products" / "m5"
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "m5_summary_by_band.csv", index=False)
    thresholds.to_csv(outdir / "m5_thresholds_by_band.csv", index=False)
    return outdir / "m5_thresholds_by_band.csv"


# The colour-envelope and latitude-loss workflows use a richer, stable table
# containing p25, p50 and p75 thresholds for both the complete OpSim sample and
# (when field coordinates are available) high/low Galactic-latitude subsets.
M5_DEPTH_SCENARIOS = {
    "worst_p25": 0.25,
    "median_p50": 0.50,
    "best_p75": 0.75,
}


def galactic_latitude_deg(ra_deg: Sequence[float], dec_deg: Sequence[float]) -> np.ndarray:
    """Convert ICRS coordinates to Galactic latitude using the IAU rotation."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    xyz = np.vstack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))
    )
    rotation = np.array(
        [
            [-0.0548755604, -0.8734370902, -0.4838350155],
            [0.4941094279, -0.4448296300, 0.7469822445],
            [-0.8676661490, -0.1980763734, 0.4559837762],
        ]
    )
    galactic_xyz = rotation @ xyz
    return np.rad2deg(np.arcsin(np.clip(galactic_xyz[2], -1.0, 1.0)))


def _opsim_field_number(path: Path) -> int:
    match = re.search(r"field[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot determine field number from {path.name}")
    return int(match.group(1))


def _opsim_latitude_groups(
    opsim_dir: Path, latitude_threshold_deg: float
) -> dict[int, str] | None:
    index_path = opsim_dir / "opsim_fields_index.csv"
    if not index_path.exists():
        return None
    table = pd.read_csv(index_path)
    required = {"field_index", "ra0_deg", "dec0_deg"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{index_path} is missing columns {sorted(missing)}")
    field_numbers = pd.to_numeric(table["field_index"], errors="raise").astype(int)
    latitude = galactic_latitude_deg(
        pd.to_numeric(table["ra0_deg"], errors="coerce").to_numpy(float),
        pd.to_numeric(table["dec0_deg"], errors="coerce").to_numpy(float),
    )
    groups = np.where(np.abs(latitude) >= float(latitude_threshold_deg), "high", "low")
    return {
        int(number): str(group)
        for number, group, keep in zip(field_numbers, groups, np.isfinite(latitude))
        if keep
    }


def compute_m5_depth_quantiles(
    opsim_dir: Path,
    latitude_threshold_deg: float = 20.0,
) -> pd.DataFrame:
    """Compute visit-weighted p25/p50/p75 m5 thresholds from an OpSim export."""
    opsim_dir = Path(opsim_dir).expanduser().resolve()
    if not 0.0 <= float(latitude_threshold_deg) <= 90.0:
        raise ValueError("latitude_threshold_deg must lie in [0, 90]")
    files = sorted(opsim_dir.glob("opsim_visits_field*.csv"))
    if not files:
        raise FileNotFoundError(f"No opsim_visits_field*.csv found in {opsim_dir}")

    latitude_groups = _opsim_latitude_groups(opsim_dir, latitude_threshold_deg)
    scopes = ("global", "high", "low") if latitude_groups is not None else ("global",)
    values = {scope: {band: [] for band in BANDS} for scope in scopes}

    for path in files:
        table = _read_m5_chunk(str(path), path)
        latitude_group = None
        if latitude_groups is not None:
            latitude_group = latitude_groups.get(_opsim_field_number(path))
            if latitude_group not in {"high", "low"}:
                raise ValueError(f"No Galactic-latitude metadata for {path.name}")
        for band in BANDS:
            band_values = table.loc[
                (table["band"] == band) & np.isfinite(table["m5"]), "m5"
            ].to_numpy(float)
            if band_values.size:
                values["global"][band].append(band_values)
                if latitude_group is not None:
                    values[latitude_group][band].append(band_values)

    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for band in BANDS:
            chunks = values[scope][band]
            if not chunks:
                continue
            depths = np.concatenate(chunks)
            for scenario, quantile in M5_DEPTH_SCENARIOS.items():
                rows.append(
                    {
                        "threshold_scope": (
                            "common_global" if scope == "global" else "latitude_conditioned"
                        ),
                        "threshold_latitude_group": "both" if scope == "global" else scope,
                        "band": band,
                        "scenario": scenario,
                        "quantile": quantile,
                        "m5": float(np.quantile(depths, quantile)),
                        "N_visits": int(depths.size),
                        "quantile_weighting": "one weight per OpSim visit",
                        "latitude_threshold_deg": (
                            np.nan if scope == "global" else float(latitude_threshold_deg)
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No finite m5 values were found")
    return result


def write_m5_depth_quantiles(
    opsim_dir: Path,
    latitude_threshold_deg: float = 20.0,
    output_file: Path | None = None,
) -> Path:
    """Compute and save the canonical m5 table; return its absolute path."""
    opsim_dir = Path(opsim_dir).expanduser().resolve()
    destination = (
        Path(output_file).expanduser().resolve()
        if output_file is not None
        else opsim_dir / "m5_depth_quantiles_by_band.csv"
    )
    table = compute_m5_depth_quantiles(opsim_dir, latitude_threshold_deg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination


def discover_m5_quantile_table(
    run_dir: Path,
    explicit: Path | None = None,
    opsim_source: Path | None = None,
) -> Path:
    """Find the canonical m5 table without requiring a copy inside each run."""
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    run_dir = Path(run_dir).expanduser().resolve()
    candidates = [
        run_dir / "m5_depth_quantiles_by_band.csv",
        run_dir / "analysis" / "latitude_depth_comparison" / "m5_depth_quantiles_by_band.csv",
    ]
    if opsim_source is not None:
        source = Path(opsim_source).expanduser().resolve()
        candidates.append(source / "m5_depth_quantiles_by_band.csv" if source.is_dir() else source.parent / "m5_depth_quantiles_by_band.csv")
    for source in _opsim_paths_from_run_info(run_dir):
        candidates.append(source / "m5_depth_quantiles_by_band.csv" if source.is_dir() else source.parent / "m5_depth_quantiles_by_band.csv")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "m5_depth_quantiles_by_band.csv was not found in the run or its recorded "
        "OpSim export. Re-run get_opsim_baseline.py or pass --m5-quantile-table."
    )


# =============================================================================
# 10. STATISTICAL HELPERS
# Primary users: downstream plotting and distribution-summary scripts.
# =============================================================================

def bootstrap_percentiles_unweighted(
    values: np.ndarray,
    percentiles: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate percentiles and their 16--84% bootstrap half-widths."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        empty = np.full(len(percentiles), np.nan)
        return empty, empty
    estimate = np.percentile(values, percentiles)
    if n_boot <= 0 or len(values) < 2:
        return estimate, np.full(len(percentiles), np.nan)
    boot = np.empty((int(n_boot), len(percentiles)), dtype=float)
    for index in range(int(n_boot)):
        draw = values[rng.integers(0, len(values), size=len(values))]
        boot[index] = np.percentile(draw, percentiles)
    low, high = np.percentile(boot, [16, 84], axis=0)
    return estimate, 0.5 * (high - low)
