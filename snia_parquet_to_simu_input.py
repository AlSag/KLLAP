#!/usr/bin/env python3
"""Convert a synthetic SN Ia Parquet population to BOOM LSST simulation inputs.

The conversion applies an m5 detection cut while preserving the SN Ia errors:

* the input ``mag`` and ``magerr`` values are used directly;
* points with ``mag <= m5`` are detections;
* fainter points are retained as upper limits (empty ``mag``, ``mag_err=m5``);
* rows from simultaneous bands are separated by a small timestamp shift because
  the BOOM simulator keeps only one alert per exact ``time_mjd``;
* one ``lightcurve_NNNN.csv`` is written per SN Ia, together with ``summary.csv``
  and ``run_info.txt``.

Run without arguments for the interactive terminal interface, or use --help for
command-line options.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "sn_id",
    "mjd",
    "band",
    "mag",
    "magerr",
    "ra",
    "dec",
}
ALLOWED_BANDS = ("u", "g", "r", "i", "z", "y")
BAND_ORDER = {band: index for index, band in enumerate(ALLOWED_BANDS)}
M5 = {"u": 23.7, "g": 24.8, "r": 24.4, "i": 23.9, "z": 23.3, "y": 22.3}
SUMMARY_METADATA = (
    "redshift",
    "distance_mpc",
    "comoving_distance_mpc",
    "distmod",
    "t0",
    "x0",
    "x1",
    "c",
    "reference_mjd",
    "salt2_tmax_mjd",
    "m_abs_b",
    "exposure_tier",
    "exptime_seconds",
    "model",
    "population_model_version",
    "master_seed",
)


def prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def finite_mask(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").map(math.isfinite)


# ----------------------------------------------------------------------------
# Read input photometry and apply selection
# ----------------------------------------------------------------------------

def load_population(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and validate the SN Ia long-format population."""
    try:
        table = pd.read_parquet(path)
    except ImportError as exc:
        raise SystemExit(
            "Reading Parquet requires pyarrow or fastparquet. In the FIESTA "
            "environment, install pyarrow if it is not already available."
        ) from exc

    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise SystemExit(f"Input Parquet misses columns: {sorted(missing)}")

    original_rows = len(table)
    table = table.copy()
    table["band"] = table["band"].astype("string").str.strip().str.lower()

    valid = (
        table["sn_id"].notna()
        & finite_mask(table["sn_id"])
        & finite_mask(table["mjd"])
        & finite_mask(table["mag"])
        & finite_mask(table["magerr"])
        & (pd.to_numeric(table["magerr"], errors="coerce") > 0)
        & finite_mask(table["ra"])
        & finite_mask(table["dec"])
        & table["band"].isin(ALLOWED_BANDS)
    )
    table = table.loc[valid].copy()
    if table.empty:
        raise SystemExit("No valid SN Ia photometry remains after schema checks")

    numeric_ids = pd.to_numeric(table["sn_id"], errors="raise")
    if not np.all(np.isclose(numeric_ids, np.rint(numeric_ids))):
        raise SystemExit("sn_id must contain integer-like values")
    table["sn_id"] = np.rint(numeric_ids).astype(np.int64)
    table["mjd"] = pd.to_numeric(table["mjd"], errors="raise").astype(float)
    table["mag"] = pd.to_numeric(table["mag"], errors="raise").astype(float)
    table["magerr"] = pd.to_numeric(table["magerr"], errors="raise").astype(float)
    table["ra"] = pd.to_numeric(table["ra"], errors="raise").astype(float)
    table["dec"] = pd.to_numeric(table["dec"], errors="raise").astype(float)

    table["_source_row"] = np.arange(len(table), dtype=np.int64)
    table["_band_order"] = table["band"].map(BAND_ORDER).astype(int)
    table = table.sort_values(
        ["sn_id", "mjd", "_band_order", "_source_row"], kind="stable"
    ).reset_index(drop=True)

    diagnostics = {
        "input_rows": int(original_rows),
        "valid_rows": int(len(table)),
        "discarded_invalid_rows": int(original_rows - len(table)),
        "events": int(table["sn_id"].nunique()),
    }
    return table, diagnostics


def apply_m5_cut(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the m5 cut while preserving the input SN Ia uncertainties."""
    out = table.copy()
    m5 = out["band"].map(M5).astype(float)
    detected = out["mag"] <= m5

    out["m5"] = m5
    out["output_mag"] = out["mag"].where(detected)
    out["output_mag_err"] = out["magerr"].where(detected, m5)
    out["_detected"] = detected
    return out


# ----------------------------------------------------------------------------
# Prepare timestamps and export BOOM tables
# ----------------------------------------------------------------------------

def add_boom_fields(table: pd.DataFrame, shift_seconds: float) -> pd.DataFrame:
    """Create BOOM columns and separate simultaneous-band timestamps."""
    if not math.isfinite(shift_seconds) or shift_seconds <= 0:
        raise SystemExit("Timestamp shift must be a positive finite number")

    out = table.copy()
    rank = out.groupby(["sn_id", "mjd"], sort=False).cumcount()
    shift_days = rank.to_numpy(dtype=float) * shift_seconds / 86400.0
    out["time_mjd"] = out["mjd"].to_numpy(dtype=float) + shift_days

    duplicated = out.duplicated(["sn_id", "time_mjd"], keep=False)
    if duplicated.any():
        raise SystemExit(
            "Repeated BOOM timestamps remain after shifting simultaneous bands"
        )

    # Prevent a shifted simultaneous group from crossing the following original
    # observing epoch for the same object.
    epochs = out[["sn_id", "mjd"]].drop_duplicates().copy()
    epochs["_next_original_mjd"] = epochs.groupby("sn_id", sort=False)["mjd"].shift(-1)
    next_epoch_lookup = epochs.set_index(["sn_id", "mjd"])["_next_original_mjd"]
    next_original = pd.Series(
        next_epoch_lookup.reindex(
            pd.MultiIndex.from_frame(out[["sn_id", "mjd"]])
        ).to_numpy(),
        index=out.index,
        dtype=float,
    )
    crosses_next_epoch = next_original.notna() & (out["time_mjd"] >= next_original)
    if crosses_next_epoch.any():
        raise SystemExit(
            "Timestamp shifts overlap the following observing epoch. Reduce "
            "--shift-seconds."
        )

    if "phase_since_reference" in out.columns:
        base_phase = pd.to_numeric(out["phase_since_reference"], errors="coerce")
    elif "reference_mjd" in out.columns:
        base_phase = out["mjd"] - pd.to_numeric(
            out["reference_mjd"], errors="coerce"
        )
    else:
        base_phase = out["mjd"] - out.groupby("sn_id", sort=False)["mjd"].transform("min")
    out["t_days"] = base_phase.to_numpy(dtype=float) + shift_days

    # BOOM's simulated-LSST producer requires ``<object>_<sequence>`` and parses
    # the integer after the first underscore. Match the KNe convention exactly,
    # e.g. 0000_1, 0000_2, ..., 0500_1.
    width = max(4, len(str(int(out["sn_id"].max()))))
    sequence = out.groupby("sn_id", sort=False).cumcount() + 1
    out["candidate_id"] = [
        f"{int(sn_id):0{width}d}_{int(seq)}"
        for sn_id, seq in zip(out["sn_id"], sequence)
    ]
    out["_was_shifted"] = rank > 0
    return out


def build_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Build the object lookup and expected-alert counts required by BOOM."""
    first = table.groupby("sn_id", sort=True).first()
    counts = table.groupby("sn_id", sort=True)["_detected"].sum().rename("n_detected")

    summary = pd.DataFrame(index=first.index)
    summary["i"] = summary.index.astype(np.int64)
    summary["objectid"] = summary.index.astype(np.int64)
    summary["fieldRA"] = first["ra"].astype(float)
    summary["fieldDec"] = first["dec"].astype(float)
    width = max(4, len(str(int(summary.index.max()))))
    summary["target_name"] = [f"SNIa_{int(sn_id):0{width}d}" for sn_id in summary.index]
    summary["n_detected"] = counts.astype(np.int64)

    for column in SUMMARY_METADATA:
        if column in first.columns:
            summary[column] = first[column]

    return summary.reset_index(drop=True)


def write_run_info(
    output_path: Path,
    input_path: Path,
    table: pd.DataFrame,
    diagnostics: dict[str, int],
    shift_seconds: float,
) -> None:
    """Write a compact, machine-readable provenance file."""
    info = configparser.ConfigParser(interpolation=None)
    info["run"] = {
        "name": output_path.name,
        "population": "SNIa",
        "n_events": str(diagnostics["events"]),
        "n_rows": str(diagnostics["valid_rows"]),
    }
    info["input"] = {
        "source_parquet": str(input_path.resolve()),
        "model": str(table["model"].iloc[0]) if "model" in table.columns else "unknown",
        "population_model_version": (
            str(table["population_model_version"].iloc[0])
            if "population_model_version" in table.columns
            else "unknown"
        ),
    }
    info["boom_export"] = {
        "selection": "input_mag_le_m5_with_upper_limits",
        "magnitude": "input_parquet_mag",
        "magnitude_error": "input_parquet_magerr_for_detections",
        "upper_limits": "mag_empty_and_mag_err_equal_to_m5",
        "m5": ",".join(f"{band}:{M5[band]:g}" for band in ALLOWED_BANDS),
        "timestamp_shift_seconds": f"{shift_seconds:g}",
    }
    with (output_path / "run_info.txt").open("w", encoding="utf-8") as handle:
        info.write(handle)


def export_population(
    input_path: Path,
    output_path: Path,
    shift_seconds: float,
    overwrite: bool,
) -> dict[str, object]:
    """Convert the full Parquet population and write BOOM input files."""
    if not input_path.is_file():
        raise SystemExit(f"Input Parquet does not exist: {input_path}")

    existing = list(output_path.glob("lightcurve_*.csv")) if output_path.is_dir() else []
    existing += [
        path for path in (output_path / "summary.csv", output_path / "run_info.txt")
        if path.exists()
    ]
    if existing and not overwrite:
        raise SystemExit(
            f"{output_path} already contains BOOM outputs. Use --overwrite to replace them."
        )
    # Validate the replacement before touching an existing export.
    raw, diagnostics = load_population(input_path)
    table = apply_m5_cut(add_boom_fields(raw, shift_seconds))
    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()

    summary = build_summary(table)
    width = max(4, len(str(int(table["sn_id"].max()))))

    output_columns = [
        "objectid",
        "candidate_id",
        "time_mjd",
        "t_days",
        "band",
        "mag",
        "mag_err",
    ]
    for sn_id, rows in table.groupby("sn_id", sort=True):
        lightcurve = pd.DataFrame(
            {
                "objectid": pd.Series([pd.NA] * len(rows), dtype="Int64"),
                "candidate_id": rows["candidate_id"].astype(str).to_numpy(),
                "time_mjd": rows["time_mjd"].to_numpy(dtype=float),
                "t_days": rows["t_days"].to_numpy(dtype=float),
                "band": rows["band"].astype(str).to_numpy(),
                "mag": rows["output_mag"].to_numpy(dtype=float),
                "mag_err": rows["output_mag_err"].to_numpy(dtype=float),
            }
        )
        lightcurve.loc[0, "objectid"] = int(sn_id)
        lightcurve[output_columns].to_csv(
            output_path / f"lightcurve_{int(sn_id):0{width}d}.csv", index=False
        )

    summary.to_csv(output_path / "summary.csv", index=False)
    write_run_info(output_path, input_path, table, diagnostics, shift_seconds)

    report: dict[str, object] = {
        **diagnostics,
        "output_directory": str(output_path.resolve()),
        "lightcurve_files": int(table["sn_id"].nunique()),
        "output_photometry_rows": int(len(table)),
        "timestamp_shift_seconds": float(shift_seconds),
        "shifted_rows": int(table["_was_shifted"].sum()),
        "detections": int(table["_detected"].sum()),
        "upper_limits": int((~table["_detected"]).sum()),
        "selection": "input_mag_le_m5_with_upper_limits",
        "magnitude_error": "input_parquet_magerr_for_detections",
    }
    (output_path / "conversion_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


# ----------------------------------------------------------------------------
# Terminal interface
# ----------------------------------------------------------------------------

def configure_interactively(args: argparse.Namespace) -> argparse.Namespace:
    print("\nSN Ia Parquet -> BOOM input")
    print("Input mag and magerr are preserved; detections require mag <= m5.")
    print("Sub-threshold points are written as upper limits.\n")

    default_input = str(args.input or Path(
        "snia_salt2_comoving_100_500mpc_1000_rubin_limits_6band.parquet"
    ))
    args.input = Path(prompt_text("Input SN Ia Parquet", default_input)).expanduser()

    default_output = str(args.output or Path("snia_boom_input"))
    args.output = Path(prompt_text("Output BOOM directory", default_output)).expanduser()

    shift_text = prompt_text(
        "Shift between simultaneous bands [seconds]",
        f"{args.shift_seconds:g}",
    )
    try:
        args.shift_seconds = float(shift_text)
    except ValueError as exc:
        raise ValueError("Timestamp shift must be numeric") from exc

    if args.output.exists() and any(args.output.iterdir()):
        answer = prompt_text("Replace existing BOOM output files? (y/N)", "N")
        args.overwrite = answer.strip().lower() in {"y", "yes", "o", "oui"}
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SN Ia photometry to BOOM inputs using the input mag and "
            "magerr values, an m5 cut and upper limits."
        )
    )
    parser.add_argument("--input", type=Path, help="Input SN Ia Parquet file")
    parser.add_argument("--output", type=Path, help="Output BOOM input directory")
    parser.add_argument(
        "--shift-seconds",
        type=float,
        default=30.0,
        help="Shift between simultaneous bands (default: 30 seconds)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing lightcurve CSVs, summary.csv and run_info.txt",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for paths and settings in the terminal",
    )
    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        try:
            args = configure_interactively(args)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            return 130
        except ValueError as exc:
            parser.error(str(exc))
    elif args.input is None or args.output is None:
        parser.error("--input and --output are required outside interactive mode")

    report = export_population(
        input_path=args.input,
        output_path=args.output,
        shift_seconds=args.shift_seconds,
        overwrite=args.overwrite,
    )
    print("\nSN Ia BOOM export completed")
    print("=" * 29)
    print(f"Events                         : {report['lightcurve_files']}")
    print(f"Photometry rows                : {report['output_photometry_rows']}")
    print(f"Detections (expected alerts)   : {report['detections']}")
    print(f"Upper limits                   : {report['upper_limits']}")
    print(f"Discarded invalid rows         : {report['discarded_invalid_rows']}")
    print(f"Rows shifted in time           : {report['shifted_rows']}")
    print(f"Output directory               : {report['output_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
