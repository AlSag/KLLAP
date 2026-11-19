#!/usr/bin/env python3
"""Audit a BOOM colour-filter run and compute conditional efficiency.

The input population can be restricted to an inclusive file-index interval,
for example ``1000..1999`` for files ``lightcurve_1000.csv`` through
``lightcurve_1999.csv``.

The script distinguishes four outcomes for the requested simulated population:

* passed_filter: at least one valid input photometry point and a stats.csv row
  with n_passed_filter > 0;
* rejected_filter: valid input photometry exists, but no filter pass is recorded;
* no_valid_photometry: the input CSV exists but has no usable photometry row;
* missing_input_file: an expected numbered light-curve CSV is absent.

The reported ``conditional_filter_efficiency`` excludes the last two categories
from its denominator. ``global_efficiency`` keeps the full requested population
in the denominator and therefore includes photometric availability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


ID_COLUMNS = ("object_id", "objectid", "event_id", "injected_event_index")
TIME_COLUMNS = ("time_mjd", "mjd", "jd")
MAG_COLUMNS = ("mag", "magpsf", "mag_psf")
MAG_ERROR_COLUMNS = ("mag_err", "sigmapsf", "magpsf_err")
FLUX_COLUMNS = ("flux_njy", "psfFlux", "psf_flux")
FLUX_ERROR_COLUMNS = ("flux_err_njy", "psfFluxErr", "psf_flux_err")
BAND_COLUMNS = ("band", "filter", "fid")


def finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(str(value).strip()))
    except (TypeError, ValueError):
        return False


def positive_finite_number(value: object) -> bool:
    return finite_number(value) and float(str(value).strip()) > 0


def canonical_id(value: object) -> str | None:
    """Normalize integer-like identifiers while preserving other strings."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def first_existing(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def file_index(path: Path) -> str | None:
    match = re.search(r"(\d+)$", path.stem)
    return str(int(match.group(1))) if match else None


# ----------------------------------------------------------------------------
# Read and validate BOOM products
# ----------------------------------------------------------------------------

def inspect_lightcurve(path: Path) -> dict[str, object]:
    """Return the object ID and number of rows BOOM can use."""
    fallback_id = file_index(path)
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return {
                    "object_id": fallback_id,
                    "file_index": fallback_id,
                    "n_rows": 0,
                    "n_valid_points": 0,
                    "input_file": str(path),
                    "input_issue": "missing_header",
                }
            columns = [name.strip() for name in reader.fieldnames]
            id_column = first_existing(columns, ID_COLUMNS)
            time_column = first_existing(columns, TIME_COLUMNS)
            mag_column = first_existing(columns, MAG_COLUMNS)
            mag_error_column = first_existing(columns, MAG_ERROR_COLUMNS)
            flux_column = first_existing(columns, FLUX_COLUMNS)
            flux_error_column = first_existing(columns, FLUX_ERROR_COLUMNS)
            band_column = first_existing(columns, BAND_COLUMNS)
            object_id = None
            n_rows = 0
            n_valid = 0
            for row in reader:
                n_rows += 1
                if object_id is None and id_column:
                    object_id = canonical_id(row.get(id_column))

                time_ok = time_column is None or finite_number(row.get(time_column))
                band_ok = band_column is None or bool(str(row.get(band_column, "")).strip())

                mag_ok = mag_column is not None and finite_number(row.get(mag_column))
                if mag_ok and mag_error_column is not None:
                    mag_ok = positive_finite_number(row.get(mag_error_column))

                flux_ok = flux_column is not None and finite_number(row.get(flux_column))
                if flux_ok and flux_error_column is not None:
                    flux_ok = positive_finite_number(row.get(flux_error_column))

                # A row needs a finite time and band when those columns exist,
                # plus either valid magnitude photometry or valid flux photometry.
                if time_ok and band_ok and (mag_ok or flux_ok):
                    n_valid += 1

            return {
                "object_id": object_id or fallback_id,
                "file_index": fallback_id,
                "n_rows": n_rows,
                "n_valid_points": n_valid,
                "input_file": str(path),
                "input_issue": "" if n_valid else "no_usable_photometry_row",
            }
    except (OSError, csv.Error, UnicodeError) as exc:
        return {
            "object_id": fallback_id,
            "file_index": fallback_id,
            "n_rows": 0,
            "n_valid_points": 0,
            "input_file": str(path),
            "input_issue": f"read_error: {exc}",
        }


def load_filter_stats(path: Path) -> dict[str, int]:
    """Return object_id -> n_passed_filter for accepted BOOM objects."""
    accepted: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        columns = [name.strip() for name in reader.fieldnames]
        id_column = first_existing(columns, ID_COLUMNS)
        if id_column is None:
            raise ValueError(
                f"{path} needs one of these ID columns: {', '.join(ID_COLUMNS)}"
            )
        pass_column = "n_passed_filter" if "n_passed_filter" in columns else None
        for row in reader:
            object_id = canonical_id(row.get(id_column))
            if object_id is None:
                continue
            if pass_column:
                try:
                    n_passed = int(float(str(row.get(pass_column, 0)).strip() or 0))
                except ValueError:
                    n_passed = 0
            else:
                n_passed = 1
            if n_passed > 0:
                accepted[object_id] = accepted.get(object_id, 0) + n_passed
    return accepted


def match_stats_id(record: dict[str, object], accepted_ids: set[str]) -> str | None:
    candidates = [
        canonical_id(record.get("object_id")),
        canonical_id(record.get("file_index")),
    ]
    # BOOM may replace a false-like object ID 0 with the fallback ID 1000000.
    if "0" in candidates:
        candidates.append("1000000")
    matches = [candidate for candidate in candidates if candidate in accepted_ids]
    return matches[0] if matches else None


def percentage(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


# ----------------------------------------------------------------------------
# Terminal configuration and audit orchestration
# ----------------------------------------------------------------------------

def prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def configure_interactively(args: argparse.Namespace) -> argparse.Namespace:
    """Collect paths and the inclusive input-index interval to audit."""
    print("\nBOOM colour-filter efficiency")
    print("Provide the inputs from the same BOOM run.\n")

    default_input = str(args.input_dir or Path("boom_input"))
    args.input_dir = Path(prompt_text(
        "BOOM input directory containing lightcurve_*.csv",
        default_input,
    )).expanduser()

    default_stats = str(args.filter_stats or Path("stats.csv"))
    args.filter_stats = Path(prompt_text(
        "BOOM filter stats.csv",
        default_stats,
    )).expanduser()

    available_indices = []
    if args.input_dir.is_dir():
        available_indices = [
            int(index)
            for path in args.input_dir.glob(args.pattern)
            if (index := file_index(path)) is not None
        ]

    default_start = args.start_index if args.start_index is not None else 0
    if args.end_index is not None:
        default_end = args.end_index
    elif args.expected_events is not None:
        default_end = default_start + args.expected_events - 1
    elif available_indices:
        default_end = max(available_indices)
    else:
        default_end = 99

    try:
        args.start_index = int(prompt_text(
            "First input light-curve index processed by BOOM",
            str(default_start),
        ))
        args.end_index = int(prompt_text(
            "Last input light-curve index processed by BOOM (inclusive)",
            str(default_end),
        ))
    except ValueError as exc:
        raise ValueError(
            "First and last input light-curve indices must be integers"
        ) from exc

    # The interval is now authoritative; keep this value only for backward
    # compatibility with the former summary field and command-line option.
    args.expected_events = args.end_index - args.start_index + 1

    default_output_dir = (
        args.filter_stats.parent
        if args.filter_stats.parent != Path("")
        else Path.cwd()
    )
    output_dir = Path(prompt_text(
        "Output directory",
        str(default_output_dir),
    )).expanduser()
    args.audit_output = output_dir / "filter_efficiency_audit.csv"
    args.summary_output = output_dir / "filter_efficiency_summary.json"
    print()
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute colour-filter efficiency conditional on at least one valid "
            "BOOM input photometry point."
        )
    )
    parser.add_argument("--input-dir", type=Path,
                        help="Directory containing lightcurve_*.csv BOOM inputs")
    parser.add_argument("--filter-stats", type=Path,
                        help="stats.csv produced by the BOOM filter run")
    parser.add_argument("--expected-events", type=int, default=None,
                        help=(
                            "Backward-compatible population size. With "
                            "--start-index S and no --end-index, selects "
                            "S through S+N-1"
                        ))
    parser.add_argument("--start-index", type=int, default=None,
                        help="First processed light-curve file index (inclusive)")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Last processed light-curve file index (inclusive)")
    parser.add_argument("--pattern", default="lightcurve_*.csv",
                        help="Input filename pattern (default: lightcurve_*.csv)")
    parser.add_argument("--audit-output", type=Path,
                        default=Path("filter_efficiency_audit.csv"))
    parser.add_argument("--summary-output", type=Path,
                        default=Path("filter_efficiency_summary.json"))
    parser.add_argument("--interactive", action="store_true",
                        help="Ask for paths and population size in the terminal")
    args = parser.parse_args()

    interactive = args.interactive or len(sys.argv) == 1
    if interactive:
        try:
            args = configure_interactively(args)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            return 130
        except ValueError as exc:
            parser.error(str(exc))
    elif args.input_dir is None or args.filter_stats is None:
        parser.error(
            "--input-dir and --filter-stats are required outside interactive mode"
        )

    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if not args.filter_stats.is_file():
        parser.error(f"filter stats file does not exist: {args.filter_stats}")
    if args.expected_events is not None and args.expected_events <= 0:
        parser.error("--expected-events must be positive")

    start_index = args.start_index if args.start_index is not None else 0
    if args.end_index is not None:
        end_index = args.end_index
        if args.expected_events is not None:
            implied_count = end_index - start_index + 1
            if implied_count != args.expected_events:
                parser.error(
                    "--expected-events disagrees with the inclusive interval "
                    "defined by --start-index and --end-index"
                )
    elif args.expected_events is not None:
        end_index = start_index + args.expected_events - 1
    else:
        numbered_indices = [
            int(index)
            for path in args.input_dir.glob(args.pattern)
            if (index := file_index(path)) is not None
        ]
        if not numbered_indices:
            parser.error(
                f"no numbered files match {args.pattern!r} in {args.input_dir}"
            )
        end_index = max(numbered_indices)

    if start_index < 0:
        parser.error("--start-index must be zero or positive")
    if end_index < start_index:
        parser.error("--end-index must be greater than or equal to --start-index")

    expected = end_index - start_index + 1

    available_paths = sorted(
        args.input_dir.glob(args.pattern),
        key=lambda path: (
            int(file_index(path)) if file_index(path) is not None else sys.maxsize,
            path.name,
        ),
    )
    if not available_paths:
        parser.error(f"no files match {args.pattern!r} in {args.input_dir}")

    # Select the same inclusive numbered interval that BOOM processed. This
    # supports chunks such as 0..999, 1000..1999, etc., without copying files.
    paths = [
        path for path in available_paths
        if (index := file_index(path)) is not None
        and start_index <= int(index) <= end_index
    ]

    accepted = load_filter_stats(args.filter_stats)
    accepted_ids = set(accepted)
    records = [inspect_lightcurve(path) for path in paths]

    seen_file_indices = {
        canonical_id(record["file_index"])
        for record in records
        if canonical_id(record["file_index"]) is not None
    }
    for index in range(start_index, end_index + 1):
        index_text = str(index)
        if index_text not in seen_file_indices:
            records.append({
                "object_id": index_text,
                "file_index": index_text,
                "n_rows": 0,
                "n_valid_points": 0,
                "input_file": "",
                "input_issue": "missing_input_file",
            })

    matched_accepted_ids: set[str] = set()
    for record in records:
        matched_id = match_stats_id(record, accepted_ids)
        if matched_id is not None:
            matched_accepted_ids.add(matched_id)
        valid = int(record["n_valid_points"]) > 0
        passed = matched_id is not None
        record["stats_object_id"] = matched_id or ""
        record["n_passed_filter"] = accepted.get(matched_id, 0) if matched_id else 0
        if not valid:
            record["status"] = (
                "missing_input_file"
                if record["input_issue"] == "missing_input_file"
                else "no_valid_photometry"
            )
        elif passed:
            record["status"] = "passed_filter"
        else:
            record["status"] = "rejected_filter"

    unmatched_accepted = sorted(accepted_ids - matched_accepted_ids)
    if unmatched_accepted:
        print(
            "ERROR: accepted stats IDs could not be matched to BOOM input files: "
            + ", ".join(unmatched_accepted),
            file=sys.stderr,
        )
        print(
            "Check that --input-dir and --filter-stats belong to the same run.",
            file=sys.stderr,
        )
        return 2

    n_available_files = len(available_paths)
    n_files = len(paths)
    n_evaluable = sum(record["status"] in {"passed_filter", "rejected_filter"}
                      for record in records)
    n_passed = sum(record["status"] == "passed_filter" for record in records)
    n_rejected = sum(record["status"] == "rejected_filter" for record in records)
    n_no_points = sum(record["status"] == "no_valid_photometry" for record in records)
    n_missing = sum(record["status"] == "missing_input_file" for record in records)

    conditional_efficiency = percentage(n_passed, n_evaluable)
    global_efficiency = percentage(n_passed, expected)
    summary = {
        "selection_start_index": start_index,
        "selection_end_index_inclusive": end_index,
        "expected_events": expected,
        "input_files_available_in_directory": n_available_files,
        "input_files_found": n_files,
        "evaluable_lightcurves_with_valid_points": n_evaluable,
        "lightcurves_without_valid_points": n_no_points,
        "missing_input_files": n_missing,
        "passed_filter": n_passed,
        "rejected_filter_among_evaluable": n_rejected,
        "conditional_filter_efficiency_percent": conditional_efficiency,
        "global_efficiency_percent": global_efficiency,
        "conditional_efficiency_definition": (
            "passed_filter / evaluable_lightcurves_with_valid_points"
        ),
        "global_efficiency_definition": "passed_filter / expected_events",
    }

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_columns = [
        "object_id", "file_index", "stats_object_id", "input_file",
        "n_rows", "n_valid_points", "n_passed_filter", "status", "input_issue",
    ]
    with args.audit_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: int(row["file_index"])
                                if str(row["file_index"]).isdigit() else sys.maxsize))

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def show(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.2f}%"

    print("\nBOOM colour-filter efficiency audit")
    print("=" * 38)
    print(f"Selected file-index interval     : {start_index}..{end_index} (inclusive)")
    print(f"Expected simulated events        : {expected}")
    print(f"Input files available in folder  : {n_available_files}")
    print(f"Input light-curve files found    : {n_files}")
    print(f"Evaluable light curves           : {n_evaluable}")
    print(f"Without valid photometry         : {n_no_points}")
    print(f"Missing input files              : {n_missing}")
    print(f"Passed the colour filter         : {n_passed}")
    print(f"Rejected among evaluable curves  : {n_rejected}")
    print(f"Conditional filter efficiency    : {show(conditional_efficiency)}")
    print(f"Global efficiency                : {show(global_efficiency)}")
    print(f"\nPer-event audit: {args.audit_output}")
    print(f"Summary:         {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
