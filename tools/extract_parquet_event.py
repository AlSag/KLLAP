#!/usr/bin/env python3
"""Extract one KNe event from a sharded generator Parquet dataset.

Examples
--------
Synthetic event 165, written with the historical per-event CSV convention::

    python extract_parquet_event.py RUN_DIR 165

LSST-like event 165 with a chosen destination::

    python extract_parquet_event.py RUN_DIR 165 \
        --product lsstlike --output lightcurve_LSSTlike_0165.csv

The source dataset is never loaded in full. ``pyarrow.dataset`` applies an
``event_id`` predicate and uses Parquet statistics to skip unrelated shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one event from synthetic/LSST-like Parquet shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path, help="Generator run directory")
    parser.add_argument("event_id", type=int, help="Integer injected-event ID")
    parser.add_argument(
        "--product",
        choices=["synthetic", "lsstlike"],
        default="synthetic",
        help="Parquet population product to query",
    )
    parser.add_argument("--output", type=Path, help="Destination CSV path")
    parser.add_argument(
        "--keep-event-id",
        action="store_true",
        help="Keep the Parquet-only repeated event_id column in the CSV",
    )
    parser.add_argument(
        "--repeat-objectid",
        action="store_true",
        help=(
            "Keep objectid on every output row. By default only the first row "
            "contains objectid, matching the historical VM CSV convention."
        ),
    )
    return parser


def extract_event(run_dir: Path, product: str, event_id: int) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError(
            "This utility requires pyarrow: `python -m pip install pyarrow`."
        ) from exc

    parquet_dir = run_dir.expanduser().resolve() / "parquet"
    paths = sorted(parquet_dir.glob(f"{product}_part_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No {product}_part_*.parquet files found in {parquet_dir}"
        )
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    table = dataset.to_table(filter=ds.field("event_id") == int(event_id))
    if table.num_rows == 0:
        raise KeyError(f"Event {event_id} is absent from the {product} dataset")
    return table.to_pandas()


def main() -> None:
    args = build_parser().parse_args()
    frame = extract_event(args.run_dir, args.product, args.event_id)
    if not args.keep_event_id:
        frame = frame.drop(columns=["event_id"], errors="ignore")
    if "objectid" in frame and not args.repeat_objectid:
        frame["objectid"] = ""
        frame.at[frame.index[0], "objectid"] = f"{args.event_id:04d}"
    prefix = "lightcurve" if args.product == "synthetic" else "lightcurve_LSSTlike"
    destination = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path.cwd() / f"{prefix}_{args.event_id:04d}.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame):,} rows to {destination}")


if __name__ == "__main__":
    main()
