#!/usr/bin/env python3
"""Terminal interface for the OpSim export command.

Only command-line parsing and interactive questions belong here. Database
selection, sky sampling, dust evaluation, and table production remain in
``get_opsim_baseline.py``.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

from kne_pipeline_common import prompt_choice, prompt_text


def runtime_configuration(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--selection-mode", choices=["ALL_DDF", "LOWDUST", "ALL_NON_DDF", "ALL_SURVEY", "ONE_DDF", "TARGET"], default="ALL_SURVEY")
    parser.add_argument("--opsim-db", type=Path, help="Explicit baseline SQLite database; default uses rubin_sim.data.get_baseline()")
    parser.add_argument("--ddf-name", default="COSMOS")
    parser.add_argument("--target-name", default="DD:COSMOS, lowdust, DDF COSMOS")
    parser.add_argument("--n-fields", type=int, default=100)
    parser.add_argument("--fov-radius-deg", type=float, default=1.75)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sampling-mode", choices=["UNIFORM", "STRATIFIED"], default="UNIFORM")
    parser.add_argument(
        "--dust-map-nside",
        type=int,
        default=128,
        help="HEALPix nside of the Rubin Milky-Way E(B-V) map",
    )
    parser.add_argument(
        "--dust-map-path",
        type=Path,
        help="Optional directory containing the Rubin dust-map files",
    )
    parser.add_argument(
        "--dust-interp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate the E(B-V) HEALPix map at each sampled sky position",
    )
    parser.add_argument(
        "--m5-latitude-threshold-deg",
        type=float,
        default=20.0,
        help="Boundary used for the high/low-latitude m5 summaries",
    )
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(argv)
    if args.interactive or (argv is None and len(sys.argv) == 1):
        args.selection_mode = prompt_choice(
            "OpSim survey selection",
            [("ALL_SURVEY", "All survey visits"), ("LOWDUST", "Low-dust WFD"), ("ALL_DDF", "All DDF"), ("ALL_NON_DDF", "All non-DDF"), ("ONE_DDF", "One DDF"), ("TARGET", "Exact target name")],
            args.selection_mode,
        )
        args.n_fields = int(prompt_text("Number of sampled fields", str(args.n_fields)))
        args.sampling_mode = prompt_choice(
            "Field-center sampling",
            [("UNIFORM", "Uniform among eligible centers"), ("STRATIFIED", "Stratified by target")],
            args.sampling_mode,
        )
        args.seed = int(prompt_text("Random seed", str(args.seed if args.seed is not None else secrets.randbits(32))))
        if args.selection_mode == "ONE_DDF":
            args.ddf_name = prompt_text("DDF name", args.ddf_name)
        if args.selection_mode == "TARGET":
            args.target_name = prompt_text("Exact target_name", args.target_name)
        args.m5_latitude_threshold_deg = float(
            prompt_text(
                "m5 high/low Galactic-latitude boundary [deg]",
                str(args.m5_latitude_threshold_deg),
            )
        )
    if args.seed is None:
        args.seed = secrets.randbits(32)
    if args.dust_map_nside <= 0:
        parser.error("--dust-map-nside must be strictly positive")
    if not 0 <= args.m5_latitude_threshold_deg <= 90:
        parser.error("--m5-latitude-threshold-deg must lie in [0, 90]")
    return args

# Calcul d'aire HEALPix
AREA_NSIDE = 1024

# Mode d'échantillonnage des centres
STRATIFY_WEIGHT = "n_centers"     # "n_centers" | "n_obs"
MIN_FIELDS_PER_TARGET = 1

# Plots revisit cadence
PLOT_LIN = True
PLOT_LOG = False
BINS = 60


# ============================================================
# HELPERS
# ============================================================
