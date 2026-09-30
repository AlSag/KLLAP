from rubin_sim import data
from rubin_sim.maf.maps import DustMap
from rubin_sim.phot_utils import DustValues
import sqlite3
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import astropy.units as u
import secrets
import re
from pathlib import Path
from datetime import datetime
import argparse
import sys

from kne_pipeline_common import (
    prompt_choice,
    prompt_text,
    slugify,
    write_m5_depth_quantiles,
)

try:
    import healpy as hp
    HAS_HEALPY = True
except Exception:
    HAS_HEALPY = False


# ============================================================
# CONFIG / CLI
# ============================================================

def runtime_configuration() -> argparse.Namespace:
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
    args = parser.parse_args()
    if args.interactive or len(sys.argv) == 1:
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


RUNTIME = runtime_configuration()
SELECTION_MODE = RUNTIME.selection_mode
DDF_NAME = RUNTIME.ddf_name
TARGET_NAME = RUNTIME.target_name
N_FIELDS = int(RUNTIME.n_fields)
FOV_RADIUS_DEG = float(RUNTIME.fov_radius_deg)
SEED = int(RUNTIME.seed)
DUST_MAP_NSIDE = int(RUNTIME.dust_map_nside)
DUST_MAP_PATH = RUNTIME.dust_map_path.expanduser().resolve() if RUNTIME.dust_map_path else None
DUST_INTERP = bool(RUNTIME.dust_interp)
M5_LATITUDE_THRESHOLD_DEG = float(RUNTIME.m5_latitude_threshold_deg)

# Calcul d'aire HEALPix
AREA_NSIDE = 1024

# Mode d'échantillonnage des centres
SAMPLING_MODE = RUNTIME.sampling_mode      # "UNIFORM" | "STRATIFIED"
STRATIFY_WEIGHT = "n_centers"     # "n_centers" | "n_obs"
MIN_FIELDS_PER_TARGET = 1

# Plots revisit cadence
PLOT_LIN = True
PLOT_LOG = False
BINS = 60


# ============================================================
# HELPERS
# ============================================================

def slugify(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def save_m5_violin_plot(m5_values_by_band, outfile):
    """Save the pooled per-visit m5 distributions for the exported fields."""
    band_order = list("ugrizy")
    band_colors = {
        "u": "#ff7f0e",
        "g": "#2ca02c",
        "r": "#d62728",
        "i": "#9467bd",
        "z": "#8c564b",
        "y": "#e377c2",
    }

    bands = []
    distributions = []
    for band in band_order:
        chunks = m5_values_by_band.get(band, [])
        values = np.concatenate(chunks) if chunks else np.array([], dtype=float)
        values = values[np.isfinite(values)]
        if values.size >= 2:
            bands.append(band)
            distributions.append(values)

    if not distributions:
        print("No finite m5 distributions available: violin plot not created.")
        return

    positions = np.arange(1, len(bands) + 1)
    quantiles = np.asarray(
        [np.percentile(values, [16, 25, 50, 75, 84]) for values in distributions]
    )

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    violins = ax.violinplot(
        distributions,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=200,
        bw_method="scott",
    )
    for body in violins["bodies"]:
        body.set_facecolor("#7fb3d5")
        body.set_edgecolor("#7fb3d5")
        body.set_alpha(0.62)

    # Nested percentile intervals, followed by the p50 depth-cut marker.
    ax.vlines(
        positions,
        quantiles[:, 0],
        quantiles[:, 4],
        color="#1f77b4",
        linewidth=3,
        alpha=0.78,
        zorder=3,
    )
    ax.vlines(
        positions,
        quantiles[:, 1],
        quantiles[:, 3],
        color="#1f77b4",
        linewidth=9,
        alpha=0.52,
        zorder=4,
    )
    for position, band, median in zip(positions, bands, quantiles[:, 2]):
        ax.scatter(
            position,
            median,
            marker="D",
            s=72,
            color=band_colors[band],
            edgecolor="none",
            zorder=5,
        )

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=3, label="p16-p84"),
        Line2D([0], [0], color="#1f77b4", linewidth=9, alpha=0.52, label="p25-p75"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#ff7f0e",
            markeredgecolor="none",
            markersize=8,
            label="median (p50 threshold)",
        ),
    ]

    ax.set_xticks(positions, bands)
    ax.set_xlabel("LSST band")
    ax.set_ylabel(r"$m_5$ [mag]")
    ax.set_title(r"Per-visit LSST-like $m_5$ distributions; threshold = p50")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)

    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved m5 violin plot:", outfile)


def get_rubin_dust_coefficients() -> dict[str, float]:
    """Return Rubin coefficients R_band for A_band = R_band * E(B-V)."""
    raw = getattr(DustValues(), "r_x", None)
    if raw is None:
        raise RuntimeError("DustValues() has no attribute r_x")

    coefficients = {}
    for band in "ugrizy":
        if band in raw:
            coefficients[band] = float(raw[band])
        elif f"lsst{band}" in raw:
            coefficients[band] = float(raw[f"lsst{band}"])
        else:
            raise KeyError(
                f"Could not find Rubin dust coefficient for band {band!r} "
                f"in DustValues.r_x keys={list(raw)}"
            )
    return coefficients


def lookup_mw_ebv(dust_map, ra_deg, dec_deg) -> np.ndarray:
    """Evaluate the Rubin Milky-Way E(B-V) map at ICRS positions in degrees."""
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))

    if ra.shape != dec.shape:
        raise ValueError(f"RA and Dec must have the same shape, got {ra.shape} and {dec.shape}")
    if np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)):
        raise ValueError("Cannot query E(B-V) for non-finite RA/Dec values")
    if np.any((dec < -90.0) | (dec > 90.0)):
        raise ValueError("Declinations used for E(B-V) must be within [-90, 90] degrees")

    # DustMap expects ICRS longitude/latitude in radians for non-HEALPix points.
    slice_points = {
        "ra": np.deg2rad(np.mod(ra, 360.0)),
        "dec": np.deg2rad(dec),
    }
    try:
        result = dust_map.run(slice_points)
    except Exception as exc:
        location = str(DUST_MAP_PATH) if DUST_MAP_PATH is not None else "RUBIN_SIM_DATA_DIR"
        raise RuntimeError(
            "Unable to read the Rubin Milky-Way dust map. Install the Rubin data "
            f"files (for example with rs_download_data) or set --dust-map-path. "
            f"Current map location: {location}"
        ) from exc

    if "ebv" not in result:
        raise RuntimeError("DustMap did not return the expected 'ebv' values")

    ebv = np.asarray(result["ebv"], dtype=float).reshape(-1)
    if ebv.size != ra.size:
        raise RuntimeError(f"DustMap returned {ebv.size} values for {ra.size} positions")
    if np.any(~np.isfinite(ebv)):
        raise RuntimeError("DustMap returned non-finite E(B-V) values")
    if np.any(ebv < 0.0):
        raise RuntimeError("DustMap returned negative E(B-V) values")
    return ebv


def add_mw_extinction_to_centers(
    centers: pd.DataFrame,
    dust_map,
    dust_coefficients: dict[str, float],
) -> pd.DataFrame:
    """Add one E(B-V) and the six Rubin A_band values to each event position."""
    out = centers.copy()
    out["ebv_mw"] = lookup_mw_ebv(
        dust_map,
        out["fieldRA"].to_numpy(dtype=float),
        out["fieldDec"].to_numpy(dtype=float),
    )
    for band in "ugrizy":
        out[f"R_{band}_mw"] = float(dust_coefficients[band])
        out[f"A_{band}_mw"] = out["ebv_mw"] * float(dust_coefficients[band])
    return out


def build_selection(mode, ddf_name=None, target_name=None):
    """
    Retourne un dict avec :
      - where_sql : morceau SQL à injecter dans WHERE
      - params    : paramètres SQL
      - label     : suffixe pour le dossier d'export
    """
    mode = mode.upper()

    if mode == "ALL_DDF":
        return {
            "where_sql": "instr(UPPER(target_name), 'DDF') > 0",
            "params": (),
            "label": "all_ddf",
        }

    if mode == "LOWDUST":
        return {
            "where_sql": "target_name = ?",
            "params": ("lowdust",),
            "label": "lowdust",
        }

    if mode == "ALL_NON_DDF":
        return {
            "where_sql": "instr(UPPER(target_name), 'DDF') = 0",
            "params": (),
            "label": "all_non_ddf",
        }

    if mode == "ALL_SURVEY":
        return {
            "where_sql": "1 = 1",
            "params": (),
            "label": "all_survey",
        }

    if mode == "ONE_DDF":
        if ddf_name is None:
            raise ValueError("DDF_NAME must be set when SELECTION_MODE='ONE_DDF'")
        return {
            "where_sql": """
                instr(UPPER(target_name), 'DDF') > 0
                AND instr(UPPER(target_name), UPPER(?)) > 0
            """,
            "params": (ddf_name,),
            "label": slugify(ddf_name),
        }

    if mode == "TARGET":
        if target_name is None:
            raise ValueError("TARGET_NAME must be set when SELECTION_MODE='TARGET'")
        return {
            "where_sql": "target_name = ?",
            "params": (target_name,),
            "label": slugify(target_name),
        }

    raise ValueError(
        "SELECTION_MODE must be 'ALL_DDF', 'LOWDUST', 'ALL_NON_DDF', "
        "'ALL_SURVEY', 'ONE_DDF', or 'TARGET'."
    )


def list_matching_targets(con, selection):
    query = f"""
        SELECT target_name, COUNT(*) AS n
        FROM observations
        WHERE {selection["where_sql"]}
        GROUP BY target_name
        ORDER BY n DESC
    """
    return pd.read_sql_query(query, con, params=selection["params"])


def get_unique_centers_from_selection(con, selection):
    """
    Retourne tous les centres uniques (fieldRA, fieldDec) de la sélection,
    avec l'agrégation des target_name associés.
    """
    query = f"""
        SELECT
            fieldRA,
            fieldDec,
            GROUP_CONCAT(DISTINCT target_name) AS target_name,
            COUNT(DISTINCT target_name) AS n_target_names
        FROM observations
        WHERE {selection["where_sql"]}
        GROUP BY fieldRA, fieldDec
    """
    return pd.read_sql_query(query, con, params=selection["params"])


def get_centers_by_target_from_selection(con, selection):
    """
    Retourne les centres uniques par target_name.
    Une même position peut apparaître dans plusieurs target_name.
    """
    query = f"""
        SELECT
            target_name,
            fieldRA,
            fieldDec,
            COUNT(*) AS n_obs_center
        FROM observations
        WHERE {selection["where_sql"]}
        GROUP BY target_name, fieldRA, fieldDec
    """
    return pd.read_sql_query(query, con, params=selection["params"])


def allocate_n_per_stratum(strata_df, n_total, weight_col="n_centers", min_per_target=1):
    """
    Allocation entière par strate avec méthode du plus grand reste.
    On clippe ensuite à la capacité disponible ; le complément sera rajouté globalement.
    """
    out = strata_df.copy().reset_index(drop=True)

    if len(out) == 0:
        out["n_alloc"] = []
        return out

    capacities = out["n_centers"].to_numpy(dtype=int)
    weights = pd.to_numeric(out[weight_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    n_strata = len(out)

    if n_total <= 0:
        out["n_alloc"] = 0
        return out

    if n_total < n_strata:
        base = np.zeros(n_strata, dtype=int)
        order = np.argsort(-weights)
        base[order[:n_total]] = 1
        alloc = base
    else:
        base = np.full(n_strata, int(min_per_target), dtype=int)
        remaining = n_total - int(base.sum())

        if remaining < 0:
            base[:] = 0
            order = np.argsort(-weights)
            base[order[:n_total]] = 1
            alloc = base
        else:
            if np.sum(weights) <= 0:
                frac = np.full(n_strata, 1.0 / n_strata, dtype=float)
            else:
                frac = weights / np.sum(weights)

            raw = remaining * frac
            add = np.floor(raw).astype(int)
            leftover = remaining - int(np.sum(add))

            remainders = raw - add
            order = np.argsort(-remainders)
            if leftover > 0:
                add[order[:leftover]] += 1

            alloc = base + add

    alloc = np.minimum(alloc, capacities)
    out["n_alloc"] = alloc
    return out


def sample_positions_uniform_from_selection(con, selection, n_fields, rng):
    parent_centers = get_unique_centers_from_selection(con, selection)

    idx = rng.choice(
        parent_centers.index.to_numpy(),
        size=n_fields,
        replace=False if len(parent_centers) >= n_fields else True
    )
    sampled = parent_centers.loc[idx].reset_index(drop=True)

    print("all unique centers in parent selection:", len(parent_centers))
    print("sampled/exported centers:", len(sampled))

    return sampled, None, parent_centers


def sample_positions_stratified_from_selection(
    con,
    selection,
    n_fields,
    rng,
    weight_by="n_centers",
    min_fields_per_target=1,
):
    parent_centers = get_unique_centers_from_selection(con, selection)
    by_target = get_centers_by_target_from_selection(con, selection)

    strata = (
        by_target.groupby("target_name", as_index=False)
        .agg(
            n_centers=("fieldRA", "size"),
            n_obs=("n_obs_center", "sum"),
        )
        .sort_values(["n_centers", "n_obs"], ascending=False)
        .reset_index(drop=True)
    )

    if weight_by not in {"n_centers", "n_obs"}:
        raise ValueError("weight_by must be 'n_centers' or 'n_obs'")

    alloc = allocate_n_per_stratum(
        strata_df=strata,
        n_total=n_fields,
        weight_col=weight_by,
        min_per_target=min_fields_per_target,
    )

    sampled_parts = []

    for _, row in alloc.iterrows():
        target = row["target_name"]
        n_alloc = int(row["n_alloc"])

        if n_alloc <= 0:
            continue

        sub = by_target[by_target["target_name"] == target].copy().reset_index(drop=True)

        if len(sub) == 0:
            continue

        idx = rng.choice(
            sub.index.to_numpy(),
            size=n_alloc,
            replace=False if len(sub) >= n_alloc else True
        )

        tmp = sub.loc[idx, ["fieldRA", "fieldDec"]].copy()
        tmp["sampled_target_name"] = target
        sampled_parts.append(tmp)

    if len(sampled_parts) > 0:
        sampled_raw = pd.concat(sampled_parts, ignore_index=True)
    else:
        sampled_raw = pd.DataFrame(columns=["fieldRA", "fieldDec", "sampled_target_name"])

    # Déduplication des centres identiques tirés dans plusieurs strates
    sampled_unique = (
        sampled_raw.groupby(["fieldRA", "fieldDec"], as_index=False)
        .agg(
            sampled_target_name=("sampled_target_name", lambda s: ",".join(sorted(set(s)))),
            sampled_n_target_names=("sampled_target_name", lambda s: len(set(s))),
        )
    )

    # Rattacher l'agrégat parent complet
    sampled_unique = sampled_unique.merge(
        parent_centers,
        on=["fieldRA", "fieldDec"],
        how="left",
        suffixes=("_sampled", ""),
    )

    if "target_name" not in sampled_unique.columns:
        sampled_unique["target_name"] = sampled_unique["sampled_target_name"]
    else:
        sampled_unique["target_name"] = sampled_unique["target_name"].fillna(sampled_unique["sampled_target_name"])

    if "n_target_names" not in sampled_unique.columns:
        sampled_unique["n_target_names"] = sampled_unique["sampled_n_target_names"]
    else:
        sampled_unique["n_target_names"] = sampled_unique["n_target_names"].fillna(sampled_unique["sampled_n_target_names"])

    for col in ["sampled_target_name", "sampled_n_target_names"]:
        if col in sampled_unique.columns:
            sampled_unique = sampled_unique.drop(columns=col)

    # Top-up si la déduplication a réduit le nombre total
    if len(sampled_unique) < n_fields:
        already = sampled_unique[["fieldRA", "fieldDec"]].copy()
        remaining = parent_centers.merge(
            already,
            on=["fieldRA", "fieldDec"],
            how="left",
            indicator=True,
        )
        remaining = remaining[remaining["_merge"] == "left_only"].drop(columns=["_merge"])

        n_missing = min(n_fields - len(sampled_unique), len(remaining))
        if n_missing > 0:
            idx_add = rng.choice(
                remaining.index.to_numpy(),
                size=n_missing,
                replace=False if len(remaining) >= n_missing else True
            )
            add = remaining.loc[idx_add].copy()
            sampled_unique = pd.concat([sampled_unique, add], ignore_index=True)

    sampled_unique = sampled_unique.reset_index(drop=True)

    # Rapport par strate
    sampled_raw_counts = (
        sampled_raw.groupby("sampled_target_name")
        .size()
        .rename("n_sampled_raw")
        .reset_index()
        .rename(columns={"sampled_target_name": "target_name"})
    )

    sampled_unique_counts = (
        sampled_raw.drop_duplicates(subset=["fieldRA", "fieldDec", "sampled_target_name"])
        .groupby("sampled_target_name")
        .size()
        .rename("n_sampled_after_within_target_dedup")
        .reset_index()
        .rename(columns={"sampled_target_name": "target_name"})
    )

    strata_report = alloc.merge(sampled_raw_counts, on="target_name", how="left")
    strata_report = strata_report.merge(sampled_unique_counts, on="target_name", how="left")
    strata_report["n_sampled_raw"] = strata_report["n_sampled_raw"].fillna(0).astype(int)
    strata_report["n_sampled_after_within_target_dedup"] = (
        strata_report["n_sampled_after_within_target_dedup"].fillna(0).astype(int)
    )

    print("all unique centers in parent selection:", len(parent_centers))
    print("sampled/exported centers after stratification:", len(sampled_unique))

    return sampled_unique, strata_report, parent_centers


def sample_positions_from_selection(
    con,
    selection,
    n_fields,
    rng,
    sampling_mode="UNIFORM",
    stratify_weight="n_centers",
    min_fields_per_target=1,
):
    sampling_mode = sampling_mode.upper()

    if sampling_mode == "UNIFORM":
        return sample_positions_uniform_from_selection(con, selection, n_fields, rng)

    if sampling_mode == "STRATIFIED":
        return sample_positions_stratified_from_selection(
            con=con,
            selection=selection,
            n_fields=n_fields,
            rng=rng,
            weight_by=stratify_weight,
            min_fields_per_target=min_fields_per_target,
        )

    raise ValueError("SAMPLING_MODE must be 'UNIFORM' or 'STRATIFIED'")


def visits_covering_position(con, selection, ra0_deg, dec0_deg, fov_radius_deg):
    """
    Extrait les visites qui couvrent une position (ra0,dec0)
    dans la sélection demandée.
    """
    dec_min = dec0_deg - fov_radius_deg
    dec_max = dec0_deg + fov_radius_deg

    query = f"""
        SELECT observationStartMJD, band, fiveSigmaDepth, fieldRA, fieldDec, target_name
        FROM observations
        WHERE {selection["where_sql"]}
          AND fieldDec BETWEEN ? AND ?
    """

    params = tuple(selection["params"]) + (dec_min, dec_max)
    df = pd.read_sql_query(query, con, params=params)

    if len(df) == 0:
        return df[["observationStartMJD", "band", "fiveSigmaDepth"]].copy()

    centers = SkyCoord(
        ra=df["fieldRA"].to_numpy() * u.deg,
        dec=df["fieldDec"].to_numpy() * u.deg,
        frame="icrs",
    )
    pos0 = SkyCoord(ra=ra0_deg * u.deg, dec=dec0_deg * u.deg, frame="icrs")

    sep_deg = centers.separation(pos0).deg
    keep = sep_deg <= fov_radius_deg

    visits = df.loc[keep, ["observationStartMJD", "band", "fiveSigmaDepth"]].copy()
    visits = visits.sort_values("observationStartMJD").reset_index(drop=True)

    return visits


def selection_time_span(con, selection):
    """
    Durée globale de la sélection parent dans la base OpSim.
    """
    query = f"""
        SELECT
            MIN(observationStartMJD) AS mjd_min,
            MAX(observationStartMJD) AS mjd_max,
            COUNT(*) AS n_visits
        FROM observations
        WHERE {selection["where_sql"]}
    """
    row = pd.read_sql_query(query, con, params=selection["params"]).iloc[0]

    mjd_min = float(row["mjd_min"]) if pd.notnull(row["mjd_min"]) else np.nan
    mjd_max = float(row["mjd_max"]) if pd.notnull(row["mjd_max"]) else np.nan
    n_visits = int(row["n_visits"]) if pd.notnull(row["n_visits"]) else 0

    span_days = mjd_max - mjd_min if np.isfinite(mjd_min) and np.isfinite(mjd_max) else np.nan
    span_years = span_days / 365.25 if np.isfinite(span_days) else np.nan

    return {
        "parent_mjd_min": mjd_min,
        "parent_mjd_max": mjd_max,
        "parent_span_days": span_days,
        "parent_span_years": span_years,
        "parent_n_visits": n_visits,
    }


def union_area_of_fields_healpy(ra_deg, dec_deg, radius_deg, nside=1024):
    """
    Aire de l'union des disques centrés sur (ra,dec) de rayon radius_deg.
    Retourne l'aire en deg^2.
    """
    if not HAS_HEALPY:
        return np.nan, 0

    covered = set()
    radius_rad = np.deg2rad(radius_deg)

    for ra, dec in zip(ra_deg, dec_deg):
        if not np.isfinite(ra) or not np.isfinite(dec):
            continue

        theta = np.deg2rad(90.0 - dec)
        phi = np.deg2rad(ra % 360.0)
        vec = hp.ang2vec(theta, phi)

        pix = hp.query_disc(nside, vec, radius_rad, inclusive=True)
        covered.update(pix.tolist())

    area_deg2 = len(covered) * hp.nside2pixarea(nside, degrees=True)
    return float(area_deg2), int(len(covered))


# ============================================================
# EXPLORATION RAPIDE DE LA DATABASE
# ============================================================

opsim_db = RUNTIME.opsim_db.expanduser().resolve() if RUNTIME.opsim_db else data.get_baseline()
if not Path(opsim_db).exists():
    raise FileNotFoundError(f"OpSim database not found: {opsim_db}")
print("Baseline opsim sqlite:", opsim_db)

con = sqlite3.connect(opsim_db)

top = pd.read_sql_query(
    """
    SELECT target_name, COUNT(*) AS n
    FROM observations
    GROUP BY target_name
    ORDER BY n DESC
    LIMIT 30;
    """,
    con
)
print("Top targets:\n", top)

target = top["target_name"].iloc[0]
visits = pd.read_sql_query(
    """
    SELECT observationStartMJD, band, fiveSigmaDepth, target_name
    FROM observations
    WHERE target_name = ?
    ORDER BY observationStartMJD;
    """,
    con,
    params=(target,)
)

print("Chosen target_name:", target, "Nvisits:", len(visits))
print(visits.head())


# ============================================================
# MAIN
# ============================================================

rng = np.random.default_rng(SEED)

selection = build_selection(
    SELECTION_MODE,
    ddf_name=DDF_NAME,
    target_name=TARGET_NAME,
)

matched = list_matching_targets(con, selection)
print("Matching targets:")
print(matched.head(20))

print("Selection mode:", SELECTION_MODE)
print("Export label:", selection["label"])
print("Sampling mode:", SAMPLING_MODE)
if SAMPLING_MODE.upper() == "STRATIFIED":
    print("Stratify weight:", STRATIFY_WEIGHT)
    print("Min fields per target:", MIN_FIELDS_PER_TARGET)

sampled_centers, strata_report, parent_centers = sample_positions_from_selection(
    con=con,
    selection=selection,
    n_fields=N_FIELDS,
    rng=rng,
    sampling_mode=SAMPLING_MODE,
    stratify_weight=STRATIFY_WEIGHT,
    min_fields_per_target=MIN_FIELDS_PER_TARGET,
)

# One dust-map lookup per sampled source position. The same foreground value is
# then attached to all OpSim visits covering that position.
DUST_R_X = get_rubin_dust_coefficients()
mw_dust_map = DustMap(
    interp=DUST_INTERP,
    nside=DUST_MAP_NSIDE,
    map_path=str(DUST_MAP_PATH) if DUST_MAP_PATH is not None else None,
)
sampled_centers = add_mw_extinction_to_centers(
    sampled_centers,
    mw_dust_map,
    DUST_R_X,
)

OUTDIR = RUNTIME.outdir.expanduser().resolve() if RUNTIME.outdir else Path(f"opsim_exports_{selection['label']}_{slugify(SAMPLING_MODE)}_seed_{SEED}")
OUTDIR.mkdir(parents=True, exist_ok=True)

run_txt = OUTDIR / "run.txt"
content = (
    f"SELECTION_MODE: {SELECTION_MODE}\n"
    f"OPSIM_DB: {opsim_db}\n"
    f"DDF_NAME: {DDF_NAME}\n"
    f"TARGET_NAME: {TARGET_NAME}\n"
    f"EXPORT_LABEL: {selection['label']}\n"
    f"WHERE_SQL: {selection['where_sql']}\n"
    f"PARAMS: {selection['params']}\n"
    f"SEED: {SEED}\n"
    f"N_FIELDS: {N_FIELDS}\n"
    f"FOV_RADIUS_DEG: {FOV_RADIUS_DEG}\n"
    f"AREA_NSIDE: {AREA_NSIDE}\n"
    f"SAMPLING_MODE: {SAMPLING_MODE}\n"
    f"STRATIFY_WEIGHT: {STRATIFY_WEIGHT}\n"
    f"MIN_FIELDS_PER_TARGET: {MIN_FIELDS_PER_TARGET}\n"
    f"DUST_MAP: rubin_sim.maf.maps.DustMap\n"
    f"DUST_MAP_NSIDE: {DUST_MAP_NSIDE}\n"
    f"DUST_MAP_INTERP: {DUST_INTERP}\n"
    f"DUST_MAP_PATH: {DUST_MAP_PATH if DUST_MAP_PATH is not None else 'RUBIN_SIM_DATA_DIR'}\n"
    f"DUST_RELATION: A_band = R_band * E(B-V)\n"
    f"DUST_R_X: {DUST_R_X}\n"
    f"VISIT_FILE_COLUMNS: MJD,band,m5\n"
    f"FIELD_METADATA_FILE: opsim_fields_index.csv\n"
    f"TIMESTAMP: {datetime.now().isoformat(timespec='seconds')}\n"
)
run_txt.write_text(content)
print("Saved:", run_txt)

if strata_report is not None:
    strata_report.to_csv(OUTDIR / "strata_sampling_summary.csv", index=False)
    print("Saved:", OUTDIR / "strata_sampling_summary.csv")
    print(strata_report.head(20))

fields_index = []
all_dt = []
counts = []
m5_values_by_band = {band: [] for band in "ugrizy"}

global_mjd_min_exported = np.inf
global_mjd_max_exported = -np.inf

for j, row in sampled_centers.iterrows():
    ra0 = float(row["fieldRA"])
    dec0 = float(row["fieldDec"])
    target_name = row["target_name"]
    n_target_names = int(row["n_target_names"])
    ebv_mw = float(row["ebv_mw"])

    v = visits_covering_position(con, selection, ra0, dec0, FOV_RADIUS_DEG)
    counts.append(len(v))

    # Keep visit files deliberately compact.  Field identity, sky position and
    # Milky-Way extinction are stored once per field in opsim_fields_index.csv.
    # The shared pipeline reader accepts this schema and the older verbose one.
    visits_csv = (
        v[["observationStartMJD", "band", "fiveSigmaDepth"]]
        .rename(columns={"observationStartMJD": "MJD", "fiveSigmaDepth": "m5"})
        .sort_values("MJD")
        .reset_index(drop=True)
    )

    # Keep the exact finite m5 values from the rows exported for each sampled
    # field. These pooled, per-visit values define the synthetic depth cut.
    for band in m5_values_by_band:
        band_values = pd.to_numeric(
            visits_csv.loc[visits_csv["band"].astype(str) == band, "m5"],
            errors="coerce",
        ).to_numpy(float)
        band_values = band_values[np.isfinite(band_values)]
        if band_values.size:
            m5_values_by_band[band].append(band_values)

    outname = OUTDIR / f"opsim_visits_field{j:03d}.csv"
    visits_csv.to_csv(outname, index=False)

    if len(visits_csv) > 0:
        field_mjd_min = float(visits_csv["MJD"].min())
        field_mjd_max = float(visits_csv["MJD"].max())
        field_span_days = field_mjd_max - field_mjd_min
        field_span_years = field_span_days / 365.25

        global_mjd_min_exported = min(global_mjd_min_exported, field_mjd_min)
        global_mjd_max_exported = max(global_mjd_max_exported, field_mjd_max)
    else:
        field_mjd_min = np.nan
        field_mjd_max = np.nan
        field_span_days = np.nan
        field_span_years = np.nan

    field_metadata = {
        "field_index": j,
        "ra0_deg": ra0,
        "dec0_deg": dec0,
        "target_name": target_name,
        "n_target_names": n_target_names,
        "Nvisits": len(visits_csv),
        "field_mjd_min": field_mjd_min,
        "field_mjd_max": field_mjd_max,
        "field_span_days": field_span_days,
        "field_span_years": field_span_years,
        "ebv_mw": ebv_mw,
    }
    for band in "ugrizy":
        field_metadata[f"R_{band}_mw"] = float(DUST_R_X[band])
        field_metadata[f"A_{band}_mw"] = ebv_mw * float(DUST_R_X[band])
    fields_index.append(field_metadata)

    if j < 3:
        print("Saved:", outname, "Nvisits:", len(visits_csv), "target_name:", target_name)

    if len(v) < 2:
        continue

    v["delta_t_days"] = v.groupby("band")["observationStartMJD"].diff()
    dt = v[["observationStartMJD", "band", "delta_t_days", "fiveSigmaDepth"]].dropna()
    dt = dt.rename(columns={"fiveSigmaDepth": "m5"})
    dt["field_index"] = j
    all_dt.append(dt)

fields_index_df = pd.DataFrame(fields_index)
fields_index_df.to_csv(OUTDIR / "opsim_fields_index.csv", index=False)
print("Saved fields index:", OUTDIR / "opsim_fields_index.csv")

# This is an OpSim field-sample property, so calculate and store it once here.
# generate_kne_lightcurves.py only reads this summary and never recomputes it.
m5_summary_rows = []
for band in "ugrizy":
    chunks = m5_values_by_band[band]
    values = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    m5_summary_rows.append(
        {
            "band": band,
            "m5_median": float(np.median(values)) if values.size else np.nan,
            "N_exported_visit_rows": int(values.size),
            "weighting": "one_equal_weight_per_exported_field_visit_row",
        }
    )

m5_summary = pd.DataFrame(m5_summary_rows)
m5_summary_path = OUTDIR / "opsim_m5_medians_by_band.csv"
m5_summary.to_csv(m5_summary_path, index=False)
print("Saved per-band m5 medians:", m5_summary_path)
print(m5_summary)

# This canonical table is consumed directly from the OpSim export directory by
# the envelope and latitude-loss scripts. It therefore needs to be generated
# only once and never copied into individual simulation runs.
m5_quantile_path = write_m5_depth_quantiles(
    OUTDIR,
    latitude_threshold_deg=M5_LATITUDE_THRESHOLD_DEG,
)
print("Saved canonical m5 depth quantiles:", m5_quantile_path)

save_m5_violin_plot(
    m5_values_by_band,
    OUTDIR / "m5_distribution_by_band_violin.png",
)

parent_time_stats = selection_time_span(con, selection)

parent_area_deg2, parent_n_healpix = union_area_of_fields_healpy(
    ra_deg=parent_centers["fieldRA"].to_numpy(dtype=float),
    dec_deg=parent_centers["fieldDec"].to_numpy(dtype=float),
    radius_deg=FOV_RADIUS_DEG,
    nside=AREA_NSIDE,
)
parent_f_sky = parent_area_deg2 / 41253.0 if np.isfinite(parent_area_deg2) else np.nan

exported_area_deg2, exported_n_healpix = union_area_of_fields_healpy(
    ra_deg=fields_index_df["ra0_deg"].to_numpy(dtype=float),
    dec_deg=fields_index_df["dec0_deg"].to_numpy(dtype=float),
    radius_deg=FOV_RADIUS_DEG,
    nside=AREA_NSIDE,
)
exported_f_sky = exported_area_deg2 / 41253.0 if np.isfinite(exported_area_deg2) else np.nan

if np.isfinite(global_mjd_min_exported) and np.isfinite(global_mjd_max_exported):
    exported_span_days = global_mjd_max_exported - global_mjd_min_exported
    exported_span_years = exported_span_days / 365.25
else:
    exported_span_days = np.nan
    exported_span_years = np.nan

mean_field_span_years = float(np.nanmean(fields_index_df["field_span_years"])) if len(fields_index_df) else np.nan
median_field_span_years = float(np.nanmedian(fields_index_df["field_span_years"])) if len(fields_index_df) else np.nan
m5_median_by_band = m5_summary.set_index("band")["m5_median"].to_dict()

survey_summary = pd.DataFrame([{
    "selection_mode": SELECTION_MODE,
    "selection_label": selection["label"],
    "sampling_mode": SAMPLING_MODE,
    "stratify_weight": STRATIFY_WEIGHT if SAMPLING_MODE.upper() == "STRATIFIED" else "",
    "N_FIELDS_requested": N_FIELDS,
    "N_FIELDS_exported": len(fields_index_df),
    "N_parent_unique_centers": len(parent_centers),
    "FOV_RADIUS_DEG": FOV_RADIUS_DEG,

    "parent_area_deg2_union_all_centers": parent_area_deg2,
    "parent_f_sky_union_all_centers": parent_f_sky,

    "exported_area_deg2_union_sampled_fields": exported_area_deg2,
    "exported_f_sky_union_sampled_fields": exported_f_sky,

    "exported_over_parent_area_ratio": (
        exported_area_deg2 / parent_area_deg2
        if np.isfinite(exported_area_deg2) and np.isfinite(parent_area_deg2) and parent_area_deg2 > 0
        else np.nan
    ),

    "exported_mjd_min": global_mjd_min_exported if np.isfinite(global_mjd_min_exported) else np.nan,
    "exported_mjd_max": global_mjd_max_exported if np.isfinite(global_mjd_max_exported) else np.nan,
    "exported_span_days": exported_span_days,
    "exported_span_years": exported_span_years,

    "mean_field_span_years": mean_field_span_years,
    "median_field_span_years": median_field_span_years,

    "parent_mjd_min": parent_time_stats["parent_mjd_min"],
    "parent_mjd_max": parent_time_stats["parent_mjd_max"],
    "parent_span_days": parent_time_stats["parent_span_days"],
    "parent_span_years": parent_time_stats["parent_span_years"],
    "parent_n_visits": parent_time_stats["parent_n_visits"],

    "parent_healpy_area_npix": parent_n_healpix,
    "exported_healpy_area_npix": exported_n_healpix,
    "dust_map_nside": DUST_MAP_NSIDE,
    "dust_map_interp": DUST_INTERP,
    "exported_ebv_mw_min": float(fields_index_df["ebv_mw"].min()) if len(fields_index_df) else np.nan,
    "exported_ebv_mw_median": float(fields_index_df["ebv_mw"].median()) if len(fields_index_df) else np.nan,
    "exported_ebv_mw_max": float(fields_index_df["ebv_mw"].max()) if len(fields_index_df) else np.nan,
    "m5_medians_file": m5_summary_path.name,
    "m5_quantiles_file": m5_quantile_path.name,
    "m5_latitude_threshold_deg": M5_LATITUDE_THRESHOLD_DEG,
    "visit_file_columns": "MJD,band,m5",
    "field_metadata_file": "opsim_fields_index.csv",
    **{f"m5_median_{band}": m5_median_by_band.get(band, np.nan) for band in "ugrizy"},
}])

survey_summary.to_csv(OUTDIR / "opsim_survey_summary.csv", index=False)
print("Saved survey summary:", OUTDIR / "opsim_survey_summary.csv")
print(survey_summary.T)

con.close()

print("N_FIELDS =", N_FIELDS)
print("Visits per field: min/median/max =", np.min(counts), np.median(counts), np.max(counts))

all_dt = (
    pd.concat(all_dt, ignore_index=True)
    if len(all_dt)
    else pd.DataFrame(columns=["band", "delta_t_days", "field_index"])
)
print(all_dt.head())


# ============================================================
# PLOTS Δt
# ============================================================

for b in sorted(all_dt["band"].unique()):
    x_all = all_dt.loc[all_dt["band"] == b, "delta_t_days"].to_numpy()

    if PLOT_LIN:
        x = x_all[np.isfinite(x_all)]
        if len(x) > 0:
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            ax.hist(x, bins=BINS, histtype="step", linewidth=2.0)
            ax.set_xlabel(r"$\Delta t$ (days)")
            ax.set_ylabel("Counts")
            ax.set_title(f"Revisit Δt (linear bins) in band {b} (N_FIELDS={N_FIELDS})")
            ax.grid(True, alpha=0.25)
            plt.show()

    if PLOT_LOG:
        x = x_all[np.isfinite(x_all) & (x_all > 0)]
        if len(x) > 0:
            bins_log = np.logspace(np.log10(x.min()), np.log10(x.max()), BINS)

            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            ax.hist(x, bins=bins_log, histtype="step", linewidth=2.0)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\Delta t$ (days)")
            ax.set_ylabel("Counts")
            ax.set_title(f"Revisit Δt (log bins + log x) in band {b} (N_FIELDS={N_FIELDS})")
            ax.grid(True, alpha=0.25)
            plt.show()
