#!/usr/bin/env python3
"""Pure table transformations shared by KNe population generation.

This module contains deterministic operations on already generated values. It
does not import FIESTA, JAX, Astropy, or Rubin, and importing it never starts a
simulation. Physical model evaluation and interactive configuration remain in
``generate_kne_lightcurves.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RUBIN_BANDS = tuple("ugrizy")


def ensure_mw_extinction_columns(
    visits: pd.DataFrame,
    dust_coefficients: dict[str, float],
) -> pd.DataFrame:
    """Return visits with E(B-V), per-band extinction, and row extinction."""
    v = visits.copy()
    if "ebv_mw" not in v.columns:
        for candidate in ("ebv", "E_BV", "EBV", "sfd_ebv"):
            if candidate in v.columns:
                v["ebv_mw"] = pd.to_numeric(v[candidate], errors="coerce")
                break
    if "ebv_mw" not in v.columns:
        v["ebv_mw"] = np.nan
    v["ebv_mw"] = pd.to_numeric(v["ebv_mw"], errors="coerce")

    coefficients = {
        band: float(dust_coefficients[band]) for band in RUBIN_BANDS
    }
    for band, coefficient in coefficients.items():
        if f"R_{band}_mw" not in v.columns:
            v[f"R_{band}_mw"] = coefficient
        if f"A_{band}_mw" not in v.columns:
            v[f"A_{band}_mw"] = v["ebv_mw"] * coefficient

    if "R_mw" not in v.columns:
        v["R_mw"] = v["band"].astype(str).map(coefficients)
    if "A_mw" not in v.columns:
        v["A_mw"] = v["band"].astype(str).map(coefficients) * v["ebv_mw"]
    return v


def get_event_mw_extinction_summary(
    visits: pd.DataFrame,
    dust_coefficients: dict[str, float],
) -> dict[str, object]:
    """Summarize E(B-V) and Rubin extinction for one selected field/event."""
    v = ensure_mw_extinction_columns(visits, dust_coefficients)
    ebv_values = pd.to_numeric(
        v.get("ebv_mw", pd.Series(np.nan, index=v.index)), errors="coerce"
    ).to_numpy(float)
    ebv = (
        float(np.nanmedian(ebv_values))
        if np.any(np.isfinite(ebv_values))
        else np.nan
    )
    summary: dict[str, object] = {"ebv_mw": ebv}
    for column in ("fieldRA", "fieldDec"):
        values = pd.to_numeric(
            v.get(column, pd.Series(np.nan, index=v.index)), errors="coerce"
        ).to_numpy(float)
        summary[column] = (
            float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan
        )
    target_values = v.get("target_name", pd.Series("", index=v.index))
    target_values = target_values.dropna().astype(str)
    summary["target_name"] = target_values.iloc[0] if len(target_values) else ""

    for band in RUBIN_BANDS:
        coefficient = float(dust_coefficients[band])
        summary[f"R_{band}_mw"] = coefficient
        column = f"A_{band}_mw"
        values = pd.to_numeric(
            v.get(column, pd.Series(np.nan, index=v.index)), errors="coerce"
        ).to_numpy(float)
        summary[column] = (
            float(np.nanmedian(values))
            if np.any(np.isfinite(values))
            else coefficient * ebv if np.isfinite(ebv) else np.nan
        )
    return summary


def trim_lsstlike_after_last_detection(
    observations: pd.DataFrame,
    days_after_last_detection: float = 10.0,
) -> tuple[pd.DataFrame, float, float, bool]:
    """Keep valid visits through a fixed time after the last detection."""
    if days_after_last_detection < 0:
        raise ValueError("days_after_last_detection must be non-negative")

    out = observations.loc[
        np.isfinite(pd.to_numeric(observations["mag_true"], errors="coerce"))
        & np.isfinite(pd.to_numeric(observations["t_days"], errors="coerce"))
    ].copy()
    out = out.sort_values("observationStartMJD").reset_index(drop=True)

    detected = out["detected"].astype(bool)
    if not detected.any():
        return out, np.nan, np.nan, False

    last_detection = float(out.loc[detected, "t_days"].iloc[-1])
    window_end = last_detection + float(days_after_last_detection)
    out = out.loc[out["t_days"] <= window_end].reset_index(drop=True)
    return out, last_detection, window_end, True


def apply_synthetic_m5_median_cut(
    measurements: pd.DataFrame,
    objectid: str,
    m5_medians: dict[str, float],
) -> pd.DataFrame:
    """Keep finite synthetic points at or brighter than their band's m5."""
    out = measurements.copy()
    magnitudes = pd.to_numeric(out["mag"], errors="coerce").to_numpy(float)
    thresholds = pd.to_numeric(
        out["band"].map(m5_medians), errors="coerce"
    ).to_numpy(float)
    keep = (
        np.isfinite(magnitudes)
        & np.isfinite(thresholds)
        & (magnitudes <= thresholds)
    )
    out = out.loc[keep].reset_index(drop=True)

    out["objectid"] = ""
    if len(out):
        out.at[0, "objectid"] = str(objectid)
    out["candidate_id"] = [
        f"{objectid}_{index}" for index in range(1, len(out) + 1)
    ]
    return out


def keep_only_mw_extincted_synthetic_magnitude(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Replace the saved magnitude by its MW-extincted value."""
    if "mag_mw_extincted" not in measurements.columns:
        raise ValueError(
            "MW-extincted synthetic output requires finite A_band values from "
            "the selected OpSim field"
        )
    out = measurements.copy()
    out["mag"] = pd.to_numeric(out["mag_mw_extincted"], errors="coerce")
    return out.drop(columns=["mag_mw_extincted"])


def derive_sky_seed(
    intrinsic_seed: int,
    latitude_mode: str,
    threshold_deg: float,
    latitude_max_deg: float | None = None,
) -> int:
    """Derive a reproducible sky seed for one latitude selection."""
    role_code = {"all": 0xA11, "low": 0x10A, "high": 0xA19, "bin": 0xB17}[
        latitude_mode
    ]
    threshold_code = int(round(float(threshold_deg) * 1000.0))
    seed_value = int(intrinsic_seed) ^ 0x534B59 ^ role_code ^ threshold_code
    if latitude_mode == "bin":
        if latitude_max_deg is None:
            raise ValueError("latitude_max_deg is required for latitude_mode='bin'")
        maximum_code = int(round(float(latitude_max_deg) * 1000.0))
        seed_value ^= (maximum_code * 0x9E3779B1) & 0xFFFFFFFF
    return int(seed_value % (2**32))
