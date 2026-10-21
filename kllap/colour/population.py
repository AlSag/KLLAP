"""Shared population loading and colour interpolation for colour analyses.

This module contains no command-line interface and creates no figures.  It
owns the scientific data preparation used by the iso-colour command: metadata
loading, physical-parameter transforms, saved-envelope lookup, and log-time
interpolation of two-band colours.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import kllap.envelope.core as envelope_core
from kllap.population_io import SyntheticPopulationSource


STAGES = envelope_core.STAGES
PARAMETER_LABELS = {
    "log10_mej_dyn": r"$\log_{10}(M_{\rm dyn}/M_\odot)$",
    "log10_mej_wind": r"$\log_{10}(M_{\rm wind}/M_\odot)$",
    "mej_dyn": r"$M_{\rm dyn}\ (M_\odot)$",
    "mej_wind": r"$M_{\rm wind}\ (M_\odot)$",
    "v_ej_dyn": r"$v_{\rm dyn}/c$",
    "v_ej_wind": r"$v_{\rm wind}/c$",
    "Ye_dyn": r"$Y_{e,\rm dyn}$",
    "Ye_wind": r"$Y_{e,\rm wind}$",
    "inclination_EM": r"$\iota\ ({\rm rad})$",
    "inclination_deg": r"$\iota\ ({\rm deg})$",
    "cos_inclination": r"$\cos\iota$",
    "phi": r"$\phi\ ({\rm deg})$",
    "luminosity_distance": r"$D_L\ ({\rm Mpc})$",
    "redshift": r"$z$",
    "ebv_mw": r"$E(B-V)_{\rm MW}$",
    "log10_mej": r"$\log_{10}(M_{\rm ej}/M_\odot)$",
    "log10_vej": r"$\log_{10}(v_{\rm ej}/c)$",
    "beta": r"$\beta$",
    "log10_kappa_r": r"$\log_{10}(\kappa_r/[{\rm cm^2\,g^{-1}}])$",
}


def ask(text: str, default="") -> str:
    """Small prompt helper retained for the interactive public command."""
    suffix = f" [{default}]" if default != "" else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or str(default)


def choose(title, choices, default):
    print("\n" + title)
    for index, (key, label) in enumerate(choices, 1):
        marker = " [défaut]" if key == default else ""
        print(f"  {index}. {label}{marker}")
    selected = int(ask("Choix", next(
        i for i, (key, _) in enumerate(choices, 1) if key == default
    ))) - 1
    if not 0 <= selected < len(choices):
        raise ValueError("Choix invalide")
    return choices[selected][0]


def parse_parameter_pairs(text: str) -> list[tuple[str, str]]:
    pairs = [tuple(item.strip().split(":")) for item in text.split(",")]
    if any(len(pair) != 2 or not all(pair) or pair[0] == pair[1] for pair in pairs):
        raise ValueError("Utiliser des plans x:y avec deux paramètres distincts")
    return pairs


def read_bounds(path, times, stage, percentile_pair=(5.0, 95.0), repetition=0):
    """Read containing temporal bins without filling or extrapolating them."""
    table = pd.read_csv(path)
    if "stage" in table:
        table = table.loc[table.stage.astype(str).eq(stage)].copy()
    if "repetition" in table:
        values = pd.to_numeric(table.repetition, errors="coerce")
        table = table.loc[values.eq(repetition)].copy()
    if "N_requested" in table and not table.empty:
        values = pd.to_numeric(table.N_requested, errors="coerce")
        table = table.loc[values.eq(values.max())].copy()
    lower = next((name for name in ("t_min_days", "t_low_days") if name in table), None)
    upper = next((name for name in ("t_max_days", "t_high_days") if name in table), None)
    if lower is None or upper is None:
        raise ValueError("Le CSV nécessite des bornes temporelles min/max")
    is_profile = {"percentile", "value", "valid"}.issubset(table.columns)
    if not is_profile and not {"color_min", "color_max"}.issubset(table.columns):
        raise ValueError("Le CSV nécessite color_min/color_max ou percentile/value/valid")
    if is_profile and "estimator_kind" in table:
        table = table.loc[table.estimator_kind.astype(str).eq("fixed")]
    end = pd.to_numeric(table[upper], errors="coerce").max()
    rows = []
    for time in times:
        lows = pd.to_numeric(table[lower], errors="coerce")
        highs = pd.to_numeric(table[upper], errors="coerce")
        mask = lows.le(time) & highs.gt(time)
        if np.isclose(time, end, rtol=0, atol=1e-10):
            mask = highs.eq(end) & lows.le(time)
        selected = table.loc[mask]
        row = {
            "time_days": float(time), "color_min": np.nan,
            "color_max": np.nan, "bounds_status": "missing_or_invalid",
            "bounds_source": str(path),
        }
        if is_profile:
            values = []
            for quantile in percentile_pair:
                numeric = pd.to_numeric(selected.percentile, errors="coerce")
                subset = selected.loc[np.isclose(numeric, quantile)]
                if len(subset) > 1:
                    raise ValueError(f"Bornes ambiguës pour t={time:g}, p{quantile:g}")
                if len(subset) == 1 and str(subset.valid.iloc[0]).lower() in ("true", "1"):
                    values.append(float(subset.value.iloc[0]))
            if len(values) == 2:
                row.update(color_min=values[0], color_max=values[1])
        elif len(selected) == 1:
            row.update(
                color_min=float(selected.color_min.iloc[0]),
                color_max=float(selected.color_max.iloc[0]),
            )
        elif len(selected) > 1:
            raise ValueError(f"Plusieurs bins couvrent t={time:g}; CSV ambigu")
        if (np.isfinite(row["color_min"]) and np.isfinite(row["color_max"])
                and row["color_min"] <= row["color_max"]):
            row["bounds_status"] = "valid"
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_values(metadata: pd.DataFrame, name: str) -> np.ndarray:
    derived = {
        "mej_dyn": ("log10_mej_dyn", lambda values: 10.0 ** values),
        "mej_wind": ("log10_mej_wind", lambda values: 10.0 ** values),
        "inclination_deg": ("inclination_EM", np.rad2deg),
        "cos_inclination": ("inclination_EM", np.cos),
    }
    if name in metadata:
        return pd.to_numeric(metadata[name], errors="coerce").to_numpy(float)
    if name in derived and derived[name][0] in metadata:
        column, transform = derived[name]
        values = pd.to_numeric(metadata[column], errors="coerce").to_numpy(float)
        return transform(values)
    raise ValueError(f"Paramètre '{name}' absent de summary.csv")


def axis_domain(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Aucune valeur de paramètre finie")
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        raise ValueError("Paramètre constant : pas de densité 2D identifiable")
    return low, high


def resolve_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.name in ("parquet", "csv") and (root.parent / "summary.csv").is_file():
        return root.parent
    return root


def load_metadata(root: Path, maximum: int, seed: int) -> pd.DataFrame:
    from kllap.colour.common import load_summary

    metadata = load_summary(root)
    if metadata.empty:
        raise ValueError(f"summary.csv absent ou vide dans {root}")
    if metadata.event_id.duplicated().any():
        raise ValueError("summary.csv contient des event_id dupliqués")
    if 0 < maximum < len(metadata):
        positions = np.sort(
            np.random.default_rng(seed).choice(len(metadata), maximum, replace=False)
        )
        metadata = metadata.iloc[positions].copy()
    return metadata.reset_index(drop=True)


def load_event_colours(args, root, metadata, bands, times, depths):
    """Interpolate saved magnitudes and return colour/support/depth masks."""
    from kllap.colour.common import canonical_event_id, interpolate_band

    source = SyntheticPopulationSource(root)
    if source.kind == "missing":
        raise FileNotFoundError("Aucune courbe synthétique Parquet ou CSV")
    ids = metadata.event_id.astype(str).tolist()
    position = {event_id: index for index, event_id in enumerate(ids)}
    colour = np.full((len(ids), len(times)), np.nan)
    supported = np.zeros(colour.shape, dtype=bool)
    depth_pass = np.zeros(colour.shape, dtype=bool)
    selection_system = (
        "lsst" if args.photometric_system == "ps1"
        and args.stage in envelope_core.M5_STAGES else args.photometric_system
    )
    systems = tuple(dict.fromkeys((args.photometric_system, selection_system)))
    read_ids = ids
    if source.kind == "legacy_csv_files":
        read_ids = [event_id for event_id in ids if event_id in source.csv_by_event]
    seen = set()
    print(f"Lecture unique de {len(ids):,} événements ({source.kind}); aucun split")
    for event_id, frame in source.iter_events(read_ids, bands, systems):
        event_id = canonical_event_id(event_id)
        if event_id in seen:
            raise ValueError(f"Événement {event_id} présent dans plusieurs shards")
        seen.add(event_id)
        index = position[event_id]
        row = metadata.iloc[index]
        evaluation_times = times
        if args.time_frame == "source":
            redshift = float(row.get("redshift", np.nan))
            if not np.isfinite(redshift) or redshift < 0:
                raise ValueError("Le temps source nécessite un redshift fini >= 0")
            evaluation_times = times * (1.0 + redshift)
        frame = frame.copy()
        frame["t_days"] = pd.to_numeric(frame.t_days, errors="coerce")
        frame["band"] = frame.band.astype(str).str.strip().str.lower()
        frame["magnitude"] = pd.to_numeric(frame.mag, errors="coerce")
        if "photometric_system" not in frame:
            frame["photometric_system"] = "lsst"
        frame["photometric_system"] = (
            frame.photometric_system.astype(str).str.strip().str.lower()
        )
        frame = frame.loc[
            np.isfinite(frame.t_days) & (frame.t_days > 0)
            & np.isfinite(frame.magnitude)
        ]
        target = frame.loc[frame.photometric_system.eq(args.photometric_system)]
        selection = frame.loc[frame.photometric_system.eq(selection_system)]
        magnitudes = [interpolate_band(target, band, evaluation_times) for band in bands]
        selection_magnitudes = [
            interpolate_band(selection, band, evaluation_times) for band in bands
        ]
        if args.stage not in envelope_core.MW_EXTINCTED_STAGES:
            magnitudes = [
                values - envelope_core.event_extinction_mag(
                    row, args.photometric_system, band
                )
                for values, band in zip(magnitudes, bands)
            ]
            if args.stage in envelope_core.M5_STAGES:
                selection_magnitudes = [
                    values - envelope_core.event_extinction_mag(
                        row, selection_system, band
                    )
                    for values, band in zip(selection_magnitudes, bands)
                ]
        finite = np.isfinite(magnitudes[0]) & np.isfinite(magnitudes[1])
        supported[index] = finite
        colour[index] = magnitudes[0] - magnitudes[1]
        passed = finite.copy()
        if args.stage in envelope_core.M5_STAGES:
            for values, band in zip(selection_magnitudes, bands):
                passed &= np.isfinite(values) & (values <= depths[band])
        depth_pass[index] = passed
        if len(seen) % max(1000, len(ids) // 10) == 0:
            print(f"  {len(seen):,}/{len(ids):,}")
    if not supported.any():
        raise ValueError(f"Aucune couleur {args.color_pair} disponible")
    return colour, supported, depth_pass
