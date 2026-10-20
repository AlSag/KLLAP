#!/usr/bin/env python3
"""Project a colour-selected KNe population into physical-parameter planes.

Run without arguments for interactive terminal prompts. Saved synthetic
magnitudes are read once, with the same log-time interpolation, MW convention
and global depth cuts as build_color_envelopes.py. No surrogate is evaluated.
There is no train/validation/test split. Contours are highest-density regions
of the selected population, NOT confidence regions or selection boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

import kne_envelope_core as envelope_core
from kne_m5 import M5_SCENARIOS


# Public compatibility alias; the canonical stage ordering lives in the
# numerical envelope module.
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

ENVELOPE_FAMILIES = {
    "p5-p95": (5.0, 95.0),
    "p16-p84": (16.0, 84.0),
    "p25-p75": (25.0, 75.0),
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--color-pair", default="g-r")
    parser.add_argument("--photometric-system", choices=("lsst", "ps1", "ztf"), default="lsst")
    parser.add_argument("--stage", choices=STAGES, default="m5_with_mw")
    parser.add_argument("--times", default="1,3,6,10", help="Post-merger times in days.")
    parser.add_argument("--time-frame", choices=("observer", "source"), default="observer")
    parser.add_argument(
        "--parameter-pairs", default="Ye_wind:log10_mej_dyn",
        help="x:y pairs separated by commas, e.g. Ye_wind:log10_mej_dyn,Ye_dyn:log10_mej_wind",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--color-ranges", help="One min:max interval per requested time, or a single interval for all times. Use --color-ranges=-0.2:0.5 for negative bounds.")
    selection.add_argument(
        "--bounds-file", type=Path,
        help=(
            "Saved envelope rules, fixed-percentile reference CSV, or a "
            "convergence directory containing percentile_reference_profiles.csv."
        ),
    )
    parser.add_argument(
        "--envelope-family", choices=tuple(ENVELOPE_FAMILIES),
        default="p16-p84",
        help="Statistically valid fixed-percentile envelope read from a reference profile CSV.",
    )
    parser.add_argument(
        "--percentile-pair", default=None,
        help="Deprecated explicit low,high override. Prefer --envelope-family.",
    )
    parser.add_argument("--reference-repetition", type=int, default=0, help="Reference CSV repetition when multiple profiles are saved.")
    parser.add_argument("--m5-scenario", choices=M5_SCENARIOS, default="best_p75")
    parser.add_argument("--threshold-scope", choices=("common_global", "latitude_conditioned"), default="common_global")
    parser.add_argument("--m5-quantile-table", type=Path)
    parser.add_argument("--ztf-m5-g", type=float)
    parser.add_argument("--ztf-m5-r", type=float)
    parser.add_argument("--ztf-m5-i", type=float)
    parser.add_argument("--max-events", type=int, default=0, help="Analysis sample size, 0 = all. This is NOT the display limit.")
    parser.add_argument("--max-scatter-points", type=int, default=25000, help="Display cap for each point category; contours still use every selected event.")
    parser.add_argument("--grid-size", type=int, default=100, help="Number of density cells per axis.")
    parser.add_argument("--smooth-sigma", type=float, default=1.5, help="Gaussian smoothing standard deviation in grid cells.")
    parser.add_argument(
        "--hdr-level", type=int, choices=(50, 90), default=90,
        help="Single highest-density region displayed in both 1D and 2D.",
    )
    parser.add_argument("--min-contour-events", type=int, default=100, help="Display safeguard, not a statistical significance threshold.")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def ask(text, default=""):
    suffix = f" [{default}]" if default != "" else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or str(default)


def choose(title, choices, default):
    print("\n" + title)
    for index, (key, label) in enumerate(choices, 1):
        print(f"  {index}. {label}" + (" [défaut]" if key == default else ""))
    value = ask("Choix", next(i for i, (key, _) in enumerate(choices, 1) if key == default))
    index = int(value) - 1
    if not 0 <= index < len(choices):
        raise ValueError("Choix invalide")
    return choices[index][0]


def interactive(args):
    print("Espace des paramètres KNe sélectionnés par une couleur")
    args.run_dir = Path(ask("Dossier du run (contenant summary.csv)", "."))
    args.color_pair = ask("Paire de couleurs", args.color_pair)
    args.photometric_system = choose("Système photométrique", [("lsst", "LSST"), ("ps1", "PS1 (sélection Rubin sur les bandes LSST sauvegardées)"), ("ztf", "ZTF")], args.photometric_system)
    args.stage = choose("Stade analysé", [
        (STAGES[0], "Sans m5, sans MW"), (STAGES[1], "Sans m5, avec MW"),
        (STAGES[2], "Avec m5, sans MW"), (STAGES[3], "Avec m5, avec MW"),
    ], args.stage)
    args.times = ask("Temps post-merger [jours]", args.times)
    args.time_frame = choose("Référence temporelle", [("observer", "Temps observateur (comme les runs/enveloppes)"), ("source", "Temps source : t_obs = (1+z) t_source")], args.time_frame)
    args.parameter_pairs = ask("Plans x:y (séparer les plans par une virgule)", args.parameter_pairs)
    mode = choose("Sélection par couleur", [("manual", "Saisir un intervalle min:max pour chaque temps"), ("file", "Utiliser les bornes d'un CSV d'enveloppe sauvegardé")], "manual")
    if mode == "manual":
        intervals = []
        print("Exemple de syntaxe : -0.2:0.5 (magnitudes). Pas de valeur scientifique imposée.")
        for time in parse_times(args.times):
            intervals.append(ask(f"Intervalle {args.color_pair} à t={time:g} j"))
        args.color_ranges = ",".join(intervals)
    else:
        args.bounds_file = Path(ask("Dossier de convergence ou CSV de bornes"))
        args.envelope_family = choose(
            "Enveloppe statistique fixe",
            [
                ("p5-p95", "p5-p95 (90 % central)"),
                ("p16-p84", "p16-p84 (68 % central)"),
                ("p25-p75", "p25-p75 (50 % central)"),
            ],
            args.envelope_family,
        )
    if args.stage in ("m5_no_mw", "m5_with_mw"):
        if args.photometric_system == "ztf":
            for band in args.color_pair.split("-"):
                setattr(args, f"ztf_m5_{band}", float(ask(f"Magnitude limite ZTF {band} (5 sigma)")))
        else:
            args.m5_scenario = choose("Seuil m5", [("worst_p25", "p25"), ("median_p50", "p50"), ("best_p75", "p75")], args.m5_scenario)
            explicit = ask("CSV des quantiles m5 (vide = découverte automatique)")
            args.m5_quantile_table = Path(explicit) if explicit else None
    args.max_events = int(ask("Événements analysés (0 = tous)", args.max_events))
    args.hdr_level = int(choose(
        "Contour de densité affiché",
        [(90, "HDR 90 % (recommandé)"), (50, "HDR 50 %")],
        args.hdr_level,
    ))
    output = ask("Dossier de sortie (vide = automatique)")
    args.output_dir = Path(output) if output else None
    return args


def parse_times(text):
    times = np.array([float(item.strip()) for item in text.split(",")], dtype=float)
    if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("Les temps doivent être finis et strictement positifs")
    if len(set(times)) != len(times):
        raise ValueError("Les temps doivent être distincts")
    return times


def parse_parameter_pairs(text):
    pairs = [tuple(item.strip().split(":")) for item in text.split(",")]
    if any(len(pair) != 2 or not all(pair) or pair[0] == pair[1] for pair in pairs):
        raise ValueError("Utiliser des plans x:y avec deux paramètres distincts")
    return pairs


def manual_bounds(text, times):
    intervals = [item.strip().split(":") for item in text.split(",")]
    if len(intervals) == 1:
        intervals *= len(times)
    if len(intervals) != len(times) or any(len(interval) != 2 for interval in intervals):
        raise ValueError("Donner un intervalle min:max, ou un intervalle par temps")
    rows = []
    for time, interval in zip(times, intervals):
        low, high = map(float, interval)
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError("Les bornes couleur doivent être finies et min <= max")
        rows.append({"time_days": float(time), "color_min": low, "color_max": high, "bounds_status": "valid", "bounds_source": "manual"})
    return pd.DataFrame(rows)


def resolve_bounds_path(path):
    """Accept either a bounds CSV or a convergence-analysis directory."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        preferred = path / "percentile_reference_profiles.csv"
        if preferred.is_file():
            return preferred
        matches = sorted(path.glob("**/percentile_reference_profiles.csv"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"No percentile_reference_profiles.csv found below {path}"
            )
        raise ValueError(
            f"Several reference profiles exist below {path}; pass the exact CSV"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def percentile_pair_and_label(args, path):
    """Resolve the selected fixed-percentile family for profile CSV files."""
    columns = set(pd.read_csv(path, nrows=0).columns)
    is_profile = {"percentile", "value", "valid"}.issubset(columns)
    if not is_profile:
        return ENVELOPE_FAMILIES[args.envelope_family], "saved-rules"
    if args.percentile_pair:
        pair = tuple(float(value) for value in args.percentile_pair.split(","))
        if len(pair) != 2 or not 0 < pair[0] < pair[1] < 100:
            raise ValueError("Percentiles required: 0 < q_low < q_high < 100")
        return pair, f"p{pair[0]:g}-p{pair[1]:g}"
    return ENVELOPE_FAMILIES[args.envelope_family], args.envelope_family


def read_bounds(path, times, stage, percentile_pair=(5., 95.), repetition=0):
    """Use the containing bin, not nearest-time extrapolation. No invalid bin is filled."""
    table = pd.read_csv(path)
    if "stage" in table:
        table = table.loc[table.stage.astype(str).eq(stage)].copy()
    if "repetition" in table:
        table = table.loc[pd.to_numeric(table.repetition, errors="coerce").eq(repetition)].copy()
    if "N_requested" in table and not table.empty:
        largest = pd.to_numeric(table.N_requested, errors="coerce").max()
        table = table.loc[pd.to_numeric(table.N_requested, errors="coerce").eq(largest)].copy()
    lower = next((name for name in ("t_min_days", "t_low_days") if name in table), None)
    upper = next((name for name in ("t_max_days", "t_high_days") if name in table), None)
    if lower is None or upper is None:
        raise ValueError("Le CSV des bornes nécessite t_min_days/t_max_days ou t_low_days/t_high_days")
    is_profile = {"percentile", "value", "valid"}.issubset(table.columns)
    if not is_profile and not {"color_min", "color_max"}.issubset(table.columns):
        raise ValueError("Le CSV nécessite color_min/color_max ou percentile/value/valid")
    if is_profile and "estimator_kind" in table:
        table = table.loc[table.estimator_kind.astype(str).eq("fixed")]
    rows = []
    end = pd.to_numeric(table[upper], errors="coerce").max()
    for time in times:
        mask = pd.to_numeric(table[lower], errors="coerce").le(time) & pd.to_numeric(table[upper], errors="coerce").gt(time)
        if np.isclose(time, end, rtol=0, atol=1e-10):
            mask = pd.to_numeric(table[upper], errors="coerce").eq(end) & pd.to_numeric(table[lower], errors="coerce").le(time)
        selected = table.loc[mask]
        row = {"time_days": float(time), "color_min": np.nan, "color_max": np.nan, "bounds_status": "missing_or_invalid", "bounds_source": str(path)}
        if is_profile:
            values = []
            for q in percentile_pair:
                sub = selected.loc[np.isclose(pd.to_numeric(selected.percentile, errors="coerce"), q)]
                if len(sub) > 1:
                    raise ValueError(f"Bornes ambiguës pour t={time:g}, p{q:g}; choisir un CSV de référence unique")
                if len(sub) == 1 and str(sub.valid.iloc[0]).lower() in ("true", "1"):
                    values.append(float(sub.value.iloc[0]))
            if len(values) == 2:
                row.update(color_min=values[0], color_max=values[1])
        elif len(selected) == 1:
            row.update(color_min=float(selected.color_min.iloc[0]), color_max=float(selected.color_max.iloc[0]))
        elif len(selected) > 1:
            raise ValueError(f"Plusieurs bins couvrent t={time:g}; CSV ambigu")
        if np.isfinite(row["color_min"]) and np.isfinite(row["color_max"]) and row["color_min"] <= row["color_max"]:
            row["bounds_status"] = "valid"
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_values(metadata, name):
    """Expose physical mass/inclination transforms without altering stored priors."""
    derived = {
        "mej_dyn": ("log10_mej_dyn", lambda x: 10. ** x),
        "mej_wind": ("log10_mej_wind", lambda x: 10. ** x),
        "inclination_deg": ("inclination_EM", np.rad2deg),
        "cos_inclination": ("inclination_EM", np.cos),
    }
    if name in metadata:
        return pd.to_numeric(metadata[name], errors="coerce").to_numpy(float)
    if name in derived and derived[name][0] in metadata:
        column, transform = derived[name]
        return transform(pd.to_numeric(metadata[column], errors="coerce").to_numpy(float))
    raise ValueError(f"Paramètre '{name}' absent. Colonnes disponibles : {', '.join(metadata.columns)}. Bu2019_MLP n'a pas Ye_dyn/Ye_wind.")


def resolve_root(path):
    root = path.expanduser().resolve()
    return root.parent if root.name in ("parquet", "csv") and (root.parent / "summary.csv").is_file() else root


def load_metadata(root, maximum, seed):
    from dynamic_color_cut_common import load_summary

    metadata = load_summary(root)
    if metadata.empty:
        raise ValueError(f"summary.csv absent ou vide dans {root}")
    if metadata.event_id.duplicated().any():
        raise ValueError("summary.csv contient des identifiants event_id dupliqués")
    if maximum > 0 and maximum < len(metadata):
        positions = np.sort(np.random.default_rng(seed).choice(len(metadata), maximum, replace=False))
        metadata = metadata.iloc[positions].copy()
    return metadata.reset_index(drop=True)


def load_depths(args, root, bands, builder):
    if args.stage not in envelope_core.M5_STAGES:
        return {}, None
    if args.photometric_system == "ztf":
        values = {band: getattr(args, f"ztf_m5_{band}") for band in bands}
        if any(value is None or not np.isfinite(value) or not 0 < value < 40 for value in values.values()):
            raise ValueError("Donner les --ztf-m5-* des deux bandes (valeurs finies entre 0 et 40)")
        return values, "manual_ztf"
    path = builder.discover_m5_table(root, args.m5_quantile_table)
    return builder.load_m5(path, args.threshold_scope, args.m5_scenario, bands), str(path)


def load_event_colours(args, root, metadata, bands, times, depths):
    import build_color_envelopes as builder
    from dynamic_color_cut_common import canonical_event_id, interpolate_band

    source = builder.SyntheticPopulationSource(root)
    if source.kind == "missing":
        raise FileNotFoundError("Aucune courbe synthétique : attendre parquet/synthetic_part_*.parquet ou csv/lightcurve_*.csv sous le dossier du run")
    ids = metadata.event_id.astype(str).tolist()
    position = {event_id: index for index, event_id in enumerate(ids)}
    color = np.full((len(ids), len(times)), np.nan, dtype=float)
    supported = np.zeros(color.shape, dtype=bool)
    depth_pass = np.zeros(color.shape, dtype=bool)
    selection_system = (
        "lsst"
        if args.photometric_system == "ps1"
        and args.stage in envelope_core.M5_STAGES
        else args.photometric_system
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
            raise ValueError(f"Événement {event_id} présent dans plusieurs shards; refus de l'écraser")
        seen.add(event_id)
        index = position[event_id]
        row = metadata.iloc[index]
        evaluation_times = times
        if args.time_frame == "source":
            redshift = float(row.get("redshift", np.nan))
            if not np.isfinite(redshift) or redshift < 0:
                raise ValueError("Le temps source nécessite un redshift fini >= 0 pour chaque événement")
            evaluation_times = times * (1. + redshift)
        frame = frame.copy()
        frame["t_days"] = pd.to_numeric(frame.t_days, errors="coerce")
        frame["band"] = frame.band.astype(str).str.strip().str.lower()
        frame["magnitude"] = pd.to_numeric(frame.mag, errors="coerce")
        if "photometric_system" not in frame:
            frame["photometric_system"] = "lsst"
        frame["photometric_system"] = frame.photometric_system.astype(str).str.strip().str.lower()
        frame = frame.loc[np.isfinite(frame.t_days) & (frame.t_days > 0) & np.isfinite(frame.magnitude)]
        target = frame.loc[frame.photometric_system.eq(args.photometric_system)]
        selection = frame.loc[frame.photometric_system.eq(selection_system)]
        magnitudes = [interpolate_band(target, band, evaluation_times) for band in bands]
        selection_magnitudes = [interpolate_band(selection, band, evaluation_times) for band in bands]
        if args.stage not in envelope_core.MW_EXTINCTED_STAGES:
            magnitudes = [
                values
                - envelope_core.event_extinction_mag(
                    row, args.photometric_system, band
                )
                for values, band in zip(magnitudes, bands)
            ]
            if args.stage in envelope_core.M5_STAGES:
                selection_magnitudes = [
                    values
                    - envelope_core.event_extinction_mag(
                        row, selection_system, band
                    )
                    for values, band in zip(selection_magnitudes, bands)
                ]
        finite = np.isfinite(magnitudes[0]) & np.isfinite(magnitudes[1])
        supported[index] = finite
        color[index] = magnitudes[0] - magnitudes[1]
        passed = finite.copy()
        if args.stage in envelope_core.M5_STAGES:
            for values, band in zip(selection_magnitudes, bands):
                passed &= np.isfinite(values) & (values <= depths[band])
        depth_pass[index] = passed
        if len(seen) % max(1000, len(ids) // 10) == 0:
            print(f"  {len(seen):,}/{len(ids):,}")
    if not supported.any():
        raise ValueError(f"Aucune couleur {args.color_pair} disponible aux temps demandés en {args.photometric_system}. Vérifier les filtres générés et le support temporel.")
    return color, supported, depth_pass


def color_selection(color, available, low, high):
    return available & np.isfinite(color) & (color >= low) & (color <= high)


def axis_domain(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Aucune valeur de paramètre finie")
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        raise ValueError("Paramètre constant : pas de densité 2D identifiable")
    return low, high


def highest_density_regions(x, y, x_domain, y_domain, grid_size=100, sigma=1.5, masses=(.5, .9)):
    """Fast smoothed-density regions calibrated on selected-event counts.

    Reflection at domain boundaries avoids leaking probability outside the
    sampled parameter rectangle. Each selected event receives its interpolated
    density score, using the same padded field as the plot. Thresholds are
    empirical score quantiles, not quantiles of heights at uniformly spaced
    grid cells. Thus ties aside, 50/90% of the actual selected events are inside
    the density-ranked regions. Smoothed model mass is reported separately.
    """
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("x et y doivent être non vides et de même longueur")
    if grid_size < 10 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Grille >= 10 et sigma fini >= 0 nécessaires")
    if any(not 0 < mass < 1 for mass in masses):
        raise ValueError("Les masses de contours doivent être entre 0 et 1")
    histogram, xedges, yedges = np.histogram2d(x, y, bins=grid_size, range=(x_domain, y_domain))
    if not np.isclose(histogram.sum(), len(x)):
        raise ValueError("Tous les événements doivent appartenir au domaine de densité")
    smoothed = gaussian_filter(histogram, sigma=sigma, mode="reflect")
    probability = smoothed / smoothed.sum()
    gx, gy = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
    padded = np.pad(probability, 1, mode="constant")
    scores = map_coordinates(
        padded,
        [(np.asarray(x) - gx[0]) / (gx[1] - gx[0]) + 1.,
         (np.asarray(y) - gy[0]) / (gy[1] - gy[0]) + 1.],
        order=1, mode="constant", cval=0., prefilter=False,
    )
    ranked = np.sort(scores)[::-1]
    regions = []
    for mass in masses:
        level = float(ranked[int(np.ceil(mass * len(ranked))) - 1])
        inside = probability >= level
        regions.append({
            "target_mass": float(mass), "threshold_cell_probability": level,
            "smoothed_mass_inside": float(probability[inside].sum()),
            "empirical_fraction_inside": float((scores >= level).mean()),
        })
    return gx, gy, probability, regions


def highest_density_regions_1d(values, domain, grid_size=100, sigma=1.5,
                               masses=(.5, .9)):
    """Return possibly disconnected 1D density-ranked regions.

    Region thresholds are calibrated on the density scores of the selected
    events, not on uniformly sampled grid cells.  Thus the target mass refers
    to a fraction of the colour-selected simulated population.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("values must contain at least one finite value")
    low, high = map(float, domain)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("invalid 1D density domain")
    histogram, edges = np.histogram(values, bins=grid_size, range=(low, high))
    smoothed = gaussian_filter1d(
        histogram.astype(float), sigma=sigma, mode="reflect"
    )
    probability = smoothed / smoothed.sum()
    centres = .5 * (edges[:-1] + edges[1:])
    scores = np.interp(
        values, centres, probability,
        left=probability[0], right=probability[-1],
    )
    ranked = np.sort(scores)[::-1]
    regions = []
    for mass in masses:
        level = float(ranked[int(np.ceil(mass * len(ranked))) - 1])
        inside = probability >= level
        intervals = []
        start = None
        for index, flag in enumerate(np.r_[inside, False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                intervals.append((float(edges[start]), float(edges[index])))
                start = None
        regions.append({
            "target_mass": float(mass),
            "threshold_cell_probability": level,
            "smoothed_mass_inside": float(probability[inside].sum()),
            "empirical_fraction_inside": float((scores >= level).mean()),
            "intervals": intervals,
        })
    return centres, probability, regions


def scatter_subset(ax, x, y, mask, cap, rng, **style):
    positions = np.flatnonzero(mask)
    if len(positions) > cap:
        positions = rng.choice(positions, cap, replace=False)
    ax.scatter(x[positions], y[positions], rasterized=True, linewidths=0, **style)


def save_projection(args, metadata, times, bounds, color, supported, available, pair, output):
    x_name, y_name = pair
    x, y = parameter_values(metadata, x_name), parameter_values(metadata, y_name)
    finite_parameters = np.isfinite(x) & np.isfinite(y)
    x_domain, y_domain = axis_domain(x), axis_domain(y)
    columns = min(2, len(times))
    rows = math.ceil(len(times) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.1 * columns, 4.8 * rows), sharex=True, sharey=True, squeeze=False)
    contour_rows = []
    rng = np.random.default_rng(args.seed)
    for index, (time, ax) in enumerate(zip(times, axes.flat)):
        bound = bounds.iloc[index]
        selected = color_selection(color[:, index], available[:, index], bound.color_min, bound.color_max)
        selected &= finite_parameters
        rejected = available[:, index] & ~selected & finite_parameters
        unavailable = ~available[:, index] & finite_parameters
        scatter_subset(ax, x, y, unavailable, args.max_scatter_points, rng, color="#e2e5e9", s=2.2, alpha=.27, zorder=1)
        scatter_subset(ax, x, y, rejected, args.max_scatter_points, rng, color="#77818b", s=2.5, alpha=.25, zorder=2)
        scatter_subset(ax, x, y, selected, args.max_scatter_points, rng, color="#e64b4b", s=3., alpha=.30, zorder=3)
        n_selected = int(selected.sum())
        status = "valid"
        if bound.bounds_status != "valid":
            status = "missing_colour_bounds"
            annotation = "No statistically valid colour bounds at this time"
        elif n_selected < args.min_contour_events:
            status = "too_few_selected_events"
            annotation = f"Contour not shown: N < {args.min_contour_events}"
        elif np.ptp(x[selected]) == 0 or np.ptp(y[selected]) == 0:
            status = "degenerate_selected_parameters"
            annotation = "Contour not shown: constant parameter"
        else:
            annotation = ""
            gx, gy, probability, regions = highest_density_regions(
                x[selected], y[selected], x_domain, y_domain,
                args.grid_size, args.smooth_sigma,
                masses=(args.hdr_level / 100.,),
            )
            # Padding only for plotting closes contours at prior-domain edges.
            # It does NOT change probability masses used for the HDR thresholds.
            dx, dy = gx[1] - gx[0], gy[1] - gy[0]
            px, py = np.r_[gx[0] - dx, gx, gx[-1] + dx], np.r_[gy[0] - dy, gy, gy[-1] + dy]
            padded = np.pad(probability, 1, mode="constant")
            for region in regions:
                mass, level = region["target_mass"], region["threshold_cell_probability"]
                colour = "#6d1836" if mass == .5 else "#162e50"
                linestyle = "-" if mass == .5 else "--"
                ax.contour(
                    px, py, padded.T, levels=[level], colors=[colour],
                    linestyles=[linestyle], linewidths=1.75, zorder=4,
                )
                contour_rows.append({"time_days": time, "x_parameter": x_name, "y_parameter": y_name, "N_selected_finite_parameters": n_selected, "status": "valid", **region})
        if status != "valid":
            contour_rows.append({"time_days": time, "x_parameter": x_name, "y_parameter": y_name, "N_selected_finite_parameters": n_selected, "status": status})
            ax.text(.03, .04, annotation, transform=ax.transAxes, fontsize=9, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
        interval = f"{bound.color_min:.3g} ≤ {args.color_pair} ≤ {bound.color_max:.3g}" if bound.bounds_status == "valid" else "bounds unavailable"
        ax.set_title(
            f"t = {time:g} d  ·  {interval}\n"
            f"{n_selected:,} selected / "
            f"{int(available[:, index].sum()):,} colours available",
            fontsize=10.5,
        )
        ax.set_xlim(x_domain)
        ax.set_ylim(y_domain)
        ax.set_xlabel(PARAMETER_LABELS.get(x_name, x_name))
        ax.set_ylabel(PARAMETER_LABELS.get(y_name, y_name))
        ax.grid(alpha=.13)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(times):]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color="#e2e5e9", label="Colour unavailable / rejected by depth cut"),
        Line2D([], [], marker="o", linestyle="none", color="#77818b", label="Available, outside colour interval"),
        Line2D([], [], marker="o", linestyle="none", color="#e64b4b", label="Colour-selected population"),
        Line2D(
            [], [], color="#6d1836" if args.hdr_level == 50 else "#162e50",
            lw=1.75, ls="-" if args.hdr_level == 50 else "--",
            label=f"{args.hdr_level}% HDR of selected population",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    stem = (
        f"parameter_space_{x_name}_vs_{y_name}_"
        f"{args.envelope_label}_HDR{args.hdr_level}"
    )
    plot_title = (
        f"{args.color_pair} {args.envelope_label} colour selection; "
        f"{args.hdr_level}% HDR; {args.photometric_system.upper()}; {args.stage}"
    )
    fig.savefig(
        output / f"{stem}.png", dpi=180,
        metadata={"Title": plot_title},
    )
    plt.close(fig)
    return contour_rows


def save_1d_distribution(args, metadata, times, bounds, color, available,
                         selected_masks, parameter, output):
    """Plot uncluttered available/selected 1D distributions and one HDR."""
    values = parameter_values(metadata, parameter)
    finite = np.isfinite(values)
    domain = axis_domain(values)
    edges = np.linspace(domain[0], domain[1], 61)
    columns = min(2, len(times))
    rows = math.ceil(len(times) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.2 * columns, 4.25 * rows),
        sharex=True, squeeze=False,
    )
    region_rows = []
    for index, (time, ax) in enumerate(zip(times, axes.flat)):
        available_mask = available[:, index] & finite
        selected = selected_masks[index] & finite
        if available_mask.any():
            ax.hist(
                values[available_mask], bins=edges, density=True,
                histtype="step", linewidth=1.5, color="#59636e",
            )
        if selected.any():
            ax.hist(
                values[selected], bins=edges, density=True,
                histtype="stepfilled", alpha=.34, color="#e64b4b",
                edgecolor="#b62424", linewidth=1.2,
            )
        status = "valid"
        if bounds.iloc[index].bounds_status != "valid":
            status = "missing_colour_bounds"
        elif int(selected.sum()) < args.min_contour_events:
            status = "too_few_selected_events"
        elif np.ptp(values[selected]) == 0:
            status = "degenerate_selected_parameter"
        if status == "valid":
            _, _, regions = highest_density_regions_1d(
                values[selected], domain, args.grid_size, args.smooth_sigma,
                masses=(args.hdr_level / 100.,),
            )
            ymax = ax.get_ylim()[1]
            for region in regions:
                mass = region["target_mass"]
                y = ymax * .92
                colour = "#6d1836" if mass == .5 else "#162e50"
                linewidth = 4.2
                for region_index, (region_low, region_high) in enumerate(
                    region["intervals"]
                ):
                    ax.plot(
                        [region_low, region_high], [y, y], color=colour,
                        lw=linewidth, solid_capstyle="butt", zorder=5,
                    )
                    region_rows.append({
                        "time_days": float(time),
                        "parameter": parameter,
                        "target_mass": mass,
                        "region_index": region_index,
                        "region_low": region_low,
                        "region_high": region_high,
                        "N_selected_finite_parameter": int(selected.sum()),
                        "empirical_fraction_inside":
                            region["empirical_fraction_inside"],
                        "smoothed_mass_inside": region["smoothed_mass_inside"],
                        "status": "valid",
                    })
        else:
            region_rows.append({
                "time_days": float(time), "parameter": parameter,
                "N_selected_finite_parameter": int(selected.sum()),
                "status": status,
            })
        bound = bounds.iloc[index]
        interval = (
            f"{bound.color_min:.3g} ≤ {args.color_pair} ≤ {bound.color_max:.3g}"
            if bound.bounds_status == "valid" else "bounds unavailable"
        )
        ax.set_title(
            f"t = {time:g} d  ·  {interval}\n"
            f"{int(selected.sum()):,} selected / "
            f"{int(available_mask.sum()):,} available",
            fontsize=10.2,
        )
        ax.set_xlabel(PARAMETER_LABELS.get(parameter, parameter))
        ax.set_ylabel("Normalised density")
        ax.grid(alpha=.13)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(times):]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], color="#59636e", lw=1.5,
               label="Available population"),
        Line2D([], [], color="#e64b4b", lw=7, alpha=.55,
               label="Colour-selected population"),
        Line2D(
            [], [], color="#6d1836" if args.hdr_level == 50 else "#162e50",
            lw=4.2,
            label=f"{args.hdr_level}% HDR of selected population",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3,
        frameon=False, fontsize=8.5,
    )
    fig.tight_layout(rect=(0, .10, 1, 1))
    stem = (
        f"parameter_distribution_1d_{parameter}_"
        f"{args.envelope_label}_HDR{args.hdr_level}"
    )
    plot_title = (
        f"{parameter}; {args.color_pair} {args.envelope_label} colour "
        f"selection; {args.hdr_level}% HDR; "
        f"{args.photometric_system.upper()}; {args.stage}"
    )
    fig.savefig(
        output / f"{stem}.png", dpi=180,
        metadata={"Title": plot_title},
    )
    plt.close(fig)
    return region_rows


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_dir is None:
        if not sys.stdin.isatty():
            parser.error("--run-dir requis hors terminal interactif")
        args = interactive(args)
    root = resolve_root(args.run_dir)
    times = parse_times(args.times)
    pairs = parse_parameter_pairs(args.parameter_pairs)
    bands = tuple(args.color_pair.strip().lower().split("-"))
    if len(bands) != 2 or bands[0] == bands[1]:
        raise ValueError("Paire de couleurs invalide; exemple g-r")
    allowed = {"lsst": set("ugrizy"), "ps1": set("grizy"), "ztf": set("gri")}
    if not set(bands) <= allowed[args.photometric_system]:
        raise ValueError("Bandes non disponibles dans le système choisi")
    if args.max_events < 0 or args.max_scatter_points < 1 or args.min_contour_events < 3:
        raise ValueError("Limites d'effectifs invalides")
    if args.grid_size < 10 or not np.isfinite(args.smooth_sigma) or args.smooth_sigma < 0:
        raise ValueError("Grille >= 10 et sigma fini >= 0 nécessaires")
    if bool(args.bounds_file) == bool(args.color_ranges):
        raise ValueError("Choisir exactement --color-ranges OU --bounds-file")
    if args.bounds_file and args.time_frame != "observer":
        raise ValueError("Les CSV d'enveloppe du pipeline sont en temps observateur; utiliser des bornes manuelles pour le temps source")
    bounds_path = resolve_bounds_path(args.bounds_file) if args.bounds_file else None
    if bounds_path:
        q, args.envelope_label = percentile_pair_and_label(args, bounds_path)
        bounds = read_bounds(bounds_path, times, args.stage, q, args.reference_repetition)
    else:
        args.envelope_label = "manual"
        bounds = manual_bounds(args.color_ranges, times)
    if not bounds.bounds_status.eq("valid").any():
        raise ValueError("Aucun temps demandé ne possède des bornes couleur valides")
    output = args.output_dir.expanduser().resolve() if args.output_dir else root / "analysis" / f"colour_parameter_space_{args.color_pair.replace('-', '_')}_{args.photometric_system}_{args.stage}"
    if output.is_dir() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Le dossier de sortie n'est pas vide. Choisir un nouveau dossier ou --overwrite.")
    metadata = load_metadata(root, args.max_events, args.seed)
    for pair in pairs:
        for parameter in pair:
            axis_domain(parameter_values(metadata, parameter))
    import build_color_envelopes as builder

    depths, depth_source = load_depths(args, root, bands, builder)
    color, supported, available = load_event_colours(args, root, metadata, bands, times, depths)
    output.mkdir(parents=True, exist_ok=True)
    summaries, contour_rows, region_1d_rows = [], [], []
    selected_masks = []
    for index, time in enumerate(times):
        bound = bounds.iloc[index]
        selected = color_selection(color[:, index], available[:, index], bound.color_min, bound.color_max)
        selected_masks.append(selected)
        n_available = int(available[:, index].sum())
        summaries.append({**bound.to_dict(), "N_total": len(metadata), "N_model_supported": int(supported[:, index].sum()), "N_stage_available": n_available, "N_selected": int(selected.sum()), "selected_fraction_of_available": float(selected.sum() / n_available) if n_available else np.nan, "selected_fraction_of_total": float(selected.mean())})
    for pair in pairs:
        contour_rows.extend(save_projection(args, metadata, times, bounds, color, supported, available, pair, output))
    parameters = list(dict.fromkeys(
        parameter for pair in pairs for parameter in pair
    ))
    for parameter in parameters:
        region_1d_rows.extend(save_1d_distribution(
            args, metadata, times, bounds, color, available,
            selected_masks, parameter, output,
        ))
    pd.DataFrame(summaries).to_csv(output / "selection_summary.csv", index=False)
    pd.DataFrame(contour_rows).to_csv(output / "contour_diagnostics.csv", index=False)
    pd.DataFrame(region_1d_rows).to_csv(
        output / "parameter_1d_regions.csv", index=False
    )
    # Compact wide table: one row/event, not one row/light-curve point.
    event_table = metadata[["event_id"]].copy()
    for parameter in dict.fromkeys(parameter for pair in pairs for parameter in pair):
        event_table[parameter] = parameter_values(metadata, parameter)
    for index, time in enumerate(times):
        tag = f"t{time:g}d".replace(".", "p")
        event_table[f"color_{tag}"] = color[:, index]
        event_table[f"available_{tag}"] = available[:, index]
        event_table[f"selected_{tag}"] = selected_masks[index]
    event_table.to_csv(output / "event_colour_selection.csv.gz", index=False, compression="gzip")
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({
        "run_dir": str(root), "population_split": "none", "applied_depths": depths,
        "depth_source": depth_source, "N_analysed": len(metadata),
        "envelope_label": args.envelope_label,
        "contours": f"{args.hdr_level}% density-ranked region calibrated on selected-event density scores in the displayed coordinates; smoothed histogram; not a confidence interval or selection boundary",
        "interpolation": "saved magnitude linear in log10(observer time); no extrapolation",
        "summary_sha256": hashlib.sha256((root / "summary.csv").read_bytes()).hexdigest(),
        "bounds_sha256": hashlib.sha256(bounds_path.read_bytes()).hexdigest() if bounds_path else None,
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    (output / "parameter_space_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries)[["time_days", "N_total", "N_stage_available", "N_selected", "color_min", "color_max"]].to_string(index=False))
    print("Sorties :", output)


if __name__ == "__main__":
    main()
