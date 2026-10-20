"""Per-bin colour convergence against the largest tested KNe population.

Called by analyze_envelope_population_convergence_fast.py. Bounds are the
builder's selected percentile bounds, never the minimum/maximum individual
event colour. All initial consecutive valid reference bins must pass jointly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPONENTS = ("color_min", "color_median", "color_max")
PROFILE_VERSION = 2


def colour_errors(
    value,
    reference,
    tolerance_mode="absolute",
    relative_tolerance=0.10,
    absolute_tolerance_mag=0.10,
):
    """Return absolute/relative errors and convergence flags.

    ``absolute`` tests ``abs(value-reference) <= absolute_tolerance_mag``.
    ``relative`` tests ``abs(value-reference) <= relative_tolerance*abs(reference)``.
    Relative errors are still recorded for diagnostics in absolute mode, but
    they do not decide convergence. Missing or non-finite values always fail.
    """
    value, reference = np.broadcast_arrays(
        np.asarray(value, dtype=float), np.asarray(reference, dtype=float)
    )
    finite = np.isfinite(value) & np.isfinite(reference)
    difference = np.abs(value - reference)
    relative = np.full(value.shape, np.inf)
    np.divide(difference, np.abs(reference), out=relative,
              where=finite & (reference != 0))
    relative[finite & (reference == 0) & (difference == 0)] = 0.0
    if tolerance_mode == "absolute":
        tolerance = np.full(value.shape, float(absolute_tolerance_mag))
    elif tolerance_mode == "relative":
        tolerance = relative_tolerance * np.abs(reference)
    else:
        raise ValueError("tolerance_mode must be 'absolute' or 'relative'")
    # Roundoff protection only: this does not create a physical mag floor.
    slack = 32 * np.finfo(float).eps * np.maximum.reduce(
        [np.ones(value.shape), np.abs(value), np.abs(reference)]
    )
    passes = finite & (difference <= tolerance + slack)
    return difference, relative, passes


def envelope_profile(a, sample, arrays, indices, depths, bands, edges, centres,
                     reference=None):
    """Compute one full profile, delegating percentile validity to the builder."""
    import build_color_envelopes as builder
    from dynamic_color_cut_common import assign_event_splits, parse_floats

    split = assign_event_splits(sample.event_id.astype(str).tolist(),
                                a.train_fraction, a.validation_fraction, a.builder_seed)
    train_indices = indices[split.split.to_numpy() == "train"]
    key = "colour_extincted" if a.stage in builder.MW_EXTINCTED_STAGES else "colour_intrinsic"
    colours = arrays[key][train_indices]
    available = (np.isfinite(arrays["colour_intrinsic"][train_indices])
                 & np.isfinite(arrays["colour_extincted"][train_indices]))
    if a.stage in builder.M5_STAGES:
        keys = ("mag_sel_int_1", "mag_sel_int_2") if a.stage == "m5_no_mw" else (
            "mag_sel_ext_1", "mag_sel_ext_2")
        for name, band in zip(keys, bands):
            mags = arrays[name][train_indices]
            available &= np.isfinite(mags) & (mags <= depths[band])
    tails = parse_floats(a.tail_percentiles or builder.DEFAULT_TAIL_GRID)
    tails = sorted(set(tails + [25., 30., 35., 40., 42., 44., 45., 46., 47., 48., 49., 50.]))
    if reference is not None and a.bound_mode == "reference-percentiles":
        tails = sorted(set(tails + reference.loc[reference.valid, "low_percentile"].tolist()))
    rng = np.random.default_rng(a.builder_seed + 1000 * builder.STAGE_BOOTSTRAP_SEED_OFFSETS[a.stage])
    rows = []
    active = True
    reference_lookup = reference.set_index("bin_index") if reference is not None else None
    for b, centre in enumerate(centres):
        row = {"bin_index": b, "t_low_days": float(edges[b]),
               "t_high_days": float(edges[b + 1]), "t_center_days": float(centre),
               "valid": False, "status": "after_first_invalid_bin", "N_train": 0,
               "low_percentile": np.nan, "high_percentile": np.nan,
               "color_min": np.nan, "color_median": np.nan, "color_max": np.nan}
        if not active:
            rows.append(row)
            continue
        values = colours[available[:, b], b].astype(float)
        row["N_train"] = len(values)
        if len(values) < a.minimum_events:
            row["status"] = "insufficient_events"
            active = False
            rows.append(row)
            continue
        diagnostics, selected = builder.percentile_diagnostics(
            values, tails, a.n_boot, a.bootstrap_threshold_mag, a.minimum_tail_events, rng)
        if (reference_lookup is not None and a.bound_mode == "reference-percentiles"
                and bool(reference_lookup.loc[b, "valid"])):
            ref = reference_lookup.loc[b]
            qs = [float(ref.low_percentile), 50., float(ref.high_percentile)]
            by_q = diagnostics.set_index("percentile")
            if all(q in by_q.index and bool(by_q.loc[q, "percentile_valid"]) for q in qs):
                selected = {"low_percentile": qs[0], "high_percentile": qs[2],
                            **{name: float(by_q.loc[q, "value"]) for name, q in zip(COMPONENTS, qs)}}
            else:
                selected = None
        if selected is None:
            row["status"] = "no_bootstrap_valid_pair"
            active = False
        else:
            row.update({k: float(selected[k]) for k in (*COMPONENTS, "low_percentile", "high_percentile")})
            if selected["low_percentile"] >= selected["high_percentile"] and not a.allow_degenerate_envelope:
                row["status"] = "degenerate_p50_p50_excluded"
                active = False
            else:
                row.update(valid=True, status="valid")
        rows.append(row)
    return pd.DataFrame(rows)


def compare_profiles(
    profile,
    reference,
    tolerance_mode="absolute",
    relative_tolerance=0.10,
    absolute_tolerance_mag=0.10,
):
    """Compare every reference-valid bin; absent sample bins always fail."""
    ref = reference.loc[reference.valid].copy()
    if ref.empty:
        raise ValueError("Largest-N reference has no valid envelope bin")
    other = profile.set_index("bin_index").reindex(ref.bin_index)
    out = ref[["bin_index", "t_low_days", "t_high_days", "t_center_days"]].copy().reset_index(drop=True)
    out["reference_N_train"] = ref.N_train.to_numpy()
    out["sample_N_train"] = other.N_train.to_numpy()
    supported = other.valid.eq(True).to_numpy(dtype=bool)
    out["sample_bin_valid"] = supported
    out["reference_low_percentile"] = ref.low_percentile.to_numpy()
    out["reference_high_percentile"] = ref.high_percentile.to_numpy()
    out["sample_low_percentile"] = other.low_percentile.to_numpy()
    out["sample_high_percentile"] = other.high_percentile.to_numpy()
    joint = supported.copy()
    for name in COMPONENTS:
        values = other[name].to_numpy(float)
        targets = ref[name].to_numpy(float)
        difference, relative, passes = colour_errors(
            values,
            targets,
            tolerance_mode=tolerance_mode,
            relative_tolerance=relative_tolerance,
            absolute_tolerance_mag=absolute_tolerance_mag,
        )
        passes &= supported
        relative[~supported] = np.inf
        out[f"sample_{name}"] = values
        out[f"reference_{name}"] = targets
        out[f"absolute_error_{name}_mag"] = difference
        out[f"relative_error_{name}"] = relative
        out[f"passes_{name}"] = passes
        joint &= passes
    out["all_three_pass"] = joint
    return out


def repetition_record(comparison, rep, size, subset_seed, tolerance_mode="absolute"):
    record = {"repetition": rep, "subsample_seed": subset_seed, "N_requested": size,
              "N_reference_bins": len(comparison),
              "N_supported_reference_bins": int(comparison.sample_bin_valid.sum()),
              "N_bins_all_three_pass": int(comparison.all_three_pass.sum()),
              "all_reference_bins_pass": bool(comparison.all_three_pass.all())}
    absolute_errors = []
    relative_errors = []
    for name in COMPONENTS:
        absolute_error = float(comparison[f"absolute_error_{name}_mag"].max())
        relative_error = float(comparison[f"relative_error_{name}"].max())
        record[f"max_absolute_error_{name}_mag"] = absolute_error
        record[f"max_relative_error_{name}"] = relative_error
        record[f"all_bins_pass_{name}"] = bool(comparison[f"passes_{name}"].all())
        absolute_errors.append(absolute_error)
        relative_errors.append(relative_error)
    record["max_absolute_error_all_mag"] = max(absolute_errors)
    record["max_relative_error_all"] = max(relative_errors)
    if tolerance_mode == "absolute":
        record["max_metric_error_all"] = record["max_absolute_error_all_mag"]
    elif tolerance_mode == "relative":
        record["max_metric_error_all"] = record["max_relative_error_all"]
    else:
        raise ValueError("tolerance_mode must be 'absolute' or 'relative'")
    return record


def safe_quantile(values, q):
    """Quantile that preserves infinite errors instead of producing NaN."""
    v = np.sort(np.asarray(values, float))
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan
    p = (len(v) - 1) * q
    i, j = int(np.floor(p)), int(np.ceil(p))
    if i == j:
        return float(v[i])
    if np.isinf(v[j]):
        return np.inf
    return float(v[i] + (p - i) * (v[j] - v[i]))


def aggregate_colour_results(raw, required_fraction=0.9):
    rows = []
    for size, group in raw.groupby("N_requested", sort=True):
        fraction = float(group.all_reference_bins_pass.mean())
        row = {"N_requested": int(size), "repetitions": len(group),
               "fraction_all_bins_all_three_pass": fraction,
               "point_passes_stability": bool(fraction + 1e-12 >= required_fraction),
               "minimum_supported_reference_bins": int(group.N_supported_reference_bins.min())}
        metric_columns = [
            *(f"max_absolute_error_{name}_mag" for name in (*COMPONENTS, "all")),
            *(f"max_relative_error_{name}" for name in (*COMPONENTS, "all")),
            "max_metric_error_all",
        ]
        for column in metric_columns:
            for label, q in (("median", .5), ("p16", .16), ("p84", .84)):
                row[f"{column}_{label}"] = safe_quantile(group[column], q)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["all_larger_sizes_also_stable"] = np.logical_and.accumulate(
        summary.point_passes_stability.to_numpy()[::-1])[::-1]
    candidates = summary.loc[summary.all_larger_sizes_also_stable, "N_requested"]
    threshold = int(candidates.iloc[0]) if len(candidates) else None
    return summary, threshold


def save_colour_plot(raw, summary, a, threshold, path):
    fig, (ax, frac_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    x = summary.N_requested.to_numpy(float)
    absolute_mode = a.tolerance_mode == "absolute"
    factor = 1.0 if absolute_mode else 100.0
    finite = raw.max_metric_error_all.to_numpy(float) * factor
    finite = finite[np.isfinite(finite)]
    requested_tolerance = (
        a.absolute_tolerance_mag if absolute_mode else 100.0 * a.relative_tolerance
    )
    minimum_cap = 0.30 if absolute_mode else 30.0
    maximum_cap = 3.0 if absolute_mode else 1000.0
    cap = max(minimum_cap, requested_tolerance * 3,
              min(maximum_cap, float(np.max(finite)) * 1.05 if len(finite) else minimum_cap))
    def clip(values):
        return np.minimum(np.asarray(values, float) * factor, cap)
    for _, group in raw.groupby("repetition"):
        group = group.sort_values("N_requested")
        ax.plot(group.N_requested, clip(group.max_metric_error_all),
                color="0.65", alpha=.35, linewidth=.8)
    ax.fill_between(x, clip(summary.max_metric_error_all_p16),
                    clip(summary.max_metric_error_all_p84), color="#2471a3", alpha=.18,
                    label="16–84% des sous-échantillons")
    ax.plot(x, clip(summary.max_metric_error_all_median), "o-", color="#2471a3",
            label="Médiane de l'écart maximal")
    ax.axhline(requested_tolerance, color="#a93226", linestyle="--",
               label=(f"Tolérance absolue : {a.absolute_tolerance_mag:g} mag"
                      if absolute_mode else
                      f"Tolérance relative : {100 * a.relative_tolerance:g}%"))
    ax.set_ylim(0, cap * 1.06)
    ax.set_ylabel(("Écart absolu maximal (mag)\n"
                   if absolute_mode else "Écart relatif maximal (%)\n")
                  + "médiane et deux bornes, tous les bins")
    if np.any(~np.isfinite(raw.max_metric_error_all)) or np.any(finite > cap):
        ax.text(.02, .98, "Valeurs infinies ou hors échelle plafonnées en haut\n"
                "(bin de référence non supporté par le sous-échantillon)", transform=ax.transAxes,
                va="top", fontsize=8)
    ax.legend(loc="best", fontsize=9)
    frac_ax.plot(x, 100 * summary.fraction_all_bins_all_three_pass, "o-", color="#2471a3")
    frac_ax.axhline(100 * a.required_fraction, color="#a93226", linestyle="--")
    frac_ax.set_ylim(-3, 103)
    frac_ax.set_ylabel("Répétitions\ncompatibles (%)")
    frac_ax.set_xlabel("Nombre de KNe injectées")
    for axis in (ax, frac_ax):
        axis.set_xscale("log")
        axis.grid(alpha=.2)
        if threshold is not None:
            axis.axvline(threshold, color="#287d46", linestyle=":")
    if threshold is not None:
        frac_ax.text(.98, .07, f"N requis sur la grille : {threshold:,}".replace(",", " "),
                     transform=frac_ax.transAxes, ha="right", color="#287d46")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def atomic_csv(frame, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def run_colour_convergence(a, root, sizes, bands, edges, centres):
    import analyze_envelope_population_convergence_fast as fast
    import build_color_envelopes as builder
    import dynamic_color_cut_common as common

    if a.tolerance_mode not in {"absolute", "relative"}:
        raise ValueError("tolerance-mode must be 'absolute' or 'relative'")
    if not np.isfinite(a.relative_tolerance) or a.relative_tolerance <= 0:
        raise ValueError("relative-tolerance must be finite and positive")
    if not np.isfinite(a.absolute_tolerance_mag) or a.absolute_tolerance_mag <= 0:
        raise ValueError("absolute-tolerance-mag must be finite and positive")
    if a.minimum_events < 1 or a.minimum_tail_events < 0 or a.bootstrap_threshold_mag < 0:
        raise ValueError("Invalid event-count or bootstrap-validity limits")
    if not (0 < a.train_fraction < 1 and 0 <= a.validation_fraction < 1
            and a.train_fraction + a.validation_fraction < 1):
        raise ValueError("Invalid train/validation fractions")
    output = a.output_dir.expanduser().resolve() if a.output_dir else (
        root / "analysis" / f"convergence_colours_{a.color_pair.replace('-', '_')}_{a.photometric_system}_{a.stage}")
    output.mkdir(parents=True, exist_ok=True)
    bargs = fast.make_builder_args(a, root)
    depths = fast.get_depths(a, bargs, root, bands) if a.stage in builder.M5_STAGES else {}
    conf = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
            if k not in {"resume", "output_dir", "fit_exponential", "tolerance_days"}}
    conf.update(method="per_bin_colour_convergence", profile_version=PROFILE_VERSION,
                run_dir=str(root), sample_sizes=sizes, applied_depths=depths,
                comparison="all three quantities in every initial consecutive valid reference bin",
                tolerance_mode=a.tolerance_mode,
                tolerance=(f"absolute difference <= {a.absolute_tolerance_mag:g} mag"
                           if a.tolerance_mode == "absolute" else
                           f"relative difference <= {a.relative_tolerance:g}"),
                reference="largest tested N, paired by subsampling repetition")
    # Do not reuse profiles silently after changing the builder, input summary,
    # or any recorded photometry shard. Content hash for code/summary; stat for
    # potentially enormous photometry products.
    conf["source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path(fast.__file__), Path(builder.__file__),
        Path(common.__file__), root / "summary.csv")}
    files = sorted((root / "parquet").glob("synthetic_part_*.parquet"))
    if not files:
        files = sorted((root / "csv").glob("lightcurve_*.csv"))
    conf["photometry_signature"] = hashlib.sha256(json.dumps([
        [str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns] for p in files
    ]).encode()).hexdigest()
    conf_path = output / "colour_configuration.json"
    if a.resume and conf_path.exists() and json.loads(conf_path.read_text()) != conf:
        raise ValueError("Colour-run configuration differs; choose a new output directory or --no-resume")
    conf_path.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")
    checkpoint_dir = output / "colour_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tasks = [(rep, size) for rep in range(a.repetitions) for size in sizes]
    missing = any(not (checkpoint_dir / f"rep_{rep:03d}_N_{size:09d}.csv").is_file()
                  for rep, size in tasks)
    population = None
    if missing or not a.resume:
        metadata, event_ids, arrays = fast.load_population_once(a, root, bands, centres, sizes)
        population = (metadata, arrays, {eid: i for i, eid in enumerate(event_ids)})
    all_profiles, comparisons, records, references = [], [], [], []
    counter = 0
    for rep in range(a.repetitions):
        reference = None
        seed = a.subsample_seed + rep
        # Compute the largest sample first, so each repetition has its own
        # reference and optional reference-percentile targets.
        for size in [sizes[-1], *sizes[:-1]]:
            counter += 1
            checkpoint = checkpoint_dir / f"rep_{rep:03d}_N_{size:09d}.csv"
            if a.resume and checkpoint.is_file():
                profile = pd.read_csv(checkpoint, float_precision="round_trip")
                print(f"[{counter}/{len(tasks)}] reusing repetition={rep + 1}, N={size:,}")
            else:
                print(f"[{counter}/{len(tasks)}] repetition={rep + 1}/{a.repetitions}, N={size:,}")
                metadata, arrays, position = population
                sample = fast.select_loaded_metadata_subset(metadata, size, seed)
                indices = np.asarray([position[eid] for eid in sample.event_id.astype(str)], dtype=int)
                profile = envelope_profile(a, sample, arrays, indices, depths, bands, edges, centres,
                                           reference=reference)
                atomic_csv(profile, checkpoint)
            if size == sizes[-1]:
                reference = profile
                if not profile.valid.any():
                    raise ValueError(f"No valid reference envelope for repetition {rep + 1}; "
                                     "check the saved filter system, depth, and validity settings")
                refs = profile.loc[profile.valid].copy()
                refs.insert(0, "repetition", rep)
                references.append(refs)
            tagged = profile.copy()
            tagged.insert(0, "N_requested", size)
            tagged.insert(0, "repetition", rep)
            all_profiles.append(tagged)
            comparison = compare_profiles(
                profile,
                reference,
                tolerance_mode=a.tolerance_mode,
                relative_tolerance=a.relative_tolerance,
                absolute_tolerance_mag=a.absolute_tolerance_mag,
            )
            record = repetition_record(
                comparison, rep, size, seed, tolerance_mode=a.tolerance_mode
            )
            records.append(record)
            comparison.insert(0, "N_requested", size)
            comparison.insert(0, "repetition", rep)
            comparisons.append(comparison)
    raw = pd.DataFrame(records).sort_values(["repetition", "N_requested"])
    summary, threshold = aggregate_colour_results(raw, a.required_fraction)
    atomic_csv(raw, output / "envelope_colour_convergence_raw.csv")
    atomic_csv(summary, output / "envelope_colour_convergence_summary.csv")
    atomic_csv(pd.concat(comparisons, ignore_index=True), output / "envelope_colour_convergence_per_bin.csv")
    atomic_csv(pd.concat(all_profiles, ignore_index=True), output / "envelope_colour_profiles.csv")
    atomic_csv(pd.concat(references, ignore_index=True), output / "envelope_colour_reference.csv")
    plot_path = output / "envelope_colour_convergence.png"
    save_colour_plot(raw, summary, a, threshold, plot_path)
    result = dict(conf, stable_population_threshold=threshold,
                  below_largest_tested_size=bool(threshold is not None and threshold < sizes[-1]),
                  stability_rule="first tested N where the required fraction of repetitions passes all three colours in all reference bins, and every larger tested N also passes",
                  scientific_scope="conditional convergence to the largest saved sample, not an asymptotic guarantee")
    (output / "envelope_colour_convergence_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    last = pd.concat(references).groupby("repetition").t_high_days.max()
    print("Reference valid envelope last edges [days]:", last.tolist())
    print("Stable colour-population threshold:", f"{threshold:,} KNe" if threshold is not None else "not determined")
    if threshold == sizes[-1]:
        print("Only the largest tested size passes: it is compared with itself; no plateau below it is established.")
    print("Plot:", plot_path)
    print("Summary:", output / "envelope_colour_convergence_summary.csv")
    print("Per-bin comparisons:", output / "envelope_colour_convergence_per_bin.csv")

