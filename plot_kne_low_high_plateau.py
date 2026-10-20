#!/usr/bin/env python3
"""Compare additional MW event losses above and below a latitude cut.

Input: one all-sky event_window_availability.csv, OR separate low/high
availability tables at several thresholds. No light curves are regenerated.
Output: one PNG plot and one PNG table. Undefined ratios remain undefined.

An operational plateau requires all subsequent sampled cuts to lie within
the tolerance of their tail median, including a conservative bootstrap check
on the maximum relative deviation. This is a property of Q(high/low), not a
physical dust boundary or a complete optimisation of a survey selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


REQUIRED = ["event_id", "abs_b_deg", "N_bins_baseline",
            "N_bins_m5_no_mw", "N_bins_m5_with_mw"]
CONFIG_KEYS = ["photometric_system", "color_pair", "t_min_days", "t_max_days",
               "bin_width_days", "m5_scenario", "m5_threshold_scope", "m5_by_band"]


def resolve_table(text):
    path = Path(text).expanduser().resolve()
    if path.is_file():
        return path
    candidates = sorted(path.rglob("event_window_availability.csv")) if path.is_dir() else []
    if len(candidates) != 1:
        raise ValueError(f"Indiquer un CSV exact : {len(candidates)} fichiers trouves sous {path}")
    return candidates[0]


def load_table(text):
    path = resolve_table(text)
    df = pd.read_csv(path, dtype={"event_id": str})
    missing = set(REQUIRED) - set(df)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing)}")
    if df.event_id.isna().any() or df.event_id.duplicated().any():
        raise ValueError(f"event_id doit etre unique et non vide dans {path}")
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    values = df[REQUIRED[1:]].to_numpy(float)
    if not np.isfinite(values).all() or not df.abs_b_deg.between(0, 90).all():
        raise ValueError(f"Latitudes ou comptages invalides dans {path}")
    bins = df[REQUIRED[2:]].to_numpy(float)
    if (bins < 0).any() or not np.equal(bins, np.floor(bins)).all():
        raise ValueError(f"Les nombres de bins doivent etre des entiers positifs ou nuls : {path}")
    if not ((bins[:, 2] <= bins[:, 1]) & (bins[:, 1] <= bins[:, 0])).all():
        raise ValueError(f"Attendu N_with_MW <= N_m5 <= N_baseline : {path}")
    baseline = bins[:, 0] > 0
    m5_loss = baseline & (bins[:, 1] == 0)
    mw_loss = baseline & (bins[:, 1] > 0) & (bins[:, 2] == 0)
    # Columns: baseline denominator, losses from m5, additional MW losses.
    outcomes = np.column_stack([baseline, m5_loss, mw_loss]).astype(int)
    config_path = path.parent / "analysis_configuration.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return df, outcomes, config


def consistent_configs(configs):
    for key in CONFIG_KEYS:
        values = [json.dumps(c[key], sort_keys=True) for c in configs if key in c]
        if len(set(values)) > 1:
            raise ValueError(f"Analyses incompatibles : le parametre {key} differe entre les fichiers")


def high_over_low(counts):
    """Counts shape (..., thresholds, low/high, [baseline, m5 loss, MW loss])."""
    n = counts[..., 0].astype(float)
    losses = counts[..., 2].astype(float)
    rates = np.divide(losses, n, out=np.full_like(losses, np.nan), where=n > 0)
    return np.divide(rates[..., 1], rates[..., 0],
                     out=np.full_like(rates[..., 0], np.nan), where=rates[..., 0] > 0)


def allsky_counts(text, cuts, repetitions, rng):
    df, outcomes, config = load_table(text)
    valid = outcomes[:, 0].astype(bool)
    removed = int((~valid).sum())
    latitude = df.abs_b_deg.to_numpy()[valid]
    outcomes = outcomes[valid]
    if len(outcomes) == 0:
        raise ValueError("Aucune KNe avec couleur de reference")
    # Aggregating into latitude x outcome categories is exactly an event
    # bootstrap for these statistics and preserves overlap across all cuts.
    latitude_cell = np.searchsorted(cuts, latitude, side="right")
    outcome_cell = np.where(outcomes[:, 1] > 0, 0,
                            np.where(outcomes[:, 2] > 0, 1, 2))
    cells = np.bincount(latitude_cell * 3 + outcome_cell,
                        minlength=(len(cuts) + 1) * 3)

    def convert(draws):
        grouped = draws.reshape(*draws.shape[:-1], len(cuts) + 1, 3)
        lower = np.cumsum(grouped, axis=-2)[..., :-1, :]
        higher = grouped.sum(axis=-2)[..., None, :] - lower
        sides = np.stack([lower, higher], axis=-2)
        return np.stack([sides.sum(axis=-1), sides[..., 0], sides[..., 1]], axis=-1)

    observed = convert(cells)
    draws = rng.multinomial(int(cells.sum()), cells / cells.sum(), size=repetitions)
    return observed, convert(draws), [config], f"Bootstrap evenements commun a tous les seuils; {removed} sans couleur exclus"


def pair_counts(raw_pairs, repetitions, rng, mode):
    pairs = sorted(raw_pairs, key=lambda row: float(row[0]))
    cuts = np.array([float(row[0]) for row in pairs])
    runs, configs = [], []
    for cut, low_path, high_path in pairs:
        sides = []
        for side, text in enumerate([low_path, high_path]):
            df, out, config = load_table(text)
            mask = df.abs_b_deg < float(cut) if side == 0 else df.abs_b_deg >= float(cut)
            if not mask.all():
                raise ValueError(f"Des evenements sont du mauvais cote du seuil {cut} : {text}")
            sides.append((df.event_id.tolist(), out))
            configs.append(config)
        runs.append(sides)
    consistent_configs(configs)
    observed = np.array([[out.sum(axis=0) for _, out in pair] for pair in runs])
    boot = np.zeros((repetitions, len(cuts), 2, 3), dtype=int)
    if mode == "paired":
        ids = runs[0][0][0]
        reference = set(ids)
        aligned = []
        for pair in runs:
            aligned_pair = []
            for side_ids, out in pair:
                if set(side_ids) != reference:
                    raise ValueError("Mode apparie : memes event_id requis dans tous les runs")
                order = {identifier: i for i, identifier in enumerate(side_ids)}
                aligned_pair.append(out[[order[identifier] for identifier in ids]])
            aligned.append(aligned_pair)
        for start in range(0, repetitions, 50):
            stop = min(repetitions, start + 50)
            indices = rng.integers(len(ids), size=(stop - start, len(ids)))
            for j, pair in enumerate(aligned):
                for side, out in enumerate(pair):
                    boot[start:stop, j, side] = out[indices].sum(axis=1)
        note = "Bootstrap apparie; event_id doit identifier la meme KNe synthetique dans tous les runs"
    else:
        for j, pair in enumerate(runs):
            for side, (_, out) in enumerate(pair):
                patterns, freq = np.unique(out, axis=0, return_counts=True)
                draws = rng.multinomial(len(out), freq / freq.sum(), size=repetitions)
                boot[:, j, side] = draws @ patterns
        note = "Bootstrap independant; les runs doivent etre des simulations independantes"
    return cuts, observed, boot, configs, note


def tail_deviation(q):
    """Maximum relative distance to the median of a whole candidate tail."""
    q = np.asarray(q, float)
    median = np.median(q, axis=-1)
    valid = np.isfinite(q).all(axis=-1) & (q >= 0).all(axis=-1) & (median > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.max(np.abs(q / median[..., None] - 1), axis=-1)
    return np.where(valid, deviation, np.inf)


def assess_plateau(cuts, q, q_boot, counts, args):
    reliable = ((counts[..., 0] >= args.min_events).all(axis=1)
                & (counts[..., 2] >= args.min_mw_losses).all(axis=1)
                & np.isfinite(q))
    rows = []
    selected = None
    for i, cut in enumerate(cuts):
        enough = (len(cuts) - i >= args.minimum_plateau_points
                  and cuts[-1] - cut >= args.minimum_plateau_span)
        if not enough:
            rows.append(("Etendue insuffisante", np.nan, np.nan))
            continue
        if not reliable[i:].all():
            bad = i + int(np.flatnonzero(~reliable[i:])[0])
            if (counts[bad, :, 0] < args.min_events).any():
                reason = f"Effectif faible a {cuts[bad]:g} deg"
            elif (counts[bad, :, 2] < args.min_mw_losses).any():
                reason = f"Pertes rares a {cuts[bad]:g} deg"
            else:
                reason = f"Rapport indefini a {cuts[bad]:g} deg"
            rows.append((reason, np.nan, np.nan))
            continue
        observed = float(tail_deviation(q[i:]))
        deviations = tail_deviation(q_boot[:, i:])
        # A one-sided percentile bound on this maximum uses the entire tail,
        # rather than declaring stability from overlapping pointwise intervals.
        ordered = np.sort(deviations)
        upper = float(ordered[max(0, int(np.ceil(args.confidence * len(ordered))) - 1)])
        accepted = observed <= args.tolerance and upper <= args.tolerance
        rows.append(("Plateau compatible" if accepted else "Non confirme", observed, upper))
        if accepted and selected is None:
            selected = i
    return selected, rows


def bootstrap_ci(q_boot, confidence):
    alpha = (1 - confidence) / 2
    lower, upper = [], []
    for column in q_boot.T:
        finite = column[np.isfinite(column)]
        bounds = np.quantile(finite, [alpha, 1-alpha]) if len(finite) else [np.nan, np.nan]
        lower.append(bounds[0]); upper.append(bounds[1])
    return np.array(lower), np.array(upper)


def summary_text(cuts, selected, args):
    if selected is None:
        return f"Plateau non demontre sur {cuts[0]:g}-{cuts[-1]:g} deg avec ce critere"
    return (f"Premier plateau operationnel compatible : {cuts[selected]:g} deg "
            f"(teste jusqu'a {cuts[-1]:g} deg)")


def save_outputs(cuts, counts, q, lo, hi, selected, status, configs, note, args):
    output = args.output_dir.expanduser().resolve()
    plot = output / "kne_low_high_mw_plateau.png"
    table_path = output / "kne_low_high_mw_values.png"
    if not args.overwrite and any(p.exists() for p in [plot, table_path]):
        raise FileExistsError("Les sorties existent. Choisir un autre dossier ou ajouter --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    config = next((c for c in configs if c), {})
    details = " | ".join(str(config.get(k, "")) for k in
                          ["color_pair", "photometric_system", "m5_scenario"] if config.get(k))
    if "t_min_days" in config and "t_max_days" in config:
        details += f" | {config['t_min_days']:g}-{config['t_max_days']:g} j"
    summary = summary_text(cuts, selected, args)
    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.25, top=0.96)
    ax.fill_between(cuts, 100*lo, 100*hi, color="#e08616", alpha=.20,
                    label=f"Intervalle bootstrap {args.confidence:.0%} (point par point)")
    ax.plot(cuts, 100*q, "o-", color="#b96007", lw=2.5,
            label="Pertes MW : taux high / taux low")
    if cuts[0] <= args.proposed_cut <= cuts[-1]:
        ax.axvline(args.proposed_cut, color="#4e5966", ls="--", lw=1.6,
                   label=f"Coupure envisagee : {args.proposed_cut:g} deg")
    if selected is not None:
        plateau = np.median(q[selected:]) * 100
        ax.fill_between([cuts[selected], cuts[-1]],
                        plateau*(1-args.tolerance), plateau*(1+args.tolerance),
                        color="#168066", alpha=.16, label="Zone de tolerance du plateau")
        ax.axvline(cuts[selected], color="#168066", lw=2,
                   label=f"Plateau operationnel : {cuts[selected]:g} deg")
    if np.isfinite(hi).any() and np.nanmax(hi) >= .8:
        ax.axhline(100, color="gray", lw=1, ls=":")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Seuil $b_{\rm cut}$ [deg] : low $|b|<b_{\rm cut}$ ; high $|b|\geq b_{\rm cut}$")
    ax.set_ylabel("Taux de perte MW high / low [%]")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=9)
    fig.text(.12, .16, summary, fontsize=12, weight="bold", color="#168066" if selected is not None else "#4e5966")
    fig.text(.12, .115, "Q = 10 % : le taux de perte high vaut 10 % du taux low (90 % de moins).", fontsize=9)
    fig.text(.12, .075, f"Plateau : tous les seuils suivants ; tolerance {args.tolerance:.0%} ; borne bootstrap {args.confidence:.0%}.", fontsize=9)
    fig.text(.12, .035, details or "Perte MW supplementaire apres m5 ; denominateur = KNe avec couleur sans coupure.", fontsize=9, color="#555555")
    fig.savefig(plot, dpi=200)
    plt.close(fig)

    def ratio(value):
        return f"{100*value:.2f}" if np.isfinite(value) else "indefini"

    rows = []
    for i, cut in enumerate(cuts):
        nl, nh = counts[i, :, 0]
        ll, lh = counts[i, :, 2]
        fl = ll/nl if nl else np.nan
        fh = lh/nh if nh else np.nan
        rows.append([f"{cut:g}", f"{ll:,}/{nl:,}", ratio(fl),
                     f"{lh:,}/{nh:,}", ratio(fh),
                     f"{ratio(q[i])} [{ratio(lo[i])}, {ratio(hi[i])}]",
                     ratio(1-q[i]), status[i][0]])
    fig, ax = plt.subplots(figsize=(16, max(5.4, 2.6 + .43*len(rows))))
    ax.axis("off")
    cols = ["Seuil [deg]", "Pertes MW low / N low", "Taux low [%]",
            "Pertes MW high / N high", "Taux high [%]",
            f"Q high/low [%], IC {args.confidence:.0%}", "Baisse 1-Q [%]", "Plateau a partir d'ici"]
    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   colWidths=[.065,.15,.09,.15,.09,.18,.095,.18], bbox=[0,.28,1,.70])
    tab.auto_set_font_size(False); tab.set_fontsize(8.5)
    for (r, _), cell in tab.get_celld().items():
        cell.set_edgecolor("#d5dce5"); cell.set_linewidth(.5)
        if r == 0:
            cell.set_facecolor("#2e425b"); cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#dcefe7" if selected == r-1 else ("#f2f5f8" if r%2 == 0 else "white"))
    fig.text(.06, .21, summary, fontsize=12, weight="bold")
    fig.text(.06, .16, "Une perte MW = couleur possible apres m5 sans poussiere, aucun bin couleur restant apres ajout de MW.", fontsize=9)
    fig.text(.06, .12, "N = KNe avec au moins un bin couleur avant m5 et MW. Les pertes deja dues a m5 ne sont pas comptees comme pertes MW.", fontsize=9)
    fig.text(.06, .08, "Q = taux high / taux low. Les intervalles sont point par point ; le test de plateau porte sur toute la suite des seuils.", fontsize=9)
    fig.text(.06, .04, details or note, fontsize=8.5, color="#555555")
    fig.savefig(table_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return plot, table_path


def parse_cuts(text):
    cuts = np.array(sorted(set(float(x.strip()) for x in text.split(","))))
    if len(cuts) < 3 or not np.isfinite(cuts).all() or not ((cuts > 0) & (cuts < 90)).all():
        raise ValueError("Donner au moins 3 seuils distincts dans ]0,90[ separes par des virgules")
    return cuts


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--event-table", type=Path, help="Disponibilite d'une population all-sky")
    source.add_argument("--pair", nargs=3, action="append", metavar=("SEUIL", "LOW", "HIGH"))
    p.add_argument("--thresholds", default="5,10,15,20,25,30,40,50")
    p.add_argument("--bootstrap-mode", choices=["paired","independent"], default="paired")
    p.add_argument("--bootstrap-repetitions", type=int, default=2000)
    p.add_argument("--confidence", type=float, choices=[.68,.95], default=.95)
    p.add_argument("--tolerance", type=float, default=.10)
    p.add_argument("--minimum-plateau-points", type=int, default=3)
    p.add_argument("--minimum-plateau-span", type=float, default=10)
    p.add_argument("--min-events", type=int, default=100)
    p.add_argument("--min-mw-losses", type=int, default=20)
    p.add_argument("--proposed-cut", type=float, default=20)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--output-dir", type=Path, default=Path("analysis/low_high_plateau"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--interactive", action="store_true")
    return p


def ask(label, default, cast=str):
    return cast(input(f"{label} [{default}] : ").strip() or default)


def interactive(args):
    print("Pertes MW relatives high / low et plateau en latitude")
    mode = ask("1 = un fichier all-sky ; 2 = runs low/high separes", "1")
    if mode == "1":
        args.event_table = Path(ask("event_window_availability.csv all-sky", "event_window_availability.csv"))
        args.pair = None
        args.thresholds = ask("Seuils |b| en deg, separes par des virgules", args.thresholds)
    elif mode == "2":
        args.event_table = None; args.pair = []
        cuts = parse_cuts(ask("Seuils |b| en deg, separes par des virgules", args.thresholds))
        for cut in cuts:
            low = ask(f"CSV (ou dossier) low |b| < {cut:g}", "event_window_availability.csv")
            high = ask(f"CSV (ou dossier) high |b| >= {cut:g}", "event_window_availability.csv")
            args.pair.append([str(cut),low,high])
        value = ask("Memes KNe synthetiques et memes event_id dans tous les runs ? oui/non", "oui").lower()
        if value not in {"oui","non"}:
            raise ValueError("Repondre oui ou non")
        args.bootstrap_mode = "paired" if value == "oui" else "independent"
    else:
        raise ValueError("Choisir 1 ou 2")
    args.bootstrap_repetitions = ask("Repetitions bootstrap", args.bootstrap_repetitions, int)
    args.tolerance = ask("Tolerance relative du plateau (0.10 = 10 %)", args.tolerance, float)
    args.minimum_plateau_span = ask("Etendue minimale apres le debut du plateau [deg]", args.minimum_plateau_span, float)
    args.output_dir = Path(ask("Dossier de sortie", str(args.output_dir)))
    args.overwrite = ask("Remplacer les deux PNG s'ils existent ? oui/non", "non").lower() == "oui"
    return args


def main():
    p = parser(); args = p.parse_args()
    if args.interactive or (len(sys.argv) == 1 and sys.stdin.isatty()):
        args = interactive(args)
    if args.event_table is None and not args.pair:
        p.error("Donner --event-table ou plusieurs --pair, ou lancer --interactive")
    if args.bootstrap_repetitions < 100 or not 0 < args.tolerance < 1:
        p.error("Au moins 100 bootstraps et une tolerance dans ]0,1[ sont requis")
    if args.min_events < 1 or args.min_mw_losses < 1 or args.minimum_plateau_points < 3 or args.minimum_plateau_span <= 0:
        p.error("Effectifs minimum positifs, au moins 3 points et une etendue positive requis")
    rng = np.random.default_rng(args.seed)
    if args.event_table is not None:
        cuts = parse_cuts(args.thresholds)
        counts, boot, configs, note = allsky_counts(args.event_table, cuts, args.bootstrap_repetitions, rng)
    else:
        cuts, counts, boot, configs, note = pair_counts(args.pair, args.bootstrap_repetitions, rng, args.bootstrap_mode)
        if len(cuts) < 3 or len(set(cuts)) != len(cuts) or not ((cuts > 0) & (cuts < 90)).all():
            p.error("Donner au moins 3 seuils distincts dans ]0,90[")
    q = high_over_low(counts); q_boot = high_over_low(boot)
    lo, hi = bootstrap_ci(q_boot, args.confidence)
    selected, status = assess_plateau(cuts, q, q_boot, counts, args)
    plot, table = save_outputs(cuts, counts, q, lo, hi, selected, status, configs, note, args)
    print(summary_text(cuts, selected, args)); print(note)
    for i, cut in enumerate(cuts):
        print(f"{cut:g} deg : low {counts[i,0,2]}/{counts[i,0,0]}, high {counts[i,1,2]}/{counts[i,1,0]}, Q={q[i]:.4g} ; {status[i][0]}")
    print(f"Plot : {plot}"); print(f"Tableau : {table}")


if __name__ == "__main__":
    main()
