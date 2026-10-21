#!/usr/bin/env python3
"""Interactive orchestration of paired KNe populations in latitude bins.

This workflow creates or resumes one population per disjoint Galactic-latitude
bin, using a shared intrinsic catalogue, then runs the event-level latitude
analysis and paired-bin comparison.  It calls the public Python functions
directly: no hidden command-line options or subprocess command strings are
constructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kne_m5 import M5_SCENARIOS


DEFAULT_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 90.0)
M5_REQUIRED_COLUMNS = {
    "threshold_scope", "threshold_latitude_group", "band", "scenario", "m5",
}
EVENT_REQUIRED_COLUMNS = {
    "event_id", "abs_b_deg", "N_bins_baseline",
    "N_bins_m5_no_mw", "N_bins_m5_with_mw",
}

# =============================================================================
# USER-FRIENDLY TERMINAL INTERFACE AND CONFIGURATION
# =============================================================================

def prompt_text(label: str, default="") -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or str(default)


def prompt_yes_no(label: str, default=True) -> bool:
    answer = input(f"{label} [{'O/n' if default else 'o/N'}]: ").strip().lower()
    if not answer:
        return default
    if answer not in {"o", "oui", "y", "yes", "n", "non", "no"}:
        raise ValueError("Répondre oui ou non")
    return answer in {"o", "oui", "y", "yes"}


def choose(title: str, choices, default):
    print(f"\n{title}")
    default_index = next(i for i, (value, _) in enumerate(choices, 1) if value == default)
    for index, (value, label) in enumerate(choices, 1):
        suffix = " [défaut]" if value == default else ""
        print(f"  {index}. {label}{suffix}")
    answer = input(f"Choix [{default_index}]: ").strip()
    if not answer:
        return default
    index = int(answer)
    if not 1 <= index <= len(choices):
        raise ValueError(f"Le choix doit être compris entre 1 et {len(choices)}")
    return choices[index - 1][0]


def default_configuration() -> argparse.Namespace:
    return argparse.Namespace(
        project_root=Path("."), opsim_dir=None, runs_root=None,
        intrinsic_catalog=None, m5_quantile_table=None,
        bin_edges=list(DEFAULT_EDGES), n_samples=2000, intrinsic_seed=12345,
        distance_min_mpc=10.0, distance_max_mpc=600.0,
        synthetic_cadence_days=0.1, m5_scenario="best_p75",
        t_min=0.4, t_max=16.0, bin_width=0.4,
        bootstrap_repetitions=2000, confidence_level=0.95,
        plateau_relative_tolerance=0.10, minimum_plateau_bins=3,
        comparison_seed=20260930, output_dir=None,
        overwrite_final=False, dry_run=False,
    )


def interactive_configuration(args: argparse.Namespace) -> argparse.Namespace:
    print("Expérience KNe appariée par tranches de latitude galactique")
    args.project_root = Path(prompt_text("Dossier du projet", Path.cwd()))
    args.opsim_dir = Path(prompt_text("Dossier des exports OpSim", "opsim_exports_all_survey"))
    args.runs_root = Path(prompt_text("Dossier des runs", "runs"))
    args.intrinsic_catalog = Path(prompt_text(
        "Catalogue intrinsèque partagé", "paired_intrinsic_population_uniform.csv"
    ))
    explicit_m5 = prompt_text("Table m5 (vide = recherche automatique)", "")
    args.m5_quantile_table = Path(explicit_m5) if explicit_m5 else None
    args.bin_edges = [float(value.strip()) for value in prompt_text(
        "Bornes des tranches |b| [deg]", ",".join(f"{x:g}" for x in args.bin_edges)
    ).split(",") if value.strip()]
    args.n_samples = int(prompt_text("KNe par tranche", args.n_samples))
    args.intrinsic_seed = int(prompt_text("Seed intrinsèque", args.intrinsic_seed))
    args.distance_min_mpc = float(prompt_text("Distance minimale [Mpc]", args.distance_min_mpc))
    args.distance_max_mpc = float(prompt_text("Distance maximale [Mpc]", args.distance_max_mpc))
    args.synthetic_cadence_days = float(prompt_text(
        "Cadence synthétique [jours]", args.synthetic_cadence_days
    ))
    args.m5_scenario = choose(
        "Seuil m5", [("worst_p25", "p25"), ("median_p50", "p50"),
                       ("best_p75", "p75")], args.m5_scenario,
    )
    args.t_min = float(prompt_text("Début de la fenêtre [jours]", args.t_min))
    args.t_max = float(prompt_text("Fin de la fenêtre [jours]", args.t_max))
    args.bin_width = float(prompt_text("Largeur des bins temporels [jours]", args.bin_width))
    args.bootstrap_repetitions = int(prompt_text(
        "Répétitions bootstrap", args.bootstrap_repetitions
    ))
    args.confidence_level = float(choose(
        "Intervalle de confiance", [(0.68, "68 %"), (0.95, "95 %")],
        args.confidence_level,
    ))
    args.plateau_relative_tolerance = float(prompt_text(
        "Tolérance relative du plateau", args.plateau_relative_tolerance
    ))
    args.minimum_plateau_bins = int(prompt_text(
        "Nombre minimum de tranches dans le plateau", args.minimum_plateau_bins
    ))
    output = prompt_text("Dossier de sortie globale (vide = automatique)", "")
    args.output_dir = Path(output) if output else None
    args.overwrite_final = prompt_yes_no("Remplacer les figures finales existantes", False)
    args.dry_run = prompt_yes_no("Afficher seulement le plan sans lancer les calculs", False)
    return args

# =============================================================================
# FILE VALIDATION AND REPRODUCIBILITY HELPERS
# =============================================================================

class BinSpec:
    lower: float
    upper: float
    sky_seed: int
    run_name: str
    run_dir: Path
    analysis_dir: Path

    @property
    def event_table(self) -> Path:
        return self.analysis_dir / "event_window_availability.csv"

    @property
    def label(self) -> str:
        closing = "]" if math.isclose(self.upper, 90.0) else "["
        return f"[{self.lower:g}, {self.upper:g}{closing}"

def absolute(path: Path, base: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else base / path).resolve()

def number_label(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):02d}"
    return f"{value:g}".replace(".", "p")

def number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"

def derive_bin_sky_seed(intrinsic_seed: int, lower: float, upper: float) -> int:
    lower_code = int(round(float(lower) * 1000.0))
    upper_code = int(round(float(upper) * 1000.0))
    value = int(intrinsic_seed) ^ 0x534B59 ^ 0xB17 ^ lower_code
    value ^= (upper_code * 0x9E3779B1) & 0xFFFFFFFF
    return int(value % (2**32))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def csv_header(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return set(next(csv.reader(handle), []))

def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for row in reader if row)

def read_first_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.DictReader(handle), {})

def validate_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} introuvable : {path}")

def validate_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} introuvable : {path}")

def validate_catalog(path: Path, n_samples: int, seed: int) -> str:
    validate_file(path, "Catalogue intrinsèque")
    required = {
        "event_id",
        "event_index",
        "intrinsic_seed",
        "parameter_source",
        "distance_sampling",
        "log10_mej_dyn",
        "v_ej_dyn",
        "Ye_dyn",
        "log10_mej_wind",
        "v_ej_wind",
        "Ye_wind",
        "inclination_EM",
        "redshift",
        "luminosity_distance",
    }
    missing = required - csv_header(path)
    if missing:
        raise ValueError(f"Catalogue incompatible; colonnes absentes : {sorted(missing)}")
    rows = csv_row_count(path)
    if rows != n_samples:
        raise ValueError(f"Catalogue: {rows:,} événements au lieu de {n_samples:,}")
    saved_seed = read_first_row(path).get("intrinsic_seed", "")
    if saved_seed and saved_seed != str(seed):
        raise ValueError(f"Catalogue intrinsic_seed={saved_seed}, attendu={seed}")
    return sha256(path)

def validate_m5_table(path: Path, scenario: str) -> tuple[float, float]:
    validate_file(path, "Table m5")
    missing = M5_REQUIRED_COLUMNS - csv_header(path)
    if missing:
        raise ValueError(f"Table m5 incompatible; colonnes absentes : {sorted(missing)}")
    selected: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("threshold_scope") == "common_global"
                and row.get("threshold_latitude_group") == "both"
                and row.get("scenario") == scenario
                and row.get("band") in {"g", "r", "lsstg", "lsstr"}
            ):
                selected[str(row["band"])[-1]] = float(row["m5"])
    if set(selected) != {"g", "r"}:
        raise ValueError(
            f"Seuils common_global/both/{scenario} pour g et r absents de {path}"
        )
    return selected["g"], selected["r"]

def discover_m5_table(root: Path, opsim: Path, scenario: str) -> Path | None:
    preferred = [
        root / "m5_depth_quantiles_by_band.csv",
        opsim / "m5_depth_quantiles_by_band.csv",
        root / "analysis" / "latitude_depth_comparison" / "m5_depth_quantiles_by_band.csv",
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for path in [*preferred, *root.rglob("m5_depth_quantiles_by_band.csv")]:
        resolved = path.resolve()
        if resolved.is_file() and resolved not in seen:
            seen.add(resolved)
            try:
                validate_m5_table(resolved, scenario)
                candidates.append(resolved)
            except (OSError, ValueError):
                pass
    if not candidates:
        return None
    values = {validate_m5_table(path, scenario) for path in candidates}
    return candidates[0] if len(values) == 1 else None

def make_specs(
    edges: Sequence[float],
    runs_root: Path,
    seed: int,
    analysis_name: str = "latitude_bin_event_availability_g_r_best_p75_0p4to16d",
) -> list[BinSpec]:
    specs = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        sky_seed = derive_bin_sky_seed(seed, lower, upper)
        run_name = f"paired_b{number_label(lower)}to{number_label(upper)}_bin"
        run_dir = runs_root / f"{run_name}_intrinsic_seed_{seed}_sky_seed_{sky_seed}"
        analysis_dir = run_dir / "analysis" / analysis_name
        specs.append(BinSpec(lower, upper, sky_seed, run_name, run_dir, analysis_dir))
    return specs

def validate_run(spec: BinSpec, n_samples: int, catalog_hash: str) -> bool:
    if not spec.run_dir.exists():
        return False
    summary = spec.run_dir / "summary.csv"
    manifest = spec.run_dir / "parquet" / "parquet_manifest.csv"
    if not summary.is_file() or not manifest.is_file():
        raise RuntimeError(
            f"Run incomplet : {spec.run_dir}\n"
            "Il n'est pas écrasé. Vérifiez puis renommez/supprimez ce dossier avant de relancer."
        )
    if csv_row_count(summary) != n_samples:
        raise RuntimeError(f"Nombre d'événements incompatible dans {summary}")
    header = csv_header(summary)
    required = {"event_id", "intrinsic_catalog_sha256", "abs_galactic_latitude_deg"}
    if not required.issubset(header):
        raise RuntimeError(f"Colonnes absentes de {summary}: {sorted(required-header)}")
    first = read_first_row(summary)
    if first.get("intrinsic_catalog_sha256", "") not in {"", catalog_hash}:
        raise RuntimeError(f"Catalogue intrinsèque différent dans {summary}")
    with summary.open(newline="", encoding="utf-8-sig") as handle:
        latitude = [float(row["abs_galactic_latitude_deg"]) for row in csv.DictReader(handle)]
    eps = 1e-8
    if any(value < spec.lower - eps for value in latitude):
        raise RuntimeError(f"Un événement est sous la borne de {spec.label} dans {summary}")
    if math.isclose(spec.upper, 90.0):
        outside_upper = any(value > spec.upper + eps for value in latitude)
    else:
        outside_upper = any(value >= spec.upper + eps for value in latitude)
    if outside_upper:
        raise RuntimeError(f"Un événement dépasse la borne de {spec.label} dans {summary}")
    if csv_row_count(manifest) < 1:
        raise RuntimeError(f"Manifest Parquet vide : {manifest}")
    return True

def validate_event_table(path: Path, n_samples: int) -> bool:
    if not path.exists():
        return False
    missing = EVENT_REQUIRED_COLUMNS - csv_header(path)
    if missing:
        raise RuntimeError(f"Table incompatible {path}; colonnes absentes : {sorted(missing)}")
    if csv_row_count(path) != n_samples:
        raise RuntimeError(f"Table {path}: nombre de lignes incompatible")
    return True

# =============================================================================
# PATH RESOLUTION, RESUME CHECKS AND SCIENTIFIC CONFIGURATION
# =============================================================================

def resolve_paths(args: argparse.Namespace) -> tuple[argparse.Namespace, dict[str, Path]]:
    root = args.project_root.expanduser().resolve()
    paths = {
        "root": root,
        "opsim": absolute(args.opsim_dir or Path("opsim_exports_all_survey"), root),
        "runs": absolute(args.runs_root or Path("runs"), root),
        "catalog": absolute(
            args.intrinsic_catalog or Path("paired_intrinsic_population_uniform.csv"), root
        ),
    }
    m5 = absolute(args.m5_quantile_table, root) if args.m5_quantile_table else None
    if m5 is None:
        m5 = discover_m5_table(root, paths["opsim"], args.m5_scenario)
    if m5 is None:
        entered = prompt_text("Table m5 introuvable, indiquer son chemin", "m5_depth_quantiles_by_band.csv")
        m5 = absolute(Path(entered), root)
    paths["m5"] = m5
    if args.output_dir:
        output = absolute(args.output_dir, root)
    else:
        edge_slug = "_".join(number_label(value) for value in args.bin_edges)
        output = root / "analysis" / (
            f"paired_latitude_bins_{edge_slug}_comoving_"
            f"{args.distance_min_mpc:g}to{args.distance_max_mpc:g}Mpc_seed_{args.intrinsic_seed}"
        )
    paths["output"] = output.resolve()
    return args, paths


def validate_configuration(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    edges = [float(value) for value in args.bin_edges]
    if len(edges) < 5:
        raise ValueError("Il faut au moins quatre tranches de latitude")
    if edges != sorted(edges) or len(set(edges)) != len(edges):
        raise ValueError("Les bornes doivent être strictement croissantes")
    if edges[0] < 0 or edges[-1] > 90:
        raise ValueError("Les bornes doivent rester dans [0, 90]")
    args.bin_edges = edges
    if args.n_samples < 1:
        raise ValueError("Le nombre de KNe doit être positif")
    if args.bootstrap_repetitions < 100:
        raise ValueError("Utiliser au moins 100 répétitions bootstrap")
    if not 0 < args.plateau_relative_tolerance < 1:
        raise ValueError("La tolérance doit être comprise entre 0 et 1")
    if args.minimum_plateau_bins < 2 or args.minimum_plateau_bins >= len(edges) - 1:
        raise ValueError("Nombre minimal de tranches du plateau incompatible")
    if not 0 < args.distance_min_mpc < args.distance_max_mpc:
        raise ValueError("Distances invalides")
    validate_directory(paths["opsim"], "Dossier OpSim")
    validate_m5_table(paths["m5"], args.m5_scenario)


def print_configuration(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    bins = [
        f"[{number_text(low)},{number_text(high)}{'[' if high < 90 else ']'}"
        for low, high in zip(args.bin_edges[:-1], args.bin_edges[1:])
    ]
    print("\nExpérience par tranches disjointes")
    print("  Tranches         : " + ", ".join(bins))
    print(f"  Population       : {args.n_samples:,} KNe appariées par tranche")
    print(f"  Référence fixe   : {bins[0]}")
    print(f"  Catalogue partagé: {paths['catalog']}")
    print(f"  Seed intrinsèque : {args.intrinsic_seed}")
    print(f"  Distances        : {args.distance_min_mpc:g}–{args.distance_max_mpc:g} Mpc")
    print(f"  Couleur/fenêtre  : g-r, {args.t_min:g}–{args.t_max:g} j")
    print(f"  m5               : {args.m5_scenario}, seuil global commun")
    print(f"  Bootstrap        : {args.bootstrap_repetitions:,}, intervalle {args.confidence_level:.0%}")
    print(f"  Sortie globale   : {paths['output']}")


def generation_configuration(spec, paths, pairing_mode, args):
    from generate_kne_lightcurves import default_runtime_configuration
    runtime = default_runtime_configuration()
    runtime.synthetic_schema = "compact-extincted"
    runtime.run_name = spec.run_name
    runtime.runs_root = paths["runs"]
    runtime.filter_system = "lsst"
    runtime.parameter_source = "broad"
    runtime.opsim_dir = paths["opsim"]
    runtime.galactic_latitude_mode = "bin"
    runtime.galactic_latitude_min_deg = spec.lower
    runtime.galactic_latitude_max_deg = spec.upper
    runtime.output_mode = "synthetic-mjd"
    runtime.synthetic_cadence_days = args.synthetic_cadence_days
    runtime.synthetic_depth_cut = "none"
    runtime.storage_format = "parquet"
    runtime.parquet_events_per_shard = 500
    runtime.parquet_compression = "zstd"
    runtime.csv_sample_size = 0
    runtime.plot_mode = "none"
    runtime.n_samples = args.n_samples
    runtime.n_realizations = 1
    runtime.pairing_mode = pairing_mode
    runtime.intrinsic_catalog = paths["catalog"]
    runtime.seed = args.intrinsic_seed
    runtime.sky_seed = spec.sky_seed
    runtime.distance_mode = "comoving-volume"
    runtime.distance_min_mpc = args.distance_min_mpc
    runtime.distance_max_mpc = args.distance_max_mpc
    runtime.numerical_floor_absolute_mag = 0.0
    runtime.numerical_floor_margin_mag = 1.0
    return runtime


def population_configuration(spec, m5_table, args):
    from analyze_kne_latitude import population_default_configuration
    config = population_default_configuration()
    config.run_dir = spec.run_dir
    config.output_dir = spec.analysis_dir
    config.color_pair = "g-r"
    config.photometric_system = "lsst"
    config.t_min = args.t_min
    config.t_max = args.t_max
    config.bin_width = args.bin_width
    config.m5_quantile_table = m5_table
    config.m5_scenario = args.m5_scenario
    config.threshold_scope = "common_global"
    return config


def comparison_configuration(specs, output, args):
    from analyze_kne_latitude import comparison_default_configuration
    config = comparison_default_configuration()
    config.bins = [
        [number_text(spec.lower), number_text(spec.upper), str(spec.event_table)]
        for spec in specs
    ]
    config.output_dir = output
    config.bootstrap_repetitions = args.bootstrap_repetitions
    config.confidence_level = args.confidence_level
    config.plateau_relative_tolerance = args.plateau_relative_tolerance
    config.minimum_plateau_bins = args.minimum_plateau_bins
    config.seed = args.comparison_seed
    config.overwrite = args.overwrite_final
    return config

# =============================================================================
# WORKFLOW ORCHESTRATION
# =============================================================================

def run_experiment(args: argparse.Namespace) -> tuple[Path, Path]:
    args, paths = resolve_paths(args)
    validate_configuration(args, paths)
    print_configuration(args, paths)
    analysis_name = (
        f"latitude_bin_event_availability_g_r_{args.m5_scenario}_"
        f"{args.t_min:g}to{args.t_max:g}d"
    ).replace(".", "p")
    specs = make_specs(
        args.bin_edges, paths["runs"], args.intrinsic_seed, analysis_name
    )
    plot = paths["output"] / "kne_latitude_bin_relative_losses.png"
    table = paths["output"] / "kne_latitude_bin_relative_losses_values_table.png"

    if args.dry_run:
        print("\nPlan uniquement — aucun calcul ni fichier n'est créé")
        for index, spec in enumerate(specs, 1):
            print(f"  {index:02d}. génération/reprise {spec.label}: {spec.run_dir}")
            print(f"      analyse événementielle: {spec.event_table}")
        print(f"  comparaison finale: {plot}")
        return plot, table

    from generate_kne_lightcurves import run_generation
    from analyze_kne_latitude import run_bin_comparison, run_population_analysis

    paths["runs"].mkdir(parents=True, exist_ok=True)
    catalog_exists = paths["catalog"].is_file()
    catalog_hash = (
        validate_catalog(paths["catalog"], args.n_samples, args.intrinsic_seed)
        if catalog_exists else ""
    )

    print("\nÉtape 1/3 — génération dans les tranches disjointes")
    for index, spec in enumerate(specs, 1):
        if catalog_exists and validate_run(spec, args.n_samples, catalog_hash):
            print(f"[{index:02d}/{len(specs):02d}] déjà complet : {spec.label}")
            continue
        pairing_mode = "reuse" if catalog_exists else "create"
        print(f"[{index:02d}/{len(specs):02d}] génération : {spec.label}")
        run_generation(generation_configuration(spec, paths, pairing_mode, args))
        catalog_hash = validate_catalog(paths["catalog"], args.n_samples, args.intrinsic_seed)
        catalog_exists = True
        if not validate_run(spec, args.n_samples, catalog_hash):
            raise RuntimeError(f"Run attendu non créé : {spec.run_dir}")

    print("\nÉtape 2/3 — pertes événementielles dans chaque tranche")
    for index, spec in enumerate(specs, 1):
        if validate_event_table(spec.event_table, args.n_samples):
            print(f"[{index:02d}/{len(specs):02d}] analyse déjà complète : {spec.label}")
            continue
        if spec.analysis_dir.exists() and any(spec.analysis_dir.iterdir()):
            raise RuntimeError(
                f"Analyse partielle : {spec.analysis_dir}\n"
                "Elle n'est pas écrasée. Vérifiez puis retirez seulement ce dossier."
            )
        print(f"[{index:02d}/{len(specs):02d}] analyse : {spec.label}")
        run_population_analysis(population_configuration(spec, paths["m5"], args))
        if not validate_event_table(spec.event_table, args.n_samples):
            raise RuntimeError(f"Table attendue non créée : {spec.event_table}")

    print("\nÉtape 3/3 — comparaison à la référence fixe")
    complete = all(path.is_file() and path.stat().st_size > 0 for path in (plot, table))
    if complete and not args.overwrite_final:
        print("Comparaison globale déjà complète; réutilisation des deux PNG.")
    else:
        if plot.exists() != table.exists() and not args.overwrite_final:
            raise RuntimeError("Sortie finale partielle; autorisez son remplacement après vérification")
        run_bin_comparison(comparison_configuration(specs, paths["output"], args))

    print("\nTerminé.")
    print(f"Plot global  : {plot}")
    print(f"Table globale: {table}")
    print(f"Catalogue    : {paths['catalog']}")
    return plot, table


def main() -> None:
    args = interactive_configuration(default_configuration())
    if not prompt_yes_no("Lancer ou reprendre cette expérience", True):
        print("Arrêt demandé.")
        return
    run_experiment(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
