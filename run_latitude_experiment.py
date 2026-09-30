#!/usr/bin/env python3
"""Run the complete paired, disjoint-latitude KNe experiment.

Default bins are [0,5), [5,10), [10,15), [15,20), [20,25), [25,30),
[30,40), [40,50) and [50,90] degrees, with 2,000 KNe in each bin.  One
shared intrinsic catalogue makes event_id a physical pairing key across every
bin.  Generation, per-run availability analysis and the final fixed-reference
comparison are all resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 90.0)
M5_REQUIRED_COLUMNS = {
    "threshold_scope",
    "threshold_latitude_group",
    "band",
    "scenario",
    "m5",
}
EVENT_REQUIRED_COLUMNS = {
    "event_id",
    "abs_b_deg",
    "N_bins_baseline",
    "N_bins_m5_no_mw",
    "N_bins_m5_with_mw",
}


@dataclass(frozen=True)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired KNe in disjoint |b| bins, analyse losses and compare "
            "every bin to the fixed lowest-latitude reference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--generator", type=Path)
    parser.add_argument("--analyzer", type=Path)
    parser.add_argument("--comparator", type=Path)
    parser.add_argument("--opsim-dir", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--intrinsic-catalog", type=Path)
    parser.add_argument("--m5-quantile-table", type=Path)
    parser.add_argument(
        "--bin-edges",
        type=float,
        nargs="+",
        default=list(DEFAULT_EDGES),
        help="Ordered edges of contiguous disjoint |b| bins",
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--intrinsic-seed", type=int, default=12345)
    parser.add_argument("--distance-min-mpc", type=float, default=10.0)
    parser.add_argument("--distance-max-mpc", type=float, default=600.0)
    parser.add_argument("--synthetic-cadence-days", type=float, default=0.1)
    parser.add_argument(
        "--m5-scenario",
        choices=["worst_p25", "median_p50", "best_p75"],
        default="best_p75",
    )
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--t-max", type=float, default=16.0)
    parser.add_argument("--bin-width", type=float, default=0.4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--confidence-level", type=float, choices=[0.68, 0.95], default=0.95
    )
    parser.add_argument("--plateau-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--minimum-plateau-bins", type=int, default=3)
    parser.add_argument("--comparison-seed", type=int, default=20260930)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite-final", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    return parser


def prompt_text(label: str, default: str) -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or default


def prompt_yes_no(label: str, default: bool = True) -> bool:
    answer = input(f"{label} [{'O/n' if default else 'o/N'}]: ").strip().lower()
    if not answer:
        return default
    return answer in {"o", "oui", "y", "yes", "1"}


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


def make_specs(edges: Sequence[float], runs_root: Path, seed: int) -> list[BinSpec]:
    specs = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        sky_seed = derive_bin_sky_seed(seed, lower, upper)
        run_name = f"paired_b{number_label(lower)}to{number_label(upper)}_bin"
        run_dir = runs_root / f"{run_name}_intrinsic_seed_{seed}_sky_seed_{sky_seed}"
        analysis_dir = run_dir / "analysis" / "latitude_bin_event_availability_g_r_best_p75_0p4to16d"
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


def run_command(label: str, command: Sequence[str], dry_run: bool) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def generator_command(
    generator: Path,
    spec: BinSpec,
    paths: dict[str, Path],
    pairing_mode: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(generator),
        "--run-name", spec.run_name,
        "--runs-root", str(paths["runs"]),
        "--filter-system", "lsst",
        "--parameter-source", "broad",
        "--opsim-dir", str(paths["opsim"]),
        "--galactic-latitude-mode", "bin",
        "--galactic-latitude-min-deg", number_text(spec.lower),
        "--galactic-latitude-max-deg", number_text(spec.upper),
        "--output-mode", "synthetic-mjd",
        "--synthetic-cadence-days", f"{args.synthetic_cadence_days:g}",
        "--synthetic-depth-cut", "none",
        "--storage-format", "parquet",
        "--parquet-events-per-shard", "500",
        "--parquet-compression", "zstd",
        "--csv-sample-size", "0",
        "--opsim-cache-size", "32",
        "--surrogate-batch-size", "64",
        "--plot-mode", "none",
        "--n-samples", str(args.n_samples),
        "--pairing-mode", pairing_mode,
        "--intrinsic-catalog", str(paths["catalog"]),
        "--seed", str(args.intrinsic_seed),
        "--sky-seed", str(spec.sky_seed),
        "--distance-mode", "comoving-volume",
        "--distance-min-mpc", f"{args.distance_min_mpc:g}",
        "--distance-max-mpc", f"{args.distance_max_mpc:g}",
        "--numerical-floor-absolute-mag", "0",
        "--numerical-floor-margin-mag", "1",
    ]


def analyzer_command(
    analyzer: Path,
    spec: BinSpec,
    m5_table: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(analyzer),
        "--run-dir", str(spec.run_dir),
        "--output-dir", str(spec.analysis_dir),
        "--color-pair", "g-r",
        "--photometric-system", "lsst",
        "--t-min", f"{args.t_min:g}",
        "--t-max", f"{args.t_max:g}",
        "--bin-width", f"{args.bin_width:g}",
        "--m5-quantile-table", str(m5_table),
        "--m5-scenario", args.m5_scenario,
        "--threshold-scope", "common_global",
        "--latitude-bin-width", "5",
        "--threshold-min", "5",
        "--threshold-max", "70",
        "--threshold-step", "1",
        "--min-events-per-side", "100",
        "--plateau-tolerance", "0.02",
        "--plateau-ratio-tolerance", "0.1",
        "--plateau-min-thresholds", "3",
    ]


def comparator_command(
    comparator: Path,
    specs: Sequence[BinSpec],
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [sys.executable, str(comparator)]
    for spec in specs:
        command.extend(
            [
                "--bin",
                number_text(spec.lower),
                number_text(spec.upper),
                str(spec.event_table),
            ]
        )
    command.extend(
        [
            "--output-dir", str(output),
            "--bootstrap-repetitions", str(args.bootstrap_repetitions),
            "--confidence-level", f"{args.confidence_level:g}",
            "--plateau-relative-tolerance", f"{args.plateau_relative_tolerance:g}",
            "--minimum-plateau-bins", str(args.minimum_plateau_bins),
            "--seed", str(args.comparison_seed),
        ]
    )
    if args.overwrite_final:
        command.append("--overwrite")
    return command


def resolve_paths(args: argparse.Namespace) -> tuple[argparse.Namespace, dict[str, Path]]:
    interactive = not args.no_interactive
    if interactive and args.project_root == Path("."):
        args.project_root = Path(prompt_text("Dossier du projet", str(Path.cwd())))
    root = args.project_root.expanduser().resolve()
    paths = {
        "root": root,
        "generator": absolute(args.generator or Path("generate_kne_lightcurves.py"), root),
        "analyzer": absolute(args.analyzer or Path("analyze_kne_latitude_losses.py"), root),
        "comparator": absolute(args.comparator or Path("compare_kne_latitude_bins.py"), root),
        "opsim": absolute(args.opsim_dir or Path("opsim_exports_all_survey"), root),
        "runs": absolute(args.runs_root or Path("runs"), root),
        "catalog": absolute(args.intrinsic_catalog or Path("paired_intrinsic_population_uniform.csv"), root),
    }
    for key, label in (
        ("generator", "Générateur avec mode bin"),
        ("analyzer", "Script d'analyse"),
        ("comparator", "Comparateur par tranches"),
    ):
        if interactive and not paths[key].is_file():
            paths[key] = absolute(Path(prompt_text(label, str(paths[key]))), root)
    if interactive and not paths["opsim"].is_dir():
        paths["opsim"] = absolute(Path(prompt_text("Dossier OpSim", str(paths["opsim"]))), root)
    m5 = absolute(args.m5_quantile_table, root) if args.m5_quantile_table else None
    if m5 is None:
        m5 = discover_m5_table(root, paths["opsim"], args.m5_scenario)
    if m5 is None and interactive:
        m5 = absolute(
            Path(prompt_text("Table m5", "m5_depth_quantiles_by_band.csv")), root
        )
    if m5 is None:
        raise FileNotFoundError("Table m5 non résolue; utilisez --m5-quantile-table")
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
        raise ValueError("--bin-edges doit être strictement croissant")
    if edges[0] < 0 or edges[-1] > 90:
        raise ValueError("Les bornes doivent rester dans [0, 90]")
    args.bin_edges = edges
    if args.n_samples < 1:
        raise ValueError("--n-samples doit être positif")
    if args.bootstrap_repetitions < 100:
        raise ValueError("Utilisez au moins 100 répétitions bootstrap")
    if not 0 < args.plateau_relative_tolerance < 1:
        raise ValueError("La tolérance doit être comprise entre 0 et 1")
    if args.minimum_plateau_bins < 2 or args.minimum_plateau_bins >= len(edges) - 1:
        raise ValueError("Nombre minimal de tranches du plateau incompatible")
    if not 0 < args.distance_min_mpc < args.distance_max_mpc:
        raise ValueError("Distances invalides")
    for key, label in (
        ("generator", "Générateur"),
        ("analyzer", "Analyseur"),
        ("comparator", "Comparateur"),
    ):
        validate_file(paths[key], label)
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
    print(
        f"  Distances        : volume comobile, "
        f"{args.distance_min_mpc:g}–{args.distance_max_mpc:g} Mpc"
    )
    print(f"  Couleur/fenêtre  : g-r, {args.t_min:g}–{args.t_max:g} j")
    print(f"  m5               : {args.m5_scenario}, common_global")
    print(
        f"  Plateau          : toutes les tranches suivantes à ±"
        f"{100*args.plateau_relative_tolerance:.0f}%, minimum "
        f"{args.minimum_plateau_bins} tranches"
    )
    print(
        f"  Bootstrap        : {args.bootstrap_repetitions:,}, "
        f"intervalle {args.confidence_level:.0%}"
    )
    print(f"  Sortie globale   : {paths['output']}")


def main() -> None:
    args = build_parser().parse_args()
    args, paths = resolve_paths(args)
    validate_configuration(args, paths)
    print_configuration(args, paths)
    if not args.yes and not args.no_interactive and not prompt_yes_no("Lancer/reprendre ?"):
        print("Arrêt demandé.")
        return

    paths["runs"].mkdir(parents=True, exist_ok=True)
    specs = make_specs(args.bin_edges, paths["runs"], args.intrinsic_seed)
    catalog_exists = paths["catalog"].is_file()
    catalog_hash = (
        validate_catalog(paths["catalog"], args.n_samples, args.intrinsic_seed)
        if catalog_exists else ""
    )

    print("\nÉtape 1/3 — génération dans les tranches disjointes")
    for index, spec in enumerate(specs, start=1):
        if catalog_exists and validate_run(spec, args.n_samples, catalog_hash):
            print(f"[{index:02d}/{len(specs):02d}] déjà complet : {spec.label}")
            continue
        mode = "reuse" if catalog_exists else "create"
        run_command(
            f"génération {index}/{len(specs)} — {spec.label}",
            generator_command(paths["generator"], spec, paths, mode, args),
            args.dry_run,
        )
        if args.dry_run:
            catalog_exists = True
            continue
        catalog_hash = validate_catalog(paths["catalog"], args.n_samples, args.intrinsic_seed)
        catalog_exists = True
        if not validate_run(spec, args.n_samples, catalog_hash):
            raise RuntimeError(f"Run attendu non créé : {spec.run_dir}")

    print("\nÉtape 2/3 — pertes événementielles dans chaque tranche")
    for index, spec in enumerate(specs, start=1):
        if validate_event_table(spec.event_table, args.n_samples):
            print(f"[{index:02d}/{len(specs):02d}] analyse déjà complète : {spec.label}")
            continue
        if spec.analysis_dir.exists() and any(spec.analysis_dir.iterdir()) and not args.dry_run:
            raise RuntimeError(
                f"Analyse partielle : {spec.analysis_dir}\n"
                "Elle n'est pas écrasée. Vérifiez puis retirez seulement ce dossier d'analyse."
            )
        run_command(
            f"analyse {index}/{len(specs)} — {spec.label}",
            analyzer_command(paths["analyzer"], spec, paths["m5"], args),
            args.dry_run,
        )
        if not args.dry_run and not validate_event_table(spec.event_table, args.n_samples):
            raise RuntimeError(f"Table attendue non créée : {spec.event_table}")

    print("\nÉtape 3/3 — comparaison à la référence fixe")
    plot = paths["output"] / "kne_latitude_bin_relative_losses.png"
    table = paths["output"] / "kne_latitude_bin_relative_losses_values_table.png"
    complete = all(path.is_file() and path.stat().st_size > 0 for path in (plot, table))
    if complete and not args.overwrite_final:
        print("Comparaison globale déjà complète; réutilisation des deux PNG.")
    else:
        if plot.exists() != table.exists() and not args.overwrite_final and not args.dry_run:
            raise RuntimeError("Sortie finale partielle; utilisez --overwrite-final après vérification")
        run_command(
            "comparaison globale par tranches",
            comparator_command(paths["comparator"], specs, paths["output"], args),
            args.dry_run,
        )

    print("\nTerminé.")
    print(f"Plot global  : {plot}")
    print(f"Table globale: {table}")
    print(f"Catalogue    : {paths['catalog']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        print(f"\nÉchec de la commande (code {exc.returncode}).", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"\nERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
