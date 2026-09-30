#!/usr/bin/env python3
"""Shared helpers for dynamic KNe colour-envelope scripts.

The executable scripts in the same directory import this module.  It contains
only I/O, interpolation, colour matching, rule evaluation and bootstrap
helpers; no analysis is launched when it is imported.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


BANDS = tuple("ugrizy")


def prompt_text(question: str, default: str) -> str:
    raw = input(f"{question} [{default}]: ").strip()
    return raw or default


def prompt_choice(question: str, choices: list[tuple[str, str]], default: str) -> str:
    print(f"\n{question}")
    for index, (key, label) in enumerate(choices, start=1):
        marker = " [default]" if key == default else ""
        print(f"  {index}. {label}{marker}")
    raw = input("Choice: ").strip()
    if not raw:
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1][0]
    valid = {key for key, _ in choices}
    if raw in valid:
        return raw
    raise ValueError(f"Unknown choice {raw!r}")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    marker = "Y/n" if default else "y/N"
    raw = input(f"{question} [{marker}]: ").strip().lower()
    if not raw:
        return default
    if raw in {"y", "yes", "o", "oui", "1", "true"}:
        return True
    if raw in {"n", "no", "non", "0", "false"}:
        return False
    raise ValueError(f"Expected yes or no, got {raw!r}")


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(bool)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0) != 0).to_numpy(bool)
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "t"}
    ).to_numpy(bool)


def canonical_event_id(value: object) -> str:
    text = str(value).strip()
    return str(int(text)) if re.fullmatch(r"[+-]?\d+", text) else text


def event_id_from_path(path: Path) -> str:
    match = re.search(r"_(\d+)\.csv$", path.name)
    return canonical_event_id(match.group(1) if match else path.stem)


def event_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def discover_synthetic_files(root: Path) -> list[Path]:
    directories = [root]
    if (root / "csv").is_dir():
        directories.append(root / "csv")
    if root.name == "csv":
        directories.append(root.parent / "csv")
    found: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob("lightcurve_*.csv"):
            if "lsstlike" in path.name.lower():
                continue
            found[event_id_from_path(path)] = path.resolve()
    return [found[key] for key in sorted(found, key=event_sort_key)]


def discover_lsstlike_files(root: Path) -> dict[str, Path]:
    directories = [root]
    if (root / "csv").is_dir():
        directories.append(root / "csv")
    if root.name == "csv":
        directories.append(root.parent / "csv")
    found: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob("lightcurve_LSSTlike_*.csv"):
            found[event_id_from_path(path)] = path.resolve()
    return found


def run_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    return root.parent if root.name == "csv" else root


def load_summary(root: Path) -> pd.DataFrame:
    path = run_root(root) / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    id_column = next((c for c in ("event_id", "objectid", "i") if c in table), None)
    if id_column is None:
        return pd.DataFrame()
    table = table.copy()
    table["event_id"] = table[id_column].map(canonical_event_id)
    return table


def run_info_value(root: Path, key: str) -> str | None:
    path = run_root(root) / "run_info.txt"
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        found, value = line.split(":", 1)
        if found.strip() == key:
            return value.strip().strip("'\"")
    return None


def load_m5_medians(root: Path, explicit: Path | None = None) -> tuple[dict[str, float], str]:
    base = run_root(root)
    candidates: list[Path] = [explicit.expanduser().resolve()] if explicit else []
    candidates.extend([
        base / "opsim_m5_medians_by_band.csv",
        base / "products" / "opsim_m5_medians_by_band.csv",
        base.parent / "opsim_m5_medians_by_band.csv",
        Path.cwd() / "opsim_m5_medians_by_band.csv",
    ])
    run_info = base / "run_info.txt"
    if run_info.exists():
        for line in run_info.read_text(errors="replace").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            if key.strip() not in {"synthetic_m5_medians_source", "directory"}:
                continue
            recorded = Path(raw.strip().strip("'\"")).expanduser()
            candidates.append(recorded if recorded.name.endswith(".csv") else recorded / "opsim_m5_medians_by_band.csv")
    for path in sorted(set(p.resolve() for p in candidates)):
        if not path.exists() or not path.is_file():
            continue
        try:
            table = pd.read_csv(path)
        except Exception:
            continue
        if not {"band", "m5_median"}.issubset(table.columns):
            continue
        values = pd.to_numeric(table["m5_median"], errors="coerce")
        result = dict(zip(table["band"].astype(str).str.lower(), values))
        if result:
            return {k: float(v) for k, v in result.items() if np.isfinite(v)}, str(path)
    return {}, "not_found"


def read_long_lightcurve(path: Path, prefer_observed: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"t_days", "band"}.issubset(frame.columns):
        raise ValueError(f"{path} needs t_days and band columns")
    priority = ("mag_obs", "mag", "mag_true") if prefer_observed else ("mag", "mag_true", "mag_obs")
    magnitude_column = next((c for c in priority if c in frame.columns), None)
    if magnitude_column is None:
        raise ValueError(f"{path} needs mag, mag_true or mag_obs")
    work = pd.DataFrame({
        "source_row": np.arange(len(frame), dtype=int),
        "t_days": pd.to_numeric(frame["t_days"], errors="coerce"),
        "band": frame["band"].astype(str).str.strip().str.lower(),
        "magnitude": pd.to_numeric(frame[magnitude_column], errors="coerce"),
    })
    work["magnitude_error"] = (
        pd.to_numeric(frame["mag_err"], errors="coerce")
        if "mag_err" in frame.columns else np.nan
    )
    if "detected" in frame.columns:
        work["detected"] = parse_bool(frame["detected"])
    else:
        work["detected"] = True
    return work.loc[
        np.isfinite(work["t_days"])
        & np.isfinite(work["magnitude"])
        & work["band"].isin(BANDS)
    ].sort_values("t_days").reset_index(drop=True)


def band_curve(frame: pd.DataFrame, band: str) -> tuple[np.ndarray, np.ndarray]:
    sub = frame.loc[frame["band"] == band, ["t_days", "magnitude"]].copy()
    if sub.empty:
        return np.array([], float), np.array([], float)
    sub = sub.groupby("t_days", as_index=False)["magnitude"].median().sort_values("t_days")
    return sub["t_days"].to_numpy(float), sub["magnitude"].to_numpy(float)


def interpolate_band(frame: pd.DataFrame, band: str, times: np.ndarray) -> np.ndarray:
    x, y = band_curve(frame, band)
    positive = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[positive], y[positive]
    output = np.full(len(times), np.nan, dtype=float)
    if len(x) == 0:
        return output
    if len(x) == 1:
        output[np.isclose(times, x[0], atol=1e-10, rtol=0)] = y[0]
        return output
    inside = (times >= x[0]) & (times <= x[-1]) & (times > 0)
    if np.any(inside):
        output[inside] = np.interp(np.log10(times[inside]), np.log10(x), y)
    return output


def assign_event_splits(
    event_ids: Sequence[str], train_fraction: float, validation_fraction: float, seed: int
) -> pd.DataFrame:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction must be >=0 and train+validation < 1")
    ids = np.asarray([str(value) for value in event_ids], dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = ids[rng.permutation(len(ids))]
    n_train = int(np.floor(train_fraction * len(ids)))
    n_validation = int(np.floor(validation_fraction * len(ids)))
    mapping = {value: "test" for value in ids}
    mapping.update({value: "train" for value in shuffled[:n_train]})
    mapping.update({value: "validation" for value in shuffled[n_train:n_train+n_validation]})
    return pd.DataFrame({
        "event_id": ids,
        "split": [mapping[value] for value in ids],
        "split_seed": int(seed),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(1 - train_fraction - validation_fraction),
    })


def bootstrap_percentiles_batched(
    values: np.ndarray,
    percentiles: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ordinary event bootstrap with bounded memory.

    Returns the sample estimates, bootstrap P16/P84 and their half-width.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q = np.asarray(percentiles, dtype=float)
    if len(values) == 0:
        empty = np.full(len(q), np.nan)
        return empty, empty, empty, empty
    estimate = np.percentile(values, q)
    if n_boot <= 0 or len(values) < 2:
        empty = np.full(len(q), np.nan)
        return estimate, empty, empty, empty
    boot = np.empty((n_boot, len(q)), dtype=np.float32)
    cursor = 0
    while cursor < n_boot:
        count = min(batch_size, n_boot - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draws = values[indices]
        boot[cursor:cursor+count] = np.percentile(draws, q, axis=1).T
        cursor += count
    low, high = np.percentile(boot, [16, 84], axis=0)
    return estimate, low, high, 0.5 * (high - low)


def percentile_label(value: float) -> str:
    return "p" + f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def match_color_points(
    measurements: pd.DataFrame,
    band1: str,
    band2: str,
    max_separation_days: float,
) -> pd.DataFrame:
    columns = [
        "source_row_band1", "source_row_band2", "t_band1_days", "t_band2_days",
        "time_separation_days", "t_color_mean_days", "t_color_available_days",
        "mag_band1", "mag_band2", "color", "color_error",
    ]
    first = measurements.loc[measurements["band"] == band1].copy()
    second = measurements.loc[measurements["band"] == band2].copy()
    if first.empty or second.empty:
        return pd.DataFrame(columns=columns)
    candidates: list[tuple[float, int, int]] = []
    if max_separation_days <= 1e-12:
        right_by_time: dict[float, list[int]] = {}
        for idx, time in zip(second.index, second["t_days"]):
            right_by_time.setdefault(round(float(time), 10), []).append(int(idx))
        for idx, time in zip(first.index, first["t_days"]):
            for idx2 in right_by_time.get(round(float(time), 10), []):
                candidates.append((0.0, int(idx), idx2))
    else:
        t2 = second["t_days"].to_numpy(float)
        i2 = second.index.to_numpy(int)
        order = np.argsort(t2)
        t2, i2 = t2[order], i2[order]
        for idx1, time1 in zip(first.index, first["t_days"]):
            lo = np.searchsorted(t2, float(time1) - max_separation_days, side="left")
            hi = np.searchsorted(t2, float(time1) + max_separation_days, side="right")
            for pos in range(lo, hi):
                candidates.append((abs(float(time1)-t2[pos]), int(idx1), int(i2[pos])))
    candidates.sort(key=lambda item: (item[0], max(
        float(first.loc[item[1], "t_days"]), float(second.loc[item[2], "t_days"])
    )))
    used1: set[int] = set()
    used2: set[int] = set()
    rows = []
    for separation, idx1, idx2 in candidates:
        if idx1 in used1 or idx2 in used2:
            continue
        used1.add(idx1); used2.add(idx2)
        row1, row2 = first.loc[idx1], second.loc[idx2]
        e1, e2 = float(row1["magnitude_error"]), float(row2["magnitude_error"])
        error = float(np.hypot(e1, e2)) if np.isfinite(e1) and np.isfinite(e2) else np.nan
        t1, t2v = float(row1["t_days"]), float(row2["t_days"])
        rows.append({
            "source_row_band1": int(row1["source_row"]),
            "source_row_band2": int(row2["source_row"]),
            "t_band1_days": t1,
            "t_band2_days": t2v,
            "time_separation_days": float(separation),
            "t_color_mean_days": 0.5 * (t1+t2v),
            "t_color_available_days": max(t1, t2v),
            "mag_band1": float(row1["magnitude"]),
            "mag_band2": float(row2["magnitude"]),
            "color": float(row1["magnitude"]-row2["magnitude"]),
            "color_error": error,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        "t_color_available_days"
    ).reset_index(drop=True)


def evaluate_against_rules(colors: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    result = colors.copy()
    for name, default in (
        ("phase_days", np.nan), ("evaluated", False), ("passes_cut", False),
        ("rule_id", ""), ("color_min", np.nan), ("color_max", np.nan),
    ):
        result[name] = default
    if result.empty:
        return result
    result["phase_days"] = result["t_color_available_days"]
    for rule in rules.itertuples():
        mask = (
            (result["phase_days"] > float(rule.t_min_days))
            & (result["phase_days"] <= float(rule.t_max_days))
        )
        result.loc[mask, "evaluated"] = True
        result.loc[mask, "rule_id"] = str(rule.rule_id)
        result.loc[mask, "color_min"] = float(rule.color_min)
        result.loc[mask, "color_max"] = float(rule.color_max)
        result.loc[mask, "passes_cut"] = (
            (result.loc[mask, "color"] >= float(rule.color_min))
            & (result.loc[mask, "color"] <= float(rule.color_max))
        )
    return result


def criterion_results(colors: pd.DataFrame) -> dict[str, dict[str, object]]:
    evaluated = colors.loc[colors.get("evaluated", False)].sort_values("phase_days")
    result: dict[str, dict[str, object]] = {}
    if evaluated.empty:
        for criterion in ("A_first_color", "F1_two_distinct_bins", "ever_pass"):
            result[criterion] = {"evaluable": False, "passed": False, "selection_time_days": np.nan}
        return result

    first = evaluated.iloc[0]
    result["A_first_color"] = {
        "evaluable": True,
        "passed": bool(first["passes_cut"]),
        "selection_time_days": float(first["phase_days"]) if bool(first["passes_cut"]) else np.nan,
    }
    passing = evaluated.loc[evaluated["passes_cut"]]
    result["ever_pass"] = {
        "evaluable": True,
        "passed": not passing.empty,
        "selection_time_days": float(passing["phase_days"].min()) if len(passing) else np.nan,
    }
    distinct_available = int(evaluated["rule_id"].replace("", np.nan).nunique())
    distinct_passing = passing.drop_duplicates("rule_id", keep="first").sort_values("phase_days")
    result["F1_two_distinct_bins"] = {
        "evaluable": distinct_available >= 2,
        "passed": len(distinct_passing) >= 2,
        "selection_time_days": float(distinct_passing.iloc[1]["phase_days"]) if len(distinct_passing) >= 2 else np.nan,
    }
    return result


def binomial_bootstrap_interval(
    successes: int, denominator: int, n_boot: int, rng: np.random.Generator
) -> dict[str, float]:
    if denominator <= 0:
        return {key: np.nan for key in ("estimate", "p2p5", "p16", "p84", "p97p5")}
    fraction = successes / denominator
    draws = rng.binomial(denominator, fraction, size=n_boot) / denominator
    p2, p16, p84, p97 = np.percentile(draws, [2.5, 16, 84, 97.5])
    return {"estimate": fraction, "p2p5": p2, "p16": p16, "p84": p84, "p97p5": p97}


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
