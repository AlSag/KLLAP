#!/usr/bin/env python3
"""Plot empirical ZTF limiting-magnitude distributions distributed by NMMA.

The input files are scikit-learn KernelDensity objects serialized with joblib:

* ``lims_public_g.joblib``
* ``lims_public_r.joblib``
* ``lims_i.joblib``

Only load joblib files obtained from a trusted source. Joblib relies on pickle
and loading an untrusted file can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILENAMES = {
    "g": ("lims_public_g.joblib", "lims_g.joblib"),
    "r": ("lims_public_r.joblib", "lims_r.joblib"),
    "i": ("lims_i.joblib", "lims_public_i.joblib"),
}

BAND_COLORS = {
    "g": "#2ca25f",
    "r": "#de2d26",
    "i": "#756bb1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the NMMA ZTF limiting-magnitude KDEs and plot their "
            "g-, r- and i-band distributions."
        )
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path("ztf_depth_distributions"),
        help="Directory containing the trusted ZTF joblib files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/ztf_depth_distributions"),
        help="Directory receiving the PNG and CSV outputs.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200_000,
        help="Number of deterministic draws per band (default: 200000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261001,
        help="Base random seed (default: 20261001).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins (default: 80).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use command-line values without terminal questions.",
    )
    return parser.parse_args()


def prompt(label: str, default: str) -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or default


def configure_interactively(args: argparse.Namespace) -> argparse.Namespace:
    if args.non_interactive or not sys.stdin.isatty():
        return args

    print("ZTF limiting-magnitude distributions")
    args.depth_dir = Path(
        prompt("Directory containing the ZTF joblib files", str(args.depth_dir))
    )
    args.output_dir = Path(prompt("Output directory", str(args.output_dir)))
    args.samples = int(prompt("Random draws per band", str(args.samples)))
    args.seed = int(prompt("Random seed", str(args.seed)))
    args.bins = int(prompt("Histogram bins", str(args.bins)))
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_file(directory: Path, band: str) -> Path:
    for filename in FILENAMES[band]:
        path = directory / filename
        if path.is_file():
            return path
    expected = ", ".join(FILENAMES[band])
    raise FileNotFoundError(
        f"Missing ZTF {band}-band limiting-magnitude KDE in {directory}. "
        f"Expected one of: {expected}"
    )


def load_trusted_kde(path: Path):
    """Load an older sklearn KDE with aliases needed by recent sklearn."""
    try:
        import joblib
        import sklearn.metrics._dist_metrics as distance_metrics
    except ImportError as exc:
        raise ImportError(
            "This script requires joblib and scikit-learn."
        ) from exc

    if (
        not hasattr(distance_metrics, "EuclideanDistance")
        and hasattr(distance_metrics, "EuclideanDistance64")
    ):
        distance_metrics.EuclideanDistance = distance_metrics.EuclideanDistance64

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Trying to unpickle estimator KernelDensity.*",
        )
        kde = joblib.load(path)

    if not hasattr(kde, "sample") or not hasattr(kde, "score_samples"):
        raise TypeError(f"{path} does not contain a scikit-learn KernelDensity")
    if not hasattr(kde, "bandwidth_") and hasattr(kde, "bandwidth"):
        kde.bandwidth_ = kde.bandwidth
    return kde


def sample_band(kde, samples: int, seed: int) -> np.ndarray:
    draws = np.asarray(
        kde.sample(n_samples=samples, random_state=seed), dtype=float
    ).reshape(-1)
    draws = draws[np.isfinite(draws)]
    if draws.size < 1000:
        raise ValueError("Fewer than 1000 finite limiting-magnitude draws")
    return draws


def distribution_grid(kde, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.percentile(draws, [0.1, 99.9])
    padding = max(0.15, 0.08 * (high - low))
    grid = np.linspace(low - padding, high + padding, 800)
    density = np.exp(kde.score_samples(grid[:, None]))
    return grid, density


def main() -> None:
    args = configure_interactively(parse_args())
    if args.samples < 1000:
        raise ValueError("--samples must be at least 1000")
    if args.bins < 10:
        raise ValueError("--bins must be at least 10")

    depth_dir = args.depth_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    band_data: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for band_index, band in enumerate(("g", "r", "i")):
        path = resolve_file(depth_dir, band)
        kde = load_trusted_kde(path)
        band_seed = int(args.seed + 1009 * (band_index + 1))
        draws = sample_band(kde, args.samples, band_seed)
        q01, q05, q25, q50, q75, q95, q99 = np.percentile(
            draws, [1, 5, 25, 50, 75, 95, 99]
        )
        grid, density = distribution_grid(kde, draws)

        band_data[band] = {
            "draws": draws,
            "grid": grid,
            "density": density,
            "quantiles": (q25, q50, q75),
        }
        rows.append(
            {
                "band": band,
                "p01_mag": q01,
                "p05_mag": q05,
                "p25_mag": q25,
                "p50_mag": q50,
                "p75_mag": q75,
                "p95_mag": q95,
                "p99_mag": q99,
                "mean_mag": float(np.mean(draws)),
                "std_mag": float(np.std(draws, ddof=1)),
                "sampling_seed": band_seed,
                "sampling_draws": int(draws.size),
                "source_file": str(path),
                "source_sha256": sha256(path),
            }
        )

    table = pd.DataFrame(rows)
    csv_path = output_dir / "ztf_limiting_magnitude_quantiles.csv"
    table.to_csv(csv_path, index=False, float_format="%.6f")

    fig, axes = plt.subplots(
        3, 1, figsize=(8.6, 9.2), sharex=True, constrained_layout=True
    )
    line_styles = (("p25", "--"), ("p50", "-"), ("p75", ":"))

    for ax, band in zip(axes, ("g", "r", "i")):
        data = band_data[band]
        draws = np.asarray(data["draws"])
        grid = np.asarray(data["grid"])
        density = np.asarray(data["density"])
        quantiles = tuple(data["quantiles"])
        color = BAND_COLORS[band]

        ax.hist(
            draws,
            bins=args.bins,
            density=True,
            color=color,
            alpha=0.24,
            edgecolor="none",
            label="Deterministic KDE draws",
        )
        ax.plot(grid, density, color=color, lw=2.0, label="KDE density")
        for (label, style), value in zip(line_styles, quantiles):
            ax.axvline(
                value,
                color="#252525",
                linestyle=style,
                linewidth=1.35,
                label=f"{label} = {value:.2f} mag",
            )
        ax.set_ylabel("Probability density")
        ax.set_title(f"ZTF {band} band", loc="left", fontweight="bold")
        ax.grid(alpha=0.20, linewidth=0.7)
        ax.legend(frameon=False, fontsize=8.5, ncol=2)

    axes[-1].set_xlabel(r"Limiting magnitude $m_{\mathrm{lim}}$ [AB mag]")
    png_path = output_dir / "ztf_limiting_magnitude_distributions.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\nQuantiles of the sampled ZTF limiting-magnitude distributions")
    print(
        table[["band", "p25_mag", "p50_mag", "p75_mag"]]
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nPlot: {png_path}")
    print(f"Table: {csv_path}")


if __name__ == "__main__":
    main()
