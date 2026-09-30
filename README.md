# KNe Lightcurve LSST Analysis Pipeline

This repository contains the reproducible analysis path used to simulate
kilonova (KNe) light curves in Rubin/LSST observing fields, construct colour
envelopes, optionally compare them with a contaminant population (SNIa population, SNII-P, GRB Afterglow, etc...).

## Pipeline at a glance

1. `get_opsim_baseline.py` exports Rubin OpSim visits and the dust metadata for
   sampled fields. It also writes the`m5_depth_quantiles_by_band.csv` table once in the OpSim export directory.
2. `generate_kne_lightcurves.py` generates KNe populations with FIESTA's
   `Bu2026_MLP` surrogate and records the absolute OpSim source in
   `run_info.txt`.
3. `build_color_envelopes.py` constructs the four KNe colour-envelope stages
   and can optionally project an SN Ia sample onto KNe sight lines.
4. `run_latitude_experiment.py` generates paired populations in disjoint
   Galactic-latitude bins, runs `analyze_kne_latitude_losses.py`, and calls
   `compare_kne_latitude_bins.py` for the final relative-loss plot and table.

Shared code lives in `kne_pipeline_common.py` and
`dynamic_color_cut_common.py`. Small inspection utilities live in `tools/`.

## Installation

The reference environment used Python 3.12.12 and FIESTA 0.2.0. For the most
faithful reconstruction:

```bash
conda env create -f environment.yml
conda activate kne_lsst_analysis
python -m unittest discover -s tests -v
```

The shorter pip route is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

The FIESTA installation must expose the `Bu2026_MLP` model used by the
generator. Model assets are not committed here. Verify their availability in
the FIESTA installation before a production run; if they were installed
manually in the original environment, reproduce that installation separately
and record its source and checksum.

## End-to-end use

### 1. Export OpSim fields

Interactive:

```bash
python get_opsim_baseline.py
```

Non-interactive example:

```bash
python get_opsim_baseline.py \
  --selection-mode ALL_SURVEY \
  --n-fields 10000 \
  --sampling-mode UNIFORM \
  --seed 12345 \
  --m5-latitude-threshold-deg 20 \
  --outdir opsim_exports_all_survey
```

The same directory then contains visit tables, `opsim_fields_index.csv`, dust
metadata, and `m5_depth_quantiles_by_band.csv`. The m5 table is a property of
the OpSim sample and does not need to be copied into every KNe run.

### 2. Generate a KNe population

```bash
python generate_kne_lightcurves.py
```

For large populations, prefer sharded Parquet storage. The script records
physical, distance, sky, extinction, seed, and storage choices in the run.
See `python generate_kne_lightcurves.py --help` for non-interactive options.

### 3. Build colour envelopes

```bash
python build_color_envelopes.py --interactive
```

For each colour, the builder separates four stages:

| Stage | m5 cut | MW extinction in magnitudes |
|---|---:|---:|
| `intrinsic_no_m5_no_mw` | no | no |
| `no_m5_with_mw` | no | yes |
| `m5_no_mw` | yes | no |
| `m5_with_mw` | yes | yes |

The m5 table is discovered from the OpSim path stored in the run's
`run_info.txt`. `--m5-quantile-table` remains available as an explicit
override.

An SN Ia Parquet input is optional. When provided, its events can be assigned
deterministically to KNe sight lines so that the foreground extinction and m5
selection are compared on the same sky distribution. The expected input
schema and scientific limitations are documented in `docs/snia_input.md`.

### 4. Run the Galactic-latitude experiment

```bash
python run_latitude_experiment.py
```

The orchestrator creates or reuses one intrinsic catalogue, projects the same
event IDs into disjoint `|b|` bins with independent sky seeds, analyzes every
bin, and produces one relative-loss plot plus one PNG value table. Use
`--dry-run` to inspect commands and output paths before a long run.

The primary ratio for bin `k` is

```text
loss rate in bin k / loss rate in the [0°, 5°[ reference bin.
```

The bootstrap remains paired by `event_id`, and the reported plateau is the
first bin from which all later ratios are compatible with a common level under
the configured relative tolerance. See `docs/methodology.md` before assigning
a physical interpretation to that operational threshold.

## Data and repository policy

The following are deliberately excluded from Git:

- `runs/`, OpSim export directories, generated figures and tables;
- SN Ia Parquet populations and other large data products;
- local FIESTA model weights;
- credentials, private keys, machine-specific paths, and environment prefixes.

Never commit a file such as `ssh_key.txt`. Use GitHub's SSH-agent or credential
manager outside the repository.

## Provenance and citation

`generate_kne_lightcurves.py` was developed from the MIT-licensed FIESTA
example `examples/generate_lightcurves.py`; the immutable upstream reference,
license notice, and requested scientific citation are recorded in
`THIRD_PARTY_NOTICES.md` and `CITATION.cff`.

Before publishing, replace the author and repository placeholders in
`CITATION.cff`, `pyproject.toml`, and `LICENSE`, then archive a release
with a DOI if the analysis is intended to be cited.
