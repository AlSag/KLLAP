# KNe–LSST analysis pipeline

This repository contains the reproducible analysis path used to simulate
kilonova (KNe) light curves in Rubin/LSST observing fields, construct colour
envelopes, optionally compare them with an external SN Ia population, and
measure relative losses as a function of Galactic latitude.

## Pipeline at a glance

1. `get_opsim_baseline.py` exports Rubin OpSim visits and the dust metadata for
   sampled fields. It also writes the canonical
   `m5_depth_quantiles_by_band.csv` table once in the OpSim export directory.
2. `generate_kne_lightcurves.py` generates KNe populations with FIESTA and
   records the selected surrogate, passbands, and absolute OpSim source in
   `run_info.txt`. Paired LSST+PS1 and LSST+ZTF products share the same event
   parameters and sky positions.
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

For `Bu2019_MLP`, use the environment in which you installed FIESTA 0.3 and
downloaded that surrogate, for example `conda activate fiesta_bu2019`. The
environment is needed only while generating light curves. Once the Parquet or
CSV products exist, `build_color_envelopes.py` does not reload FIESTA and can
run in the normal analysis environment.

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

#### Paired LSST--ZTF envelopes

Generate both filter systems in the same run so that the physical population,
distance, merger epoch, OpSim sight line, and event IDs are paired:

```bash
python generate_kne_lightcurves.py \
  --filter-system lsst+ztf \
  --output-mode synthetic-mjd \
  --synthetic-depth-cut none \
  --storage-format parquet
```

The recommended mode is to provide explicit, documented 5-sigma limiting
magnitudes. For example, for a `g-r` comparison:

```bash
python build_color_envelopes.py \
  --run-dir runs/YOUR_LSST_ZTF_RUN \
  --photometric-system lsst+ztf \
  --color-pair g-r \
  --ztf-depth-source manual \
  --ztf-m5-g 20.8 \
  --ztf-m5-r 20.6 \
  --m5-scenario median_p50 \
  --t-min 0.4 --t-max 16.0 --bin-width 0.4
```

In combined mode, `--m5-scenario` selects the Rubin/LSST depth scenario; the
ZTF values supplied with `--ztf-m5-*` are used directly. Only the bands in the
requested colour are required. A calibrated CSV with columns `band`,
`scenario`, and `m5` can instead be supplied with:

```bash
--ztf-depth-source table \
--ztf-depth-quantile-table ztf_depth_quantiles.csv
```

The exact thresholds and their provenance are recorded in
`ztf/ztf_depth_thresholds_used.csv`.

For backward compatibility only, the builder can still read the historical
NMMA joblib distributions from a local, unversioned directory:

```text
ztf_depth_distributions/
├── lims_public_g.joblib
├── lims_public_r.joblib
└── lims_i.joblib
```

Build the paired `g-r` envelopes with:

```bash
python build_color_envelopes.py \
  --run-dir runs/YOUR_LSST_ZTF_RUN \
  --photometric-system lsst+ztf \
  --color-pair g-r \
  --ztf-depth-source joblib \
  --ztf-depth-dir ztf_depth_distributions \
  --m5-scenario median_p50 \
  --t-min 0.4 --t-max 16.0 --bin-width 0.4
```

The legacy `lims` files contain smooth empirical distributions of the ZTF 5-sigma
limiting magnitude for individual images. A larger limiting magnitude means a
deeper exposure. The builder draws reproducibly from each distribution and
uses its p25, p50, or p75 as the global band threshold. It writes the exact
values, source checksums, seed, and number of draws to the recorded threshold
table.

This is a controlled depth comparison, not a full ZTF survey simulation: it
does not reproduce visit cadence, weather correlations, or per-visit noise.
The root output directory contains direct LSST--ZTF overlays for all four
stages, named `LSST_ZTF_*_comparison.png`.

The joblib format is pickle-based and can execute code while loading. Only use
files from a trusted source. The files are external inputs and are deliberately
not committed to this repository; see `THIRD_PARTY_NOTICES.md`.

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
