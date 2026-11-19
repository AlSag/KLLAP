# KNe–LSST analysis pipeline

This repository simulates kilonova (KNe) populations, projects them through
Rubin/LSST observing conditions, constructs time-dependent colour envelopes,
and studies statistical convergence, parameter-space structure, and Galactic-
latitude selection effects. It supports FIESTA `Bu2026_MLP`, `Bu2019_MLP`,
`MetzgerModel`, and `MetzgerFullModel` light-curve backends.

Generated runs, OpSim exports, figures, model weights, and large data products
are intentionally excluded from Git. The repository contains the source,
documentation, tests, and metadata required to review the pipeline.

## Public workflows

Ten root-level scripts create scientific products. Every one is
started with a short command and then configured through terminal questions;
no long option list is required. The BOOM utilities also retain optional arguments for automation.

| Step | Script | Main products |
|---:|---|---|
| 1 | `get_opsim_baseline.py` | sampled OpSim fields and visits, dust values, m5 tables and diagnostics |
| 2 | `generate_kne_lightcurves.py` | physical KNe populations, synthetic or LSST-like light curves, run metadata |
| 3 | `build_color_envelopes.py` | colour populations, percentile envelopes and stage comparisons |
| 4 | `analyze_envelope_population_convergence_fast.py` | percentile-value and temporal-extent convergence tables and figures |
| 5 | `plot_colour_envelope_realizations.py` | empirical percentile comparisons across independent realizations |
| 6 | `plot_kne_colour_isocontours.py` | population-count maps and median iso-colour contours in physical-parameter space |
| 7 | `analyze_kne_latitude.py` | population losses, paired-bin comparisons, or low/high plateau diagnostics |
| 8 | `run_latitude_experiment.py` | complete paired latitude-bin generation and analysis workflow |
| 9 | `snia_parquet_to_simu_input.py` | SN Ia Parquet to BOOM CSVs, preserving supplied photometric errors |
| 10 | `compute_color_filter_efficiency.py` | efficiency audit for the exact input-index range processed by BOOM |

Run a workflow with, for example:

```bash
python generate_kne_lightcurves.py
```

The same form applies to each script in the table. Defaults are displayed in
brackets and pressing Enter accepts them.

The public scripts are intentionally reviewable as individual files. Each
contains labelled blocks for its terminal interface, scientific calculations,
and figures. Five smaller modules remain because their definitions are truly
shared by several workflows:

- `kne_pipeline_common.py`: stable run formats, OpSim quantities, and posterior loading;
- `kne_population_io.py`: shared Parquet/CSV discovery and metadata loading;
- `kne_m5.py`: canonical m5 scenarios and depth-table discovery;
- `kne_envelope_core.py`: shared depth selections and empirical percentiles;
- `dynamic_color_cut_common.py`: shared colour matching and rule evaluation.

These modules do not create products when executed. The detailed ownership map
is in [`docs/code_map.md`](docs/code_map.md), and the keep/remove decision is in
[`docs/script_inventory.md`](docs/script_inventory.md).

## BOOM exports

Launch `python generate_kne_lightcurves.py`, select LSST and synthetic MJD
products, then enable the additional BOOM export. Choose **complete** to retain
all finite model magnitudes, including those fainter than m5. Choose
**m5-detected** for a depth-limited benchmark. Both use Ivezić magnitude
uncertainties, without added noise. The saved scientific population retains
faint magnitudes for subsequent upper-limit envelope analysis.

`python snia_parquet_to_simu_input.py` instead reads the SN Ia `mag` and
`magerr` columns. It applies the band m5 cut and keeps fainter rows as encoded
limits (`mag` empty, `mag_err=m5`), following the supplied converter's convention.
It does not compute Ivezić errors. Check that the target VM accepts this limit
encoding; the complete KNe export contains only finite magnitude/error pairs.

`python compute_color_filter_efficiency.py` asks for the BOOM input folder,
`stats.csv` (or its parent folder), and inclusive first/last filename indices.
Use the exact same population and selected IDs as the VM run. It reports both
passed/evaluable and passed/processed, with a per-event audit. Valid photometry
is an input-availability criterion, not a claim that the colour is calculable.

## Installation

The reference environment uses Python 3.12.12, FIESTA 0.2.0,
`rubin_sim` 2.6.1, and `rubin_scheduler` 3.22.0:

```bash
conda env create -f environment.yml
conda activate kne_lsst_analysis
python -m unittest discover -s tests -v
```

A pip-based environment can instead be created from `requirements-lock.txt`.
FIESTA surrogate weights are not stored here. Record the model source,
immutable revision, and checksums for every production run.

## Scientific workflow

### 1. Prepare OpSim products

```bash
python get_opsim_baseline.py
```

The script samples positions and visits, associates dust information, and
writes the canonical `m5_depth_quantiles_by_band.csv`. Depth statistics belong
to the OpSim sample and are not recomputed by the light-curve generator.

### 2. Generate KNe light curves

```bash
python generate_kne_lightcurves.py
```

The terminal interface selects the model, parameter population, filter system,
scientific products, storage format, distance distribution, Galactic-latitude
selection, sample size, and random seeds. Large populations should use sharded
Parquet output. Every run records its physical choices and reproducibility
metadata.

For controlled sky or latitude comparisons, create a reusable intrinsic
catalogue in the first run and reuse it in later runs. This preserves the
physical parameters event by event.

### 3. Build colour envelopes

```bash
python build_color_envelopes.py
```

The generator and envelope builder no longer add or undo Galactic extinction.
Magnitudes are used as saved. New populations have `synthetic_schema=model`.
An old `compact-extincted` population still contains its original attenuation;
regenerate it to obtain a population without extinction.

| Stage | Depth selection |
|---|---|
| `no_m5` | Every finite supported colour |
| `m5` | Both bands meet the selected limiting depths |

Every selected event participates in estimation: there is no training split.
Choose the fixed interval in the terminal, for example p1–p99 or p16–p84.
There is no resampling uncertainty calculation or minimum count of 20/30 events.
A single finite value defines an empirical quantile; an empty bin stays undefined.
Counts are exported beside the percentiles.

For LSST or ZTF, the `m5` stage can additionally plot the contribution from
upper limits in the first colour band: `g_lim-r`, `r_lim-i`, `i_lim-z`, etc.
Hatching shows only the part extending beyond the detection-only interval;
it has no median curve. These are percentiles of colour-limit values, with
identification intervals saved separately. Two band limits do not identify a
colour, and an unsupported model point is not a non-detection.

The terminal interface can retain time since merger or align every KNe and
SN Ia to a peak in either band of the requested colour. Peak-relative
interpolation is linear in phase (including negative phase) and never
extrapolates. For KNe, a significant interior second peak has priority; if it
does not exist and the first saved point is the brightest, that boundary point
is used as the temporal reference. SN Ia peaks remain strict interior peaks.
Unresolved events and boundary fallbacks are identified in dedicated
peak-diagnostic CSV files. Merger-relative interpolation retains the historical
logarithmic-positive-time convention. The saved configuration records the
temporal reference, peak rule, selected percentiles, limiting depths, and
population membership.

### 4. Measure convergence

```bash
python analyze_envelope_population_convergence_fast.py
```

The parent population is loaded once. Smaller populations are reproducible
nested subsets, the largest selected size is the reference, and independent
repetitions test sensitivity to the sampled events. The analysis keeps two
questions separate:

1. Are percentile values stable within the selected magnitude tolerance?
2. Does the nonempty envelope reach the reference temporal
   extent within the selected time tolerance?

The default families are p5–p95, p16–p84, and p25–p75. See
[`MULTI_PERCENTILE_CONVERGENCE.md`](MULTI_PERCENTILE_CONVERGENCE.md).

### 5. Compare independent realizations

```bash
python plot_colour_envelope_realizations.py
```

Each realization is an independent physical draw. Nested prefixes inside it
isolate the effect of population size. The figure reports empirical p16, p50
and p84 for every nonempty bin. Repeated draws and subsets are population
experiments, without replacement within each subset.

### 6. Map iso-colours

```bash
python plot_kne_colour_isocontours.py
```

The script reads saved populations, computes colours without temporal
extrapolation, and bins physical-parameter planes. The heatmap reports the
number of stage-available KNe in every non-empty cell; requested iso-colour
contours are calculated from the median colour in every nonempty
cell. Smoothing can be applied to supported cells for contour presentation
but never fills an empty cell. By default, axes show the
full validity domain of a recognized surrogate. Regions outside the sampled
population are gray: displaying the model domain does not extrapolate colours
or invent events there.

### 7. Analyse Galactic latitude

```bash
python analyze_kne_latitude.py
```

The first terminal menu selects one of three distinct products:

- losses and availability inside one population;
- paired comparison of disjoint latitude bins;
- stability of a moving low/high latitude cut.

The plateau is an operational stability criterion, not a physical Galactic
boundary. See [`docs/low_high_plateau.md`](docs/low_high_plateau.md).

### 8. Run the complete paired-latitude experiment

```bash
python run_latitude_experiment.py
```

This is workflow orchestration: it reuses one intrinsic catalogue, generates
or resumes every disjoint latitude population, runs the event-level analysis,
and finally compares all bins with the lowest-latitude bin. It calls the Python
functions from the generator and latitude analyser directly; it does not build
or execute hidden shell commands. A terminal choice provides a plan-only mode
before long calculations.

## Testing and reproducibility

Before committing a scientific change, run:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

The real-FIESTA integration test is optional because it requires local model
assets. Numerical equivalence, not line count, is the acceptance criterion for
refactoring. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Repository policy

Do not commit generated runs, OpSim databases, figures, Parquet shards,
posterior archives, FIESTA weights, external SN Ia populations, backup files,
credentials, or machine-specific paths. Publish large scientific products in
a DOI-backed archive when needed for a paper.

## Attribution, AI use, and citation

`generate_kne_lightcurves.py` was developed from FIESTA's MIT-licensed
`examples/generate_lightcurves.py` and substantially extended. Upstream
references, scientific citations, model provenance, and external-data notices
are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Generative-AI use for code development, bug checking, repository organisation,
and supporting documentation is declared in [`AI_USAGE.md`](AI_USAGE.md).
Scientific responsibility and validation remain with the repository author.

Use [`CITATION.cff`](CITATION.cff) to cite the software. Repository-owned source
is released under the [`MIT License`](LICENSE); third-party software and data
retain their respective terms.
