# KNe–LSST analysis pipeline

This repository simulates kilonova (KNe) populations, projects them through
Rubin/LSST observing conditions, constructs time-dependent colour envelopes,
and studies their statistical convergence and astrophysical selection effects.
The current workflow also supports paired LSST–PS1 or LSST–ZTF products,
Galactic-latitude studies, parameter-space projections, and semi-analytical
Metzger light-curve backends exposed by FIESTA.

The repository contains source code, tests, and documentation. Generated runs,
OpSim exports, figures, large tables, model weights, and external data products
are intentionally excluded from version control.

## Scientific scope

The pipeline is designed to answer questions such as:

- how many simulated KNe are needed for stable colour-percentile envelopes;
- how Rubin depth and Milky Way extinction change the available population;
- which physical parameters are represented by colour-selected KNe;
- how colour availability and losses vary with Galactic latitude;
- over which luminosity-distance range two-band colours remain measurable.

Unless explicitly stated otherwise, the depth analyses use global band
thresholds derived from an OpSim sample. They do not reproduce the full Rubin
cadence, image-by-image noise, alert production, or broker selection function.

## Workflow at a glance

| Stage | Main entry point | Purpose |
|---|---|---|
| OpSim preparation | `get_opsim_baseline.py` | Sample survey fields, visits, dust, and m5 distributions |
| Population generation | `generate_kne_lightcurves.py` | Generate physical populations and synthetic/LSST-like light curves |
| Colour envelopes | `build_color_envelopes.py` | Build percentile envelopes for four MW/m5 stages |
| Population convergence | `analyze_envelope_population_convergence_fast.py` | Measure percentile-value and temporal-extent convergence |
| Convergence replot | `replot_envelope_population_convergence.py` | Rebuild figures and tables without rerunning bootstraps |
| Parameter space | `plot_kne_colour_parameter_space.py` | Project colour-selected KNe into 1D/2D physical-parameter spaces |
| Latitude experiment | `run_latitude_experiment.py` | Generate paired latitude-bin populations and run the comparison |
| Latitude analysis | `analyze_kne_latitude_losses.py` | Measure losses within one population |
| Latitude comparison | `compare_kne_latitude_bins.py` | Compare paired bins and estimate a plateau |
| Distance reach | `estimate_kne_lsst_distance_reach.py` | Estimate two-band colour availability versus distance |

`kne_pipeline_common.py`, `dynamic_color_cut_common.py`,
`kne_generation_core.py`, `kne_envelope_core.py`, and
`kne_convergence_common.py` contain shared implementation. Plotting-only code
is separated into dedicated modules. The terminal interfaces of the three
main commands live in `kne_opsim_cli.py`, `kne_generation_cli.py`, and
`kne_envelope_cli.py`; the public scripts reexport them for compatibility.
Generation, envelope-product, and OpSim plots live respectively in
`kne_generation_plots.py`, `kne_envelope_product_plots.py`, and
`kne_opsim_plots.py`. Small inspection tools live in `tools/`.
A more detailed ownership map is available in
[`docs/code_map.md`](docs/code_map.md). The separation between command-line
entry points, numerical modules, and plotting modules is described in
[`docs/refactoring_architecture.md`](docs/refactoring_architecture.md).
The keep/support/archive assessment is recorded in
[`docs/script_inventory.md`](docs/script_inventory.md).

## Installation

The recorded reference environment uses Python 3.12.12, FIESTA 0.2.0,
`rubin_sim` 2.6.1, and `rubin_scheduler` 3.22.0:

```bash
conda env create -f environment.yml
conda activate kne_lsst_analysis
python -m unittest discover -s tests -v
```

A shorter pip installation is also available:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

FIESTA surrogate weights are not stored in this repository. Record the model
source, immutable revision, and checksums for every production analysis. The
semi-analytical Metzger backends require the FIESTA implementation described in
[`docs/metzger_generation.md`](docs/metzger_generation.md).

## 1. Prepare the OpSim sample

Interactive use:

```bash
python get_opsim_baseline.py
```

Example non-interactive run:

```bash
python get_opsim_baseline.py \
  --selection-mode ALL_SURVEY \
  --n-fields 10000 \
  --sampling-mode UNIFORM \
  --seed 12345 \
  --m5-latitude-threshold-deg 20 \
  --outdir opsim_exports_all_survey
```

The output contains sampled positions, visit tables, dust information, and the
canonical `m5_depth_quantiles_by_band.csv` table. m5 statistics are properties
of the OpSim sample and are not recomputed by the light-curve generator.

## 2. Generate a KNe population

```bash
python generate_kne_lightcurves.py
```

Available backends are `Bu2026_MLP`, `Bu2019_MLP`, `MetzgerModel`, and
`MetzgerFullModel`.

The generator supports broad priors or, for `Bu2026_MLP`, posterior samples;
fixed distance or uniform-in-comoving-volume sampling; paired filter systems;
shared intrinsic catalogues; and CSV or sharded Parquet storage. Large runs
should use Parquet. Physical choices, seeds, model metadata, sky assignment,
distance conventions, and storage settings are recorded with every run.

For controlled comparisons, create a shared intrinsic catalogue in the first
run and reuse it in later runs. This preserves event-by-event ejecta parameters
and inclination. Distance replacement is explicit through the catalogue
distance policy; it must not be inferred from filenames.

## 3. Build colour envelopes

```bash
python build_color_envelopes.py --interactive
```

The builder distinguishes four stages:

| Stage | m5 cut | MW extinction included in the magnitude |
|---|---:|---:|
| `intrinsic_no_m5_no_mw` | no | no |
| `no_m5_with_mw` | no | yes |
| `m5_no_mw` | yes | no |
| `m5_with_mw` | yes | yes |

The default interpolation is performed in logarithmic time without
extrapolation. The exact statistical support, bootstrap rules, depth scenario,
and population split are saved with the output. Optional SN Ia inputs and
paired LSST–ZTF depth comparisons are documented in
[`docs/snia_input.md`](docs/snia_input.md) and
[`docs/methodology.md`](docs/methodology.md).

## 4. Measure population convergence

The recommended entry point reads the parent population once and reuses it for
all nested sample sizes:

```bash
python analyze_envelope_population_convergence_fast.py \
  --run-dir runs/YOUR_PARENT_RUN \
  --photometric-system lsst \
  --color-pair g-r \
  --stage m5_no_mw \
  --sample-sizes 1000,3000,10000,30000,50000,75000,100000,200000,300000,400000,500000,1000000 \
  --repetitions 10 \
  --n-boot 200 \
  --resume
```

The largest sample remains the reference population. Two questions are kept
separate:

1. Are the percentile values stable within the requested magnitude tolerance?
2. Does the statistically valid envelope extend to the same time as the
   reference, within the requested temporal tolerance?

The default multi-percentile analysis evaluates `p5–p95`, `p16–p84`, and
`p25–p75`. Existing CSV products can be replotted without loading the parent
population or rerunning bootstrap calculations:

```bash
python replot_envelope_population_convergence.py \
  --analysis-dir runs/YOUR_PARENT_RUN/analysis/YOUR_CONVERGENCE_DIRECTORY \
  --shape-horizon-mode reference-minus-tolerance
```

See [`MULTI_PERCENTILE_CONVERGENCE.md`](MULTI_PERCENTILE_CONVERGENCE.md) and
[`CONVERGENCE_REPLOT_USAGE.md`](CONVERGENCE_REPLOT_USAGE.md).

## 5. Inspect the selected physical-parameter space

```bash
python plot_kne_colour_parameter_space.py \
  --run-dir runs/YOUR_PARENT_RUN \
  --photometric-system lsst \
  --color-pair g-r \
  --stage m5_no_mw \
  --times 1,2,4,6 \
  --parameter-pairs Ye_wind:log10_mej_dyn \
  --bounds-file runs/YOUR_PARENT_RUN/analysis/YOUR_CONVERGENCE_DIRECTORY \
  --envelope-family p16-p84 \
  --hdr-level 90
```

`p16–p84` is the colour-selection interval. `HDR90` is the highest-density
region containing approximately 90% of the selected events in parameter
space; these are independent quantities. Only statistically valid colour bins
are used. See
[`PARAMETER_SPACE_COLOUR_SELECTION.md`](PARAMETER_SPACE_COLOUR_SELECTION.md).

## 6. Study Galactic latitude

```bash
python run_latitude_experiment.py
```

The orchestrator reuses one intrinsic catalogue across latitude bins, runs the
per-bin analysis, and compares paired event outcomes. `--dry-run` prints the
commands before a long calculation. The low/high plateau diagnostic can also
be run directly with:

```bash
python plot_kne_low_high_plateau.py --interactive
```

The plateau is an operational stability criterion, not a physical Galactic
boundary. Read [`docs/low_high_plateau.md`](docs/low_high_plateau.md) before
interpreting it.

## 7. Estimate colour reach with distance

```bash
python estimate_kne_lsst_distance_reach.py
```

For each distance bin, the script measures the fraction of injected events
with at least one temporal bin in which both requested bands are finite and
above the selected global m5 thresholds. It reports binomial intervals and
isotonic crossing distances. This remains a synthetic colour-availability
metric, not a full cadence-dependent survey efficiency.

## Testing and reproducibility

Run the default test suite before every commit that changes scientific code:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

The optional real-FIESTA integration test is enabled explicitly:

```bash
KNE_RUN_FIESTA_INTEGRATION_TESTS=1 \
  python -m unittest discover -s tests -p 'test_metzger_fiesta_integration.py' -v
```

The reproducibility checklist is in
[`docs/reproducibility.md`](docs/reproducibility.md). Numerical equivalence,
not line count, is the acceptance criterion for refactoring scientific code.

## Repository policy

Do not commit:

- `runs/`, OpSim exports, generated figures, or analysis tables;
- external SN Ia populations or posterior archives;
- FIESTA model weights or NMMA joblib inputs;
- local backup files, credentials, private keys, or machine-specific paths.

Keep immutable source revisions and checksums in run metadata or an external
data manifest. Publish large scientific products in a DOI-backed archive when
they are needed to reproduce a paper.

## Attribution, AI use, and citation

`generate_kne_lightcurves.py` was developed from FIESTA's MIT-licensed
`examples/generate_lightcurves.py` and substantially extended. The immutable
upstream reference, scientific citations, model provenance, and external-data
notices are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The use of generative AI for code development, debugging, repository
organisation, and supporting documentation is described transparently in
[`AI_USAGE.md`](AI_USAGE.md). Scientific responsibility and validation remain
with the repository author.

To cite this software, use [`CITATION.cff`](CITATION.cff). The repository's own
source code is released under the [`MIT License`](LICENSE); third-party
software and data retain their respective terms.
