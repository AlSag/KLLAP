# Reviewable pipeline architecture

This document separates user-facing commands, scientific calculations,
storage transformations, and plotting code. It is intended as a review guide:
the command-line scripts remain convenient entry points, while reusable
modules contain operations that can be tested without launching a simulation.

## Generation path

```mermaid
flowchart TD
    O[get_opsim_baseline.py] --> E[OpSim exports and m5 tables]
    E --> G[generate_kne_lightcurves.py]
    B[kne_model_backends.py] --> G
    C[kne_pipeline_common.py] --> G
    G --> P[Saved light curves and summary]
    K[kne_generation_core.py] --> G
    G --> V[kne_generation_plots.py]
```

### Public command

`generate_kne_lightcurves.py` orchestrates one generation run and reexports
its terminal functions from `kne_generation_cli.py`. Running it as a script
preserves the interactive workflow.
Importing it only defines functions and does not prompt, initialize Rubin
bandpasses, create directories, or generate events.

The three main commands have dedicated interface modules:

| Public command | Terminal-only module |
|---|---|
| `get_opsim_baseline.py` | `kne_opsim_cli.py` |
| `generate_kne_lightcurves.py` | `kne_generation_cli.py` |
| `build_color_envelopes.py` | `kne_envelope_cli.py` |

These modules contain parsers, menus, and validation of option combinations.
They do not import Rubin, FIESTA, JAX, Astropy, or plotting libraries. Public
commands reexport the historical function names so existing commands and
external imports continue to work.

### Scientific backends

`kne_model_backends.py` selects and validates the FIESTA surrogate or
semi-analytical model. Changes here can alter the physical prediction and
therefore require model-level numerical tests.

### Deterministic generation operations

`kne_generation_core.py` contains small transformations of existing arrays or
tables:

- construction of Milky-Way extinction columns and summaries;
- trimming after the last LSST-like detection;
- the inclusive synthetic m5 cut;
- selection of the stored MW-extincted magnitude;
- deterministic derivation of latitude-specific sky seeds.

It deliberately does not import FIESTA, JAX, Astropy, or Rubin. Its functions
receive their inputs explicitly and are covered by small deterministic tests.

### Diagnostic presentation

`kne_generation_plots.py` contains only the figures produced during
generation. It receives already calculated arrays and tables. It does not
draw physical parameters, modify magnitudes, apply depth cuts, classify
detections, or write scientific CSV/Parquet tables.

The extraction was checked on representative synthetic, LSST-like, overlay,
and numerical-floor figures; the before/after PNG arrays were identical pixel
by pixel.

### General shared utilities

`kne_pipeline_common.py` contains format and discovery utilities shared by
several pipeline stages. `kne_m5.py` owns m5-table discovery and percentile
selection. These shared modules avoid maintaining subtly different copies of
the same convention in multiple scripts.

`get_opsim_baseline.py` follows the same import-safety rule as the generator:
importing it does not parse terminal input, locate the Rubin database, or
create export products.

### Envelope construction

`build_color_envelopes.py` remains the user-facing command and orchestration
layer; its parser and menus live in `kne_envelope_cli.py`.
`kne_envelope_core.py` owns the numerical conventions shared with the
convergence and parameter-space analyses: MW extinction, reproducible nested
subsets, bootstrap percentile validity, tail support, and selection of the
widest valid symmetric envelope.

`kne_envelope_plots.py` receives already computed tracks and envelope tables.
It cannot select events, calculate percentiles, apply extinction, or impose an
m5 cut. Representative figures produced before and after extraction were
identical pixel by pixel.

## Analysis path

| Layer | Public entry point | Internal dependencies |
|---|---|---|
| Envelope construction | `build_color_envelopes.py` | `kne_envelope_core.py`, `kne_envelope_plots.py`, `dynamic_color_cut_common.py`, `kne_m5.py` |
| Convergence | `analyze_envelope_population_convergence_fast.py` | `kne_convergence_common.py`, `kne_envelope_core.py`, `envelope_colour_convergence.py`, `envelope_multiquantile_convergence.py` |
| Saved-result replot | `replot_envelope_population_convergence.py` | `envelope_multiquantile_convergence.py` |
| Realization comparison | `plot_colour_envelope_realizations.py` | `dynamic_color_cut_common.py`, `kne_m5.py` |
| Parameter-space maps | `plot_kne_colour_parameter_space.py` | shared population loaders |
| Iso-colour maps | `plot_kne_colour_isocontours.py` | `plot_kne_colour_parameter_space.py`, `kne_m5.py` |
| Latitude experiment | `run_latitude_experiment.py` | generator, latitude analyzer, comparator |

Named m5 scenarios (`worst_p25`, `median_p50`, and `best_p75`) are owned by
`kne_m5.py`. Commands may choose different defaults, but they must not carry a
private copy of the allowed names or percentile values. Likewise, the four
photometric envelope stages are owned by `kne_envelope_core.py`.

The recommended fast convergence command no longer imports the legacy entry
point. Shared parsing, aggregation, and plotting definitions live in
`kne_convergence_common.py`.

## How to review a scientific change

1. Identify the owning module in the tables above.
2. Check whether the change affects physical draws, interpolation, extinction,
   depth selection, statistical validity, or presentation only.
3. Add or update a deterministic test for every changed numerical convention.
4. Run `python -m compileall -q .` and
   `python -m unittest discover -s tests -v`.
5. Compare representative scientific tables before and after the change.
6. For plotting-only changes, compare the underlying plotted values; use
   pixel comparison when exact visual preservation is expected.

## Remaining refactoring order

1. Split the remaining `build_color_envelopes.main()` by output family while
   preserving its tables exactly.
2. Add a deterministic golden-output envelope fixture before moving population
   I/O.
3. Compare legacy and fast convergence outputs end to end.
4. Archive a legacy script only after no public command, import, test, or
   documentation page depends on it.

Reducing line count is not an acceptance criterion by itself. A refactor is
accepted when the public commands remain understandable and deterministic
tests demonstrate that the scientific outputs retain the declared numerical
conventions.
