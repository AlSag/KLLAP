# Code map

This map identifies the eight product-producing scripts and the five genuinely
shared implementation modules in the reviewable pipeline.

## Public product-producing scripts

| Script | Interface block | Scientific block | Figure/output block |
|---|---|---|---|
| `get_opsim_baseline.py` | survey and sampling questions | OpSim selection, coordinates, dust and depth statistics | visit, depth and sky diagnostics |
| `generate_kne_lightcurves.py` | model, population, storage and seed questions | physical draws, model evaluation, interpolation, extinction and selection | tables, metadata and diagnostic light curves |
| `build_color_envelopes.py` | population, colour, m5 and support questions | colour interpolation, stage masks, bootstrap and percentiles | envelopes and comparison products |
| `analyze_envelope_population_convergence_fast.py` | nested sizes, tolerances and bootstrap questions | value and temporal-extent convergence | summary tables and convergence figures |
| `plot_colour_envelope_realizations.py` | realization, time and validity questions | nested prefixes, percentiles, tails and bootstrap | realization and bootstrap-distribution figures |
| `plot_kne_colour_isocontours.py` | plane, time, m5, axis-domain and contour questions | population loading, cell counts, cell medians and supported-cell smoothing | KNe-count heatmaps with individual and overview median iso-colour maps |
| `analyze_kne_latitude.py` | analysis-mode and method-specific questions | population losses, paired-bin bootstrap and moving-cut plateau | latitude tables and figures |
| `run_latitude_experiment.py` | experiment and resume questions | paired generation/analysis configuration | complete latitude experiment outputs |

## Shared modules

| Module | Why it remains shared |
|---|---|
| `kne_pipeline_common.py` | stable data formats and OpSim/posterior functions used by several stages |
| `kne_population_io.py` | identical population discovery and metadata rules across analyses |
| `kne_m5.py` | one canonical interpretation of named depth scenarios |
| `kne_envelope_core.py` | one canonical set of stages and statistical-support definitions |
| `dynamic_color_cut_common.py` | identical colour matching and bootstrap helpers across workflows |

A shared module is retained only when copying it would create competing
scientific definitions. It has no terminal entry point and produces no output
on import.

## Review rules

1. Start from the public script that owns the output.
2. Read its labelled interface, calculation, and figure sections separately.
3. Follow an import only when the definition is shared in the table above.
4. Add a deterministic test for every changed numerical convention.
5. Run `python -m compileall -q .` and the complete unit-test suite.
6. Compare scientific tables before and after any numerical refactor.

Line count alone is not an acceptance criterion. The scientific conventions,
random seeds, support rules, and absence of extrapolation must remain explicit.
