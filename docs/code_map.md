# Code map and maintenance status

This map separates public entry points from shared implementation, tests, and
local study products. It is a maintenance guide; scientific definitions remain
in the code and methodology documents referenced below.

## Public entry points

| File | Responsibility | Status |
|---|---|---|
| `get_opsim_baseline.py` | Sample OpSim fields and visits; compute dust and m5 products | Core |
| `generate_kne_lightcurves.py` | Generate physical populations and saved photometric products | Core |
| `build_color_envelopes.py` | Construct colour populations and statistically valid envelopes | Core |
| `analyze_envelope_population_convergence_fast.py` | Recommended nested-subsample convergence workflow | Core |
| `replot_envelope_population_convergence.py` | Rebuild convergence plots from saved CSV products | Core |
| `plot_kne_colour_parameter_space.py` | Map colour selections into physical-parameter space | Core analysis |
| `run_latitude_experiment.py` | Orchestrate paired latitude-bin populations | Core analysis |
| `analyze_kne_latitude_losses.py` | Compute per-population latitude outcomes | Core analysis |
| `compare_kne_latitude_bins.py` | Compare paired latitude bins | Core analysis |
| `plot_kne_low_high_plateau.py` | Diagnose stability of low/high latitude ratios | Core analysis |
| `estimate_kne_lsst_distance_reach.py` | Estimate synthetic two-band colour reach | Core analysis |

## Shared implementation

| File | Ownership |
|---|---|
| `kne_pipeline_common.py` | General run formats, interpolation, OpSim m5 products, posterior loading |
| `dynamic_color_cut_common.py` | Colour matching, event splits, rule evaluation, bootstrap helpers |
| `kne_model_backends.py` | FIESTA surrogate and semi-analytical backend selection |
| `envelope_colour_convergence.py` | Detailed colour-value convergence calculations |
| `envelope_multiquantile_convergence.py` | Multi-percentile value and temporal-extent convergence |
| `analyze_envelope_population_convergence.py` | Legacy subprocess workflow and shared convergence utilities used by the fast path |

The legacy convergence entry point cannot yet be removed because the fast
implementation imports its parsing, aggregation, and plotting utilities. A
future refactor should move those reusable functions into a neutral module,
then retain the legacy script only as a compatibility wrapper.

## Utilities

Files under `tools/` inspect depth distributions, SN Ia distances, or individual
Parquet events. They are useful diagnostics but are not pipeline stages.

## Tests

Every scientific analysis added to the public workflow must have at least one
test covering its numerical convention or failure mode. The default suite is
deliberately independent of surrogate weights and large data. Tests requiring
real FIESTA models or downloads remain opt-in.

## Files that should remain local

Do not commit run directories, generated PNG/CSV products, `.bak` files,
repository-audit snapshots, OpSim databases, surrogate weights, posterior
archives, or external SN Ia populations. Representative fixtures should be
small, documented, and stored under `tests/` only when a test requires them.

## Refactoring priorities

1. Split the 1,000-line `build_color_envelopes.main()` into orchestration
   functions without changing calculations.
2. Separate interactive configuration from execution in
   `generate_kne_lightcurves.py`; the current module performs substantial work
   at import time.
3. Move shared convergence utilities out of the legacy entry point.
4. Add golden-output tests for a small deterministic population before moving
   interpolation, envelope, or extinction code.
5. Keep performance-critical population loading vectorized and benchmarked.

Reducing total line count is not itself an acceptance criterion. A refactor is
accepted only when the public commands remain understandable and deterministic
tests demonstrate unchanged scientific results within explicit tolerances.
