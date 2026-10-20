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
| `plot_convergence_support_diagnostics.py` | Display per-bin support and percentile errors from convergence CSV products | Diagnostic |
| `plot_colour_envelope_realizations.py` | Compare nested population sizes across independent realizations and inspect bootstrap distributions | Core analysis |
| `plot_kne_colour_parameter_space.py` | Map colour selections into physical-parameter space | Core analysis |
| `plot_kne_colour_isocontours.py` | Map median colour and iso-colour contours in physical-parameter planes | Core analysis |
| `run_latitude_experiment.py` | Orchestrate paired latitude-bin populations | Core analysis |
| `analyze_kne_latitude_losses.py` | Compute per-population latitude outcomes | Core analysis |
| `compare_kne_latitude_bins.py` | Compare paired latitude bins | Core analysis |
| `plot_kne_low_high_plateau.py` | Diagnose stability of low/high latitude ratios | Core analysis |
| `estimate_kne_lsst_distance_reach.py` | Estimate synthetic two-band colour reach | Core analysis |

## Shared implementation

| File | Ownership |
|---|---|
| `kne_pipeline_common.py` | General run formats, interpolation, OpSim m5 products, posterior loading |
| `kne_opsim_cli.py` | OpSim command-line parser and terminal menu |
| `kne_generation_cli.py` | Generation parser, terminal menu, and option validation |
| `kne_envelope_cli.py` | Envelope parser and terminal menu |
| `dynamic_color_cut_common.py` | Colour matching, event splits, rule evaluation, bootstrap helpers |
| `kne_m5.py` | Shared discovery and parsing of m5 percentile tables and raw OpSim depth files |
| `kne_model_backends.py` | FIESTA surrogate and semi-analytical backend selection |
| `kne_generation_core.py` | Deterministic generation table transformations |
| `kne_generation_plots.py` | Generation-only diagnostic figures |
| `kne_envelope_core.py` | Extinction, nested subsets, percentile validity, and envelope selection |
| `kne_envelope_plots.py` | Plotting primitives for envelope products |
| `kne_convergence_common.py` | Parsing, aggregation, and plotting conventions shared by convergence commands |
| `envelope_colour_convergence.py` | Detailed colour-value convergence calculations |
| `envelope_multiquantile_convergence.py` | Multi-percentile value and temporal-extent convergence |
| `analyze_envelope_population_convergence.py` | Legacy subprocess workflow retained for compatibility |

The fast convergence command no longer imports the legacy entry point. Both
use `kne_convergence_common.py`; the older command remains until an end-to-end
equivalence fixture establishes whether it can be archived.

## Utilities

Files under `tools/` inspect depth distributions, SN Ia distances, or individual
Parquet events. They are useful diagnostics but are not pipeline stages.

## Tests

Every scientific analysis added to the public workflow must have at least one
test covering its numerical convention or failure mode. The default suite is
deliberately independent of surrogate weights and large data. Tests requiring
real FIESTA models or downloads remain opt-in.

The realization comparison tests distinguish median validity from envelope-tail
validity and preserve the inclusive 20-event p16/p84 boundary.  The isocontour
tests preserve cell medians, logarithmic mass-axis conversion, and the rule
that smoothing must never populate an unsupported parameter-space cell.

## Files that should remain local

Do not commit run directories, generated PNG/CSV products, `.bak` files,
repository-audit snapshots, OpSim databases, surrogate weights, posterior
archives, or external SN Ia populations. Representative fixtures should be
small, documented, and stored under `tests/` only when a test requires them.

## Refactoring priorities

1. Add a small golden-output envelope fixture before moving population I/O.
2. Split the remaining `build_color_envelopes.main()` orchestration by output
   family without changing tables.
3. Compare the legacy and fast convergence commands, then remove the legacy
   command if it has no unique scientific output.
3. Compare the legacy and fast convergence commands end to end.
4. Keep performance-critical population loading vectorized and benchmarked.

Retention decisions are recorded in
[`script_inventory.md`](script_inventory.md).

Reducing total line count is not itself an acceptance criterion. A refactor is
accepted only when the public commands remain understandable and deterministic
tests demonstrate unchanged scientific results within explicit tolerances.
