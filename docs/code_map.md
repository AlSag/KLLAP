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
| `plot_colour_envelope_realizations.py` | Compare nested population sizes across independent realizations and inspect bootstrap distributions | Core analysis |
| `plot_kne_colour_isocontours.py` | Map median colour and iso-colour contours in physical-parameter planes | Core analysis |
| `run_latitude_experiment.py` | Orchestrate paired latitude-bin populations | Core analysis |
| `analyze_kne_latitude.py` | Unified population, paired-bin, and plateau latitude analysis | Core analysis |

## Shared implementation

Each root entry point delegates to `command.py`, `cli.py`, `realizations.py`,
`isocontours.py`, or `experiment.py` in its matching `kllap/` domain. The
remaining files below have one narrower responsibility.

| File | Ownership |
|---|---|
| `kllap/opsim/command.py` | OpSim selection and export orchestration |
| `kllap/common.py` | General run formats, interpolation, OpSim m5 products, posterior loading |
| `kllap/opsim/cli.py` | OpSim command-line parser and terminal menu |
| `kllap/generation/command.py` | Population and light-curve generation orchestration |
| `kllap/generation/cli.py` | Generation parser, terminal menu, and option validation |
| `kllap/envelope/command.py` | Colour-envelope orchestration and scientific table output |
| `kllap/envelope/cli.py` | Envelope parser and terminal menu |
| `kllap/colour/common.py` | Colour matching, event splits, rule evaluation, bootstrap helpers |
| `kllap/m5.py` | Shared discovery and parsing of m5 percentile tables and raw OpSim depth files |
| `kllap/generation/backends.py` | FIESTA surrogate and semi-analytical backend selection |
| `kllap/generation/core.py` | Deterministic generation table transformations |
| `kllap/generation/plots.py` | Generation-only diagnostic figures |
| `kllap/opsim/plots.py` | OpSim depth and revisit-cadence figures |
| `kllap/envelope/core.py` | Extinction, nested subsets, percentile validity, and envelope selection |
| `kllap/population_io.py` | Parquet/CSV discovery and scientific metadata selection shared by analyses |
| `kllap/envelope/population.py` | No-extrapolation interpolation and native colour tracks for envelope construction |
| `kllap/envelope/plots.py` | Plotting primitives for envelope products |
| `kllap/envelope/product_plots.py` | Complete envelope, AT2017gfo, SN Ia, and comparison figures |
| `kllap/convergence/common.py` | Parsing, aggregation, and plotting conventions shared by convergence commands |
| `kllap/convergence/command.py` | Single-load nested-population convergence orchestration |
| `kllap/convergence/colour.py` | Detailed colour-value convergence calculations |
| `kllap/convergence/multiquantile.py` | Multi-percentile value and temporal-extent convergence |
| `kllap/colour/population.py` | Metadata loading, parameter transforms, and colour interpolation for iso-colour maps |
| `kllap/colour/realizations.py` | Independent-realization and bootstrap comparison products |
| `kllap/colour/isocontours.py` | Median-colour grids and iso-colour figures |
| `kllap/latitude/experiment.py` | Paired-bin generation and analysis orchestration |
| `kllap/latitude/cli.py` | Unified latitude-analysis dispatcher |
| `kllap/latitude/population.py` | Per-population latitude-loss calculations and products |
| `kllap/latitude/comparison.py` | Paired disjoint-bin bootstrap comparison and products |
| `kllap/latitude/plateau.py` | Low/high latitude-cut stability calculations and products |
The single-load convergence command replaces the former subprocess workflow.
Its `--stability-metric last-edge` mode preserves the original last-bin
analysis, while avoiding repeated reads and intermediate envelope builds.

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
