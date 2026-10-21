# Script inventory

The repository exposes only commands that create a distinct scientific
product. Shared calculations, data loading, plotting helpers, and terminal
interfaces use descriptive module names and are not separate workflows.

## Public commands

| Command | Scientific output |
|---|---|
| `get_opsim_baseline.py` | OpSim fields, visits, dust, and m5 summaries |
| `generate_kne_lightcurves.py` | Physical populations and saved light curves |
| `build_color_envelopes.py` | Colour populations and percentile envelopes |
| `analyze_envelope_population_convergence_fast.py` | Nested-sample convergence tables and figures |
| `plot_colour_envelope_realizations.py` | Independent-realization and bootstrap comparisons |
| `plot_kne_colour_isocontours.py` | Median-colour maps and iso-colour contours |
| `run_latitude_experiment.py` | Paired latitude-bin generation and full analysis |
| `analyze_kne_latitude.py` | Population, paired-bin, or low/high plateau products |

## Internal package

The root commands contain no scientific implementation. Their code is grouped
under `kllap/` by domain:

- `opsim/`: OpSim command, CLI, and plots;
- `generation/`: command, CLI, model backends, deterministic core, and plots;
- `envelope/`: command, CLI, population interpolation, statistics, and plots;
- `convergence/`: command and the colour/multiquantile convergence methods;
- `colour/`: shared colour rules, population loading, realizations, and
  isocontours;
- `latitude/`: experiment, unified CLI, population analysis, paired-bin
  comparison, and plateau analysis.

Only genuinely cross-domain code stays directly under `kllap/`:
`common.py`, `m5.py`, and `population_io.py`.

The latitude modules remain separate because they answer different questions:
one population's losses, paired disjoint-bin ratios, and stability under a
moving low/high cut. They are implementation modules, not competing public
commands.

## Removed workflows

The following optional or redundant commands and their dedicated documents
were deliberately removed:

- distance-reach estimation;
- convergence-only replot and support-diagnostic commands;
- colour-selected HDR parameter-space projections.

Iso-colour maps retain the required population loading and interpolation in
`kllap/colour/population.py`; no retained command imports a deleted script.

Generated runs, local audit reports, numbered copies, and backup sources must
remain untracked. A deletion is committed separately enough to remain fully
recoverable from Git history.
