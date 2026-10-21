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

## Internal modules

- `kne_*_cli.py`: parsers, interactive menus, and validation;
- `kne_*_core.py`, `envelope_*_convergence.py`: numerical definitions;
- `kne_*_plots.py`: plotting implementations;
- `kne_population_io.py`, `kne_envelope_population.py`, and
  `kne_colour_population.py`: saved-population loading and interpolation;
- `kne_latitude_population.py`, `kne_latitude_comparison.py`, and
  `kne_latitude_plateau.py`: the three explicit latitude analyses dispatched
  by `analyze_kne_latitude.py`.

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
`kne_colour_population.py`; no retained command imports a deleted script.

Generated runs, local audit reports, numbered copies, and backup sources must
remain untracked. A deletion is committed separately enough to remain fully
recoverable from Git history.
