# Script inventory

The repository exposes exactly eight public workflows. Each creates a distinct
scientific product, owns an interactive terminal interface, and can be imported
without starting a calculation.

## Public workflows

| Script | Scientific products |
|---|---|
| `get_opsim_baseline.py` | OpSim field/visit exports, dust, m5 tables and diagnostics |
| `generate_kne_lightcurves.py` | KNe populations, synthetic or observed-like light curves and metadata |
| `build_color_envelopes.py` | colour tracks, percentile envelopes and stage comparisons |
| `analyze_envelope_population_convergence_fast.py` | nested-sample convergence tables and figures |
| `plot_colour_envelope_realizations.py` | realization comparisons and bootstrap distributions |
| `plot_kne_colour_isocontours.py` | parameter-space KNe-count maps and median iso-colour contours |
| `analyze_kne_latitude.py` | population, paired-bin and low/high-cut latitude products |
| `run_latitude_experiment.py` | complete paired latitude-bin generation and comparison |

Every public workflow is launched as `python SCRIPT.py`. Scientific automation
imports an explicit default-configuration function and a run function rather
than invoking an undocumented command-line surface.

## Shared implementation

Only definitions used by more than one public workflow remain separate:

- `kne_pipeline_common.py`: run formats, OpSim products, interpolation and posterior loading;
- `kne_population_io.py`: Parquet/CSV discovery and population metadata;
- `kne_m5.py`: m5 scenarios and depth-table discovery;
- `kne_envelope_core.py`: envelope stages and statistical-support rules;
- `dynamic_color_cut_common.py`: colour matching and bootstrap helpers.

These files do not create standalone products and are not public commands.

## Consolidated implementation

The following former one-purpose modules are now explicit sections of their
owning public scripts:

- generation and OpSim interface, calculations, and figures;
- envelope construction, convergence calculations, and figures;
- realization and iso-colour population calculations;
- the three Galactic-latitude analyses.

In particular, `kne_latitude_population.py`, `kne_latitude_comparison.py`, and
`kne_latitude_plateau.py` were consolidated into `analyze_kne_latitude.py`.

## Removed workflows

Redundant or optional commands removed from the maintained pipeline include:

- distance-reach estimation;
- convergence-only replot and support-diagnostic commands;
- colour-selected HDR parameter-space projections;
- the legacy repeated-subprocess convergence workflow.

Generated runs, figures, audit snapshots, numbered copies, and backup sources
remain untracked. Deleted source remains recoverable from Git history.
