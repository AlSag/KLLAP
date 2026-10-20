# Script inventory and retention policy

This inventory is the deletion gate for the reviewable pipeline. A tracked
scientific file is not removed merely because it is long or imported by
another command. Removal is safe only after imports, tests, documentation, and
a representative workflow no longer depend on it.

## Retained public commands

| Command | Scientific role |
|---|---|
| `get_opsim_baseline.py` | Export OpSim fields, visits, dust values, and reusable m5 summaries |
| `generate_kne_lightcurves.py` | Draw KNe populations and generate saved light curves |
| `build_color_envelopes.py` | Orchestrate colour-population and envelope construction |
| `analyze_envelope_population_convergence_fast.py` | Evaluate nested-sample convergence |
| `replot_envelope_population_convergence.py` | Replot saved convergence products without recomputation |
| `plot_colour_envelope_realizations.py` | Compare independent populations and bootstrap stability |
| `plot_kne_colour_parameter_space.py` | Project colour selections into physical-parameter space |
| `plot_kne_colour_isocontours.py` | Plot median-colour maps and iso-colour contours |
| `run_latitude_experiment.py` | Orchestrate paired low/high-latitude populations |
| `analyze_kne_latitude_losses.py` | Measure losses within one latitude population |
| `compare_kne_latitude_bins.py` | Compare paired latitude populations |
| `plot_kne_low_high_plateau.py` | Diagnose stability of low/high ratios |

`plot_convergence_support_diagnostics.py` and
`estimate_kne_lsst_distance_reach.py` are retained diagnostics. They are not
required for every run, but they produce distinct scientific products.

## Shared implementation

- `kne_pipeline_common.py`: run formats, population loading, interpolation,
  and discovery helpers;
- `kne_m5.py`: m5-table discovery and percentile selection;
- `kne_model_backends.py`: FIESTA backend selection and validation;
- `kne_generation_core.py` and `kne_generation_plots.py`: generation
  calculations and presentation;
- `kne_envelope_core.py`: extinction, nested subsets, percentile validity,
  tail support, and envelope selection;
- `kne_envelope_plots.py`: envelope plotting primitives only;
- `kne_convergence_common.py`: definitions shared by the legacy and fast
  convergence commands;
- `dynamic_color_cut_common.py`: colour, event-split, and bootstrap helpers;
- `envelope_colour_convergence.py` and
  `envelope_multiquantile_convergence.py`: convergence calculations.

Keeping shared functions in neutral modules provides one versioned definition
of every convention and lets tests exercise it without launching an
interactive workflow.

## Compatibility command

`analyze_envelope_population_convergence.py` is the older subprocess-based
workflow. The fast command no longer imports it: both now use
`kne_convergence_common.py`. It remains available until an end-to-end fixture
establishes whether the subprocess workflow still provides a distinct
reproducibility benefit.

## Cleanup status

| Candidate | Current decision | Evidence still required before deletion |
|---|---|---|
| `analyze_envelope_population_convergence.py` | Keep temporarily | End-to-end legacy/fast comparison on the same small saved population |
| `plot_convergence_support_diagnostics.py` | Keep | Produces per-bin support and heatmap diagnostics not emitted by the main convergence command |
| `estimate_kne_lsst_distance_reach.py` | Keep | Produces distance-completeness products not emitted elsewhere |
| `tools/*.py` | Keep as optional utilities | Each has a distinct data-inspection or conversion purpose |

The repository contains no tracked `.bak`, numbered-copy, or `*_before_*`
Python source file. Generated runs and local review reports remain excluded by
`.gitignore` rather than being treated as code.

Internal duplication can be removed sooner than commands. The four envelope
stages now have one canonical definition in `kne_envelope_core.py`; named m5
scenarios, labels, and percentages have one canonical definition in
`kne_m5.py`. Public compatibility aliases are retained where external imports
may reasonably depend on them.

## Deletion checklist

A script may be archived or deleted only when all of the following are true:

1. no retained command imports or launches it;
2. no test or documentation page treats it as canonical;
3. a retained command reproduces every still-useful output;
4. deterministic before/after comparisons show no lost scientific product;
5. removal is isolated in a reversible commit.

No tracked scientific script currently satisfies all five conditions.
