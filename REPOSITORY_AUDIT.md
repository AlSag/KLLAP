# Repository audit — 2026-10-14

## Baseline

- Git branch in the supplied snapshot: `main` at `f893c2e`.
- Files modified relative to `origin/main`: `generate_kne_lightcurves.py`,
  `build_color_envelopes.py`, and `THIRD_PARTY_NOTICES.md`.
- Several new analysis scripts, documentation files, and tests were untracked.
- Default tests before the audit: 68 passed, 2 optional tests skipped.
- `CITATION.cff` was invalid YAML because a prose citation had been appended
  outside the CFF structure.

## Public workflow to retain

### Generation

- `get_opsim_baseline.py`
- `generate_kne_lightcurves.py`
- `kne_model_backends.py`
- `kne_pipeline_common.py`

### Colour and convergence analysis

- `build_color_envelopes.py`
- `dynamic_color_cut_common.py`
- `analyze_envelope_population_convergence_fast.py`
- `envelope_colour_convergence.py`
- `envelope_multiquantile_convergence.py`
- `replot_envelope_population_convergence.py`
- `plot_kne_colour_parameter_space.py`

The original audit found that the fast convergence entry point imported the
legacy subprocess command. This dependency has since been removed: both
commands now import parsing, aggregation, and plotting definitions from
`kne_convergence_common.py`. The legacy command is retained only as an
explicitly documented compatibility workflow pending an end-to-end output
comparison.

### Latitude and reach analysis

- `run_latitude_experiment.py`
- `analyze_kne_latitude_losses.py`
- `compare_kne_latitude_bins.py`
- `plot_kne_low_high_plateau.py`
- `estimate_kne_lsst_distance_reach.py`

### Utilities

- `tools/extract_parquet_event.py`
- `tools/inspect_snia_luminosity_distance.py`
- `tools/plot_ztf_depth_distributions.py`

## Files not to commit

- `analyze_envelope_population_convergence_fast.py.bak`;
- `paired_intrinsic_convergence_*.csv`;
- `repository_git_*.txt` and `repository_changes_*.txt`;
- `LSST_*_seed_*/`, `PS1_*_seed_*/`, and other generated runs;
- figures, Parquet shards, OpSim databases, posterior archives, model weights,
  and external data.

These are now covered explicitly by `.gitignore`. They have not been treated
as source files in the update package.

## Documentation and metadata changes

- rewrote `README.md` around the current public workflow;
- corrected `CITATION.cff` and identified Alex Jacquesson as the software
  author;
- replaced the MIT licence placeholder;
- expanded third-party provenance and model/data distinctions;
- added `AI_USAGE.md` with the requested Yes/No declaration;
- added `docs/code_map.md` and a GitHub pre-push checklist;
- added a minimal continuous-integration workflow and repository-metadata test.

## Refactoring decision

The two largest scripts are genuinely difficult to maintain:

- `build_color_envelopes.py`: about 4,200 lines, including a `main()` of about
  1,000 lines;
- `generate_kne_lightcurves.py`: about 3,800 lines, including a configuration
  function of about 400 lines and substantial import-time execution.

They should be split, but not in the same commit that publishes many new
scientific analyses. The safe order is:

1. publish the current tested workflow and metadata cleanup;
2. add deterministic golden-output fixtures for a small population;
3. separate configuration/orchestration from numerical functions;
4. compare outputs, run time, and memory before and after each extraction.

This avoids a large, hard-to-review commit in which scientific changes and
architectural changes are mixed.

## Items still requiring author confirmation

- exact immutable origin and checksums of the production `Bu2026_MLP` assets;
- whether version `0.2.0` is the desired release tag;
- whether any generated result table should be published separately with a DOI;
- whether external SN Ia data used in a paper can be redistributed or should
  remain a cited runtime input.
