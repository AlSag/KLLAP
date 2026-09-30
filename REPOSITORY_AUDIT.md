# Repository triage record

## Retained as the public workflow

- `get_opsim_baseline.py`
- `generate_kne_lightcurves.py` (the latitude-bin-capable superset)
- `kne_pipeline_common.py`
- `dynamic_color_cut_common.py`
- `build_color_envelopes.py` (renamed because it now contains four stages)
- `analyze_kne_latitude_losses.py`
- `compare_kne_latitude_bins.py`
- `run_latitude_experiment.py`
- `tools/inspect_snia_luminosity_distance.py`
- `tools/extract_parquet_event.py`

## Superseded or omitted

The cumulative low/high latitude scripts, older three-stage envelope builders,
older generator copies, standalone m5 scripts, one-off plotting scripts,
diagnostics, hard-coded AT2017gfo overlays, and exploratory convergence scripts
were not placed in the public core. They should remain in a separate archive
until the published results no longer depend on them.

The former standalone `compute_m5_depth_quantiles.py` is functionally absorbed
by `kne_pipeline_common.py` and called automatically by
`get_opsim_baseline.py`. This avoids maintaining incompatible m5 schemas and
removes the need to copy a threshold table into every run.

## Items requiring human confirmation before publication

- replace the placeholder GitHub URL in `CITATION.cff`;
- confirm the author name and chosen MIT license;
- document the exact source/checksum of the local `Bu2026_MLP` model assets;
- decide whether any archived diagnostics underlie a figure in a thesis or
  paper and must therefore be restored under a `studies/` directory;
- confirm the provenance and redistribution terms of external SN Ia data.
