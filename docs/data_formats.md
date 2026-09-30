# Data formats

## OpSim export

`get_opsim_baseline.py` writes one `opsim_visits_fieldNNN.csv` per sampled
field. Visit tables contain the observation MJD, single-letter Rubin band, and
five-sigma depth. `opsim_fields_index.csv` stores field coordinates and MW dust
metadata. `m5_depth_quantiles_by_band.csv` contains visit-weighted p25, p50,
and p75 depths for the global sample and, when coordinates are available, for
high/low Galactic-latitude subsets.

## KNe run

Every injected event has one row in `summary.csv`, including its physical
parameters, luminosity distance, redshift, sky assignment, and extinction
metadata. Long-form synthetic and LSST-like light curves are stored either as
individual CSV files or sharded Parquet datasets. `run_info.txt` records the
configuration and the absolute OpSim source used by downstream m5 discovery.

Generated data products are intentionally excluded from version control.
Archive production inputs and outputs separately with checksums and a manifest.
