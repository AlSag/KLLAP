# Optional SN Ia input

The tested mock dataset is a long-format Parquet table containing 10,000
simulated Type Ia supernovae and LSST `ugrizy` observations. Each row is one
observation in one band.

Expected columns:

| Column | Meaning |
|---|---|
| `sn_id` | SN identifier |
| `mjd` | observation MJD |
| `band` | LSST band |
| `mag`, `magerr` | noisy AB magnitude and uncertainty |
| `flux_nJy`, `fluxerr_nJy` | noisy flux and uncertainty |
| `flux_perfect_nJy`, `mag_perfect` | noise-free model values |
| `redshift` | simulated redshift |
| `t0` | SALT2 time of maximum light |
| `x0`, `x1`, `c` | SALT2 parameters |
| `ra`, `dec` | simulated coordinates |

The example population was generated with a SALT2-H17 model and a simplified
cadence. Its native temporal reference is peak time `t0`, not an observed
physical explosion/collapse epoch. A user-defined offset can be explored, but
it is a model assumption and must not be presented as a measured collapse
time.

The dataset itself is not included because of its size and provenance. Pass
its path explicitly to `build_color_envelopes.py` and record its checksum in
the analysis release.
