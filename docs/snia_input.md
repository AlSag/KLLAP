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

For a direct KNe--SN Ia comparison, `build_color_envelopes.py` can instead
measure an event-by-event peak in a selected band of the colour (for example
`g` for `g-r`, or `z`/`y` for `z-y`). The same definition is used for both
populations:

- search the complete saved, noise-free reference-band curve;
- consider only interior local minima of magnitude (flux maxima), never the
  first or last sample;
- require a configurable magnitude prominence (default: 0.05 mag);
- select the most prominent candidate and refine its date with a bracketed,
  convex three-point quadratic fit;
- exclude an event if no candidate passes, recording the reason in
  `kne_peak_time_diagnostics.csv` or `snia_peak_time_diagnostics.csv`.

This rule deliberately handles Bu2026 curves whose first saved point is very
bright, followed by fading, rebrightening, and a later physical maximum. The
first point can be reported as the brightest saved sample, but cannot become a
resolved peak because it lies on the model boundary. SN Ia peak finding uses
`mag_perfect` even when noisy `mag` is selected for the colour itself, so noise
does not redefine the temporal origin.

The dataset itself is not included because of its size and provenance. Pass
its path explicitly to `build_color_envelopes.py` and record its checksum in
the analysis release.
