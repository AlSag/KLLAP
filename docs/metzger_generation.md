# Generate a Metzger kilonova population with FIESTA

The generator supports `Bu2026_MLP`, `Bu2019_MLP`, `MetzgerModel`, and
`MetzgerFullModel`. Keep `kllap/generation/backends.py` and `kllap/common.py`
beside `generate_kne_lightcurves.py`. The existing envelope builder reads the
saved magnitudes, so it does not need to load a Metzger model itself.
The included `build_color_envelopes.py` update removes its requirement for
Bulla's dynamical/wind masses and adds Metzger parameters to exit tables.
Replace that builder as well as the generator when installing this update.

## Install and check the environment

First try your existing environment:

```bash
conda activate fiesta_bu2019
python -c "from fiesta.models.analytical_models import MetzgerModel, MetzgerFullModel; print('Metzger models available')"
```

If the import fails, use a separate environment cloned from your working one:

```bash
conda create --name fiesta_metzger --clone fiesta_bu2019
conda activate fiesta_metzger
python -m pip install "fiestaEM @ git+https://github.com/nuclear-multimessenger-astronomy/fiestaEM.git@6ba093cc13cb768a089b07768e550a6a46b737ca"
python -c "from fiesta.models.analytical_models import MetzgerModel, MetzgerFullModel; print('OK')"
```

The pinned commit is the implementation inspected and tested for this update.
A package version number alone is not enough to establish the presence of
these classes. The installation keeps the pipeline's existing OpSim/Rubin
dependencies. No surrogate weight files are needed for Metzger. `sncosmo`
may download filter transmission curves on first use.

## Which model?

Both are the implementations provided by FIESTA in
[`kilonova_models.py`](https://github.com/nuclear-multimessenger-astronomy/fiestaEM/blob/6ba093cc13cb768a089b07768e550a6a46b737ca/src/fiesta/models/analytical_models/kilonova_models.py).

| Choice | FIESTA implementation |
| --- | --- |
| `MetzgerModel` | 300 logarithmically spaced mass shells; implementation described upstream as matching NMMA's `eff_metzger_lc` |
| `MetzgerFullModel` | 200 shells spaced linearly in velocity; Barnes et al. thermalisation; optional neutron precursor; implementation described upstream as matching Redback |

They share four parameters, but their shell layouts, thermalisation and
initial conditions differ. Their outputs need not agree. Both use a spherical
photosphere and a blackbody spectrum integrated through the requested bands.
They do not model the same spectral/angular structure as the POSSIS surrogates.
They are semi-analytical approximations, not alternative trained MLPs.

## Population parameters

These defaults are **explicit choices of simulation priors**, not training
limits or a recovered historical population. All are adjustable through the
terminal menu or arguments. Each is drawn uniformly in the variable listed:

| Variable | Meaning | Default interval |
| --- | --- | --- |
| `log10_mej` | log10 of ejecta mass in solar masses | -3 to -1 (0.001 to 0.1 solar masses) |
| `log10_vej` | log10 of minimum ejecta velocity in units of c | log10(0.05) to log10(0.3) |
| `beta` | exponent of the mass/velocity profile | 2 to 4 |
| `log10_kappa_r` | log10 of r-process opacity in cm²/g | log10(0.5) to log10(30) |

Mass, velocity and opacity are therefore **log-uniform**, rather than uniform
in their physical units. `MetzgerFullModel` defaults to `vmax=0.7c` and
`neutron_precursor=True`, as in the upstream constructor. The population's
maximum minimum velocity must be strictly smaller than `vmax`.
Changing these priors changes the colour envelope: choose them for your
scientific question and report them. Barnes thermalisation interpolation is
part of the upstream implementation; using parameter ranges outside its
calibration domain is not validated by this adapter.

The generator still records an isotropic `inclination_EM` for compatibility
with the existing catalogue/summary schema, but it does **not** affect these
spherical Metzger light curves. The Bu2026/AT2017gfo posterior reader is not
accepted for these models because it contains different physical parameters.

## Interactive generation

```bash
cd ~/repos/kne-lsst-analysis
python generate_kne_lightcurves.py
```

Select model 3 (`MetzgerModel`) or 4 (`MetzgerFullModel`). Choose the parameter
ranges and computation grid, then the familiar photometric system, OpSim
directory, distance, event count and storage options. For colour envelopes,
select the synthetic-only OpSim product and retain all finite model points;
apply depth cuts later with `build_color_envelopes.py`.

## Reproducible example

Start with a small run to inspect the curves:

```bash
python generate_kne_lightcurves.py \
  --model MetzgerModel \
  --run-name metzger_100_40p7Mpc \
  --parameter-source broad \
  --filter-system lsst+ps1 \
  --output-mode synthetic-mjd \
  --synthetic-depth-cut none \
  --synthetic-cadence-days 0.2 \
  --surrogate-batch-size 16 \
  --storage-format parquet \
  --n-samples 100 \
  --distance-mode fixed \
  --fixed-distance-mpc 40.7 \
  --galactic-latitude-mode all \
  --opsim-dir opsim_exports_all_survey \
  --plot-mode saved --max-plot-events 10 \
  --seed 12345 --sky-seed 67890
```

Replace `--model MetzgerModel` with `--model MetzgerFullModel` and use a
different run name for the other variant. `--no-metzger-neutron-precursor`
disables its precursor; `--metzger-vmax 0.8` changes its maximum shell speed.
Use `--filter-system lsst+ztf` for paired LSST/ZTF output or `ztf` for ZTF only.
ZTF depths remain a later envelope-builder selection, not a model parameter.

For example, changing the mass prior is done with:

```bash
--metzger-log10-mej-range -3 -1.3
```

The other controls are `--metzger-log10-vej-range`, `--metzger-beta-range`,
and `--metzger-log10-kappa-range` (two numbers per option).

## Time and numerical conventions

* The default **source-frame computation grid** has 1,000 logarithmic points
  from 0.1 to 30 days. FIESTA integrates the shell evolution on this grid.
* The adapter enables JAX double precision for Metzger. It leaves surrogate
  runs at their existing precision.
* FIESTA analytical predictions already include distance and spectral
  redshift effects, but return **source-frame times**. The generator converts
  those times exactly once: `t_observer = (1+z) * t_source`.
* `--synthetic-cadence-days 0.2` only resamples saved observer-frame data.
  It is independent of the integration grid and the later 0.4-day colour bins.
* The first integration point reflects the upstream initial conditions.
  Do not use it as an established early-time prediction without checking
  sensitivity to the integration start time.
* The Bu2026 numerical-floor mask is **not** applied to Metzger or Bu2019.
  The analytical implementation's own temperature clamps and blackbody
  assumptions remain in place; this adapter does not alter their physics.

Check numerical convergence separately from convergence in the number of
simulated events: repeat a small **identical** population with 2,000 source
points using `--analytical-n-times 2000`, and compare the relevant bands and
times. On one reference event (0.01 solar masses, v_min=0.1c, beta=3,
kappa=10 cm²/g, z=0.009), the maximum g-r difference from refining 1,000 to
2,000 points was approximately 0.017 mag for `MetzgerModel` and 0.010 mag
for `MetzgerFullModel` at 80 times between 0.4 and 16 source-frame days.
This is a spot check, not validation of the whole prior range.

## Outputs and pairing

Outputs remain `summary.csv`, optional `summary.parquet`, synthetic shards,
optional diagnostic PNGs, `run_info.txt` and `runtime_performance.json`.
Synthetic magnitudes contain foreground MW extinction, as in the existing
generator. The existing envelope builder constructs the corrected and
MW-present stages using the event extinction metadata.

The summary records the four Metzger parameters, model name and configuration.
`run_info.txt` also records the installed FIESTA version, implementation-file
SHA-256 and prediction-time convention. The source commit in the metadata is
the **audited reference**, not a claim that every user's installation matches
that commit.

Create a **new** shared intrinsic catalogue for each model with
`--pairing-mode create --intrinsic-catalog paired_metzger.csv`. Reuse it for
different sky cuts or distances with `--pairing-mode reuse`, the same
intrinsic seed and event count. Catalogues cannot cross model names; Metzger
catalogues must retain the same integration grid, constructor options and
population-prior bounds.
To change the distance while retaining the ejecta parameters use
`--distance-mode fixed --fixed-distance-mpc 600 --catalog-distance-policy current-fixed`.
The identical sky seed can retain sky pairing when the eligible fields and
other sky-generation settings are identical.

## Tests

```bash
python -m py_compile generate_kne_lightcurves.py kllap/generation/backends.py build_color_envelopes.py
PYTHONPATH=. python -m unittest discover -s tests -v
```

The default unit tests need no FIESTA weight or passband downloads. To check
actual predictions for both Metzger variants in all LSST/PS1/ZTF bands:

```bash
KNE_RUN_FIESTA_INTEGRATION_TESTS=1 PYTHONPATH=. \
  python -m unittest discover -s tests -p test_metzger_fiesta_integration.py -v
```

This optional check verifies finite predictions, scalar/batch agreement,
the distance-modulus shift, time conversion and spherical inclination
invariance. It does not replace validation of an end-to-end OpSim survey.

## Attribution

This update imports the model classes from FIESTA; it does not vendor or
redistribute NMMA code or `.joblib` files. Preserve the existing upstream
license notice and FIESTA scientific citation in `THIRD_PARTY_NOTICES.md`,
`licenses/FIESTA_LICENSE.txt` and `CITATION.cff`. When describing the analysis,
state the exact FIESTA class, version/commit, population priors and numerical
grid. The upstream class docstrings identify NMMA/Redback model lineage;
this does not establish agreement with every version of those packages.
