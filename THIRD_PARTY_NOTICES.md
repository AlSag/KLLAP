# Third-party notices

## FIESTA / fiestaEM

`generate_kne_lightcurves.py` was developed from the FIESTA example
`examples/generate_lightcurves.py` and substantially extended for OpSim sky
contexts, paired populations, MW extinction, storage, and analysis metadata.

- Project: `nuclear-multimessenger-astronomy/fiestaEM`
- Immutable source used for the provenance audit:
  <https://github.com/nuclear-multimessenger-astronomy/fiestaEM/blob/7bec6b39392a47f6e937f8f7a237b8742bc334ae/examples/generate_lightcurves.py>
- License: MIT
- Reference environment version: `fiestaEM 0.2.0`
- Scientific reference requested by the project: Koehn et al. (2025),
  *Efficient Bayesian analysis of kilonovae and gamma ray burst afterglows
  with FIESTA*, Astronomy & Astrophysics 704, A55,
  <https://doi.org/10.1051/0004-6361/202556626>, arXiv:2507.13807.

This repository does not redistribute FIESTA model weights. FIESTA and its
dependencies retain their own copyright and license terms.

### Surrogate assets

`Bu2026_MLP` and `Bu2019_MLP` are external FIESTA-compatible surrogate assets,
not source code authored in this repository. Their model weights and training
data are not redistributed here.

The exact origin, immutable revision, and checksums of the `Bu2026_MLP` assets
used for a production result must be stored with that result. The reference
development environment used a locally installed `Bu2026_MLP`; a package
version alone is not sufficient to identify those assets.

The optional `Bu2019_MLP` path uses a FIESTA surrogate distributed separately
from the Python package:

- Model repository:
  <https://huggingface.co/nuclear-multimessenger-astronomy/fiesta-surrogates/tree/main/KN/Bu2019_MLP>
- Model-repository license: MIT
- Training-data entry: `KN/Bu2019_raw_data.h5` in the official FIESTA training
  data repository

This repository does not redistribute the model weights. Users should record
the downloaded model revision and checksums alongside production runs.

## Rubin software and OpSim data

The reference environment uses `rubin_sim 2.6.1` and `rubin_scheduler 3.22.0`
for Rubin/LSST survey and photometric calculations. `rubin_sim` is distributed
under GPL-3.0 and is available at <https://github.com/lsst/rubin_sim>.

The Rubin packages and OpSim databases are runtime dependencies or external
inputs. They are not redistributed by this repository and retain their
upstream terms. Production results should record the exact OpSim database
filename, release or source URL, and checksum.

## NMMA ZTF depth distributions

The optional ZTF depth selection reads the external files
`lims_public_g.joblib`, `lims_public_r.joblib`, and `lims_i.joblib`
distributed with the Nuclear Multi-Messenger Astronomy (NMMA) project.

These files contain empirical ZTF limiting-magnitude distributions and are
used as external runtime inputs.

- Project: `nuclear-multimessenger-astronomy/nmma`
- Upstream repository:
  <https://github.com/nuclear-multimessenger-astronomy/nmma>
- Upstream repository license at the time of this audit: GPL-3.0-only
- Files used:
  `nmma/em/data/lims_public_g.joblib`,
  `nmma/em/data/lims_public_r.joblib`,
  and `nmma/em/data/lims_i.joblib`

The files are not redistributed by this repository. Users must provide their
own trusted copies at runtime. Their paths and SHA-256 checksums are recorded
in the generated analysis metadata.

Because joblib is pickle-based, files from untrusted sources must not be
loaded.

## FIESTA semi-analytical model lineage

The semi-analytical kilonova backends import `MetzgerModel` and
`MetzgerFullModel` from FIESTA. The implementation inspected and tested
for this integration is commit
`6ba093cc13cb768a089b07768e550a6a46b737ca`, in
`src/fiesta/models/analytical_models/kilonova_models.py`.

The upstream model documentation identifies NMMA and Redback lineage. This
repository imports the implementations through FIESTA. It does not directly
import, copy, vendor, or redistribute NMMA or Redback model source code.

If the external NMMA limiting-magnitude files are used, cite NMMA and its
companion paper (Pang et al. 2023, Nature Communications 14, 8352,
<https://doi.org/10.1038/s41467-023-43932-6>) in addition to recording the
input-file checksums.

## Scope of this notice

Python packages listed in `environment.yml` and `requirements-lock.txt` retain
their own licences. This notice highlights dependencies, adapted examples,
model assets, and external data that are scientifically or legally material to
this workflow; it is not a replacement for the licence metadata of every
transitive dependency.
