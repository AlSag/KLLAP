# Third-party notices

## FIESTA / fiestaEM

`generate_kne_lightcurves.py` was developed from the FIESTA example
`examples/generate_lightcurves.py` and substantially extended for OpSim sky
contexts, paired populations, MW extinction, storage, and analysis metadata.

- Project: `nuclear-multimessenger-astronomy/fiestaEM`
- Immutable source used for the provenance audit:
  <https://github.com/nuclear-multimessenger-astronomy/fiestaEM/blob/7bec6b39392a47f6e937f8f7a237b8742bc334ae/examples/generate_lightcurves.py>
- License: MIT
- Installed analysis version: `fiestaEM 0.2.0`
- Scientific reference requested by the project: Koehn et al. (2025),
  *FIESTA: Fast inference of electromagnetic signals and transients with
  auto-differentiable surrogate models*, Astronomy & Astrophysics 704, A55,
  arXiv:2507.13807.

This repository does not redistribute FIESTA model weights. FIESTA and its
dependencies retain their own copyright and license terms.

The optional `Bu2019_MLP` path uses a FIESTA surrogate distributed separately
from the Python package:

- Model repository:
  <https://huggingface.co/nuclear-multimessenger-astronomy/fiesta-surrogates/tree/main/KN/Bu2019_MLP>
- Model-repository license: MIT
- Training-data entry: `KN/Bu2019_raw_data.h5` in the official FIESTA training
  data repository

This repository does not redistribute the model weights. Users should record
the downloaded model revision and checksums alongside production runs.

## Rubin software

The workflow uses `rubin_sim` and `rubin_scheduler` for Rubin/LSST survey and
photometric calculations. Those packages and OpSim databases are not
redistributed by this repository and remain subject to their upstream terms.
