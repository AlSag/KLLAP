# Reproducibility checklist

Before a production analysis:

1. record the Git commit of this repository;
2. create the environment from `environment.yml` and retain `conda list
   --explicit` or a fresh package snapshot;
3. record the FIESTA version plus the origin and checksum of `Bu2026_MLP` model
   assets;
4. retain the Rubin OpSim database version and the generated OpSim export
   manifest;
5. use explicit intrinsic, sky/noise, bootstrap, and plotting seeds;
6. run a small CSV+Parquet validation population before a large Parquet run;
7. run `python -m unittest discover -s tests -v`;
8. archive configuration files, `run_info.txt`, manifests, and scientific
   tables together with checksums.

The supplied environment snapshot reflects the successful analysis machine,
but operating-system libraries, accelerators, and model files may still affect
installation. Reproducibility therefore requires both the Python environment
and the external data/model manifest.
