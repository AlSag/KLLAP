"""Optional real-FIESTA predictions; downloads sncosmo passbands on first use.

KNE_RUN_FIESTA_INTEGRATION_TESTS=1 PYTHONPATH=. python -m unittest discover \
    -s tests -p test_metzger_fiesta_integration.py -v
"""
import argparse
import os
import unittest

import numpy as np

from kne_model_backends import add_metzger_arguments, create_fiesta_model, saved_observer_times


@unittest.skipUnless(os.environ.get("KNE_RUN_FIESTA_INTEGRATION_TESTS") == "1",
                     "optional real-FIESTA test; set KNE_RUN_FIESTA_INTEGRATION_TESTS=1")
class MetzgerFiestaIntegrationTest(unittest.TestCase):
    def test_real_models_scalar_batch_distance_and_spherical_symmetry(self):
        parser = argparse.ArgumentParser()
        add_metzger_arguments(parser)
        filters = ["lsst" + b for b in "ugrizy"] + ["ps1::" + b for b in "grizy"] + ["ztf" + b for b in "gri"]
        params = dict(log10_mej=-2.0, log10_vej=-1.0, beta=3.0, log10_kappa_r=1.0,
                      redshift=0.009, luminosity_distance=40.7, inclination_EM=0.4)
        for name in ("MetzgerModel", "MetzgerFullModel"):
            with self.subTest(model=name):
                args = parser.parse_args([])
                args.model = name
                args.parameter_source = "broad"
                args.fiesta_surrogate_dir = None
                model, metadata = create_fiesta_model(args, filters)
                source_times, mags = model.predict(params)
                self.assertEqual(set(mags), set(filters))
                for values in mags.values():
                    self.assertTrue(np.isfinite(np.asarray(values)).all())
                np.testing.assert_allclose(saved_observer_times(source_times, params, name),
                                           np.asarray(source_times) * 1.009)
                batched = {k: np.array([v, v]) for k, v in params.items()}
                _, batch_mags = model.vpredict(batched)
                _, distant = model.predict({**params, "luminosity_distance": 81.4})
                _, rotated = model.predict({**params, "inclination_EM": 1.4})
                for band in filters:
                    np.testing.assert_allclose(np.asarray(batch_mags[band])[0], mags[band], rtol=1e-6, atol=1e-7)
                    np.testing.assert_allclose(np.asarray(distant[band]) - np.asarray(mags[band]), 5 * np.log10(2), atol=1e-7)
                    np.testing.assert_allclose(rotated[band], mags[band], atol=1e-7)
                self.assertTrue(metadata["jax_enable_x64"])


if __name__ == "__main__":
    unittest.main()
