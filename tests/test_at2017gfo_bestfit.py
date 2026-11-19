import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_color_envelopes import (
    at2017gfo_stage_colours,
    load_at2017gfo_bestfit_colour_curve,
)


class AT2017gfoBestFitTest(unittest.TestCase):
    def test_saved_light_curves_become_colour_curve(self):
        payload = {
            "bestfit_params": {"luminosity_distance": np.array(40.7)},
            "light_curves": {
                "times": np.array([0.4, 0.8, 1.2]),
                "ps1::r": np.array([18.0, 18.4, 18.9]),
                "ps1::i": np.array([18.2, 18.5, 18.7]),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bestfit_params.pkl"
            with path.open("wb") as stream:
                pickle.dump(payload, stream)
            curve, parameters = load_at2017gfo_bestfit_colour_curve(
                path, "r", "i"
            )

        np.testing.assert_allclose(curve["color_input"], [-0.2, -0.1, 0.2])
        self.assertEqual(curve["source_filter_band1"].iloc[0], "ps1::r")
        self.assertEqual(parameters["luminosity_distance"], 40.7)

        for stage in ("no_m5", "m5"):
            actual = at2017gfo_stage_colours(curve, stage, "ps1")
            np.testing.assert_allclose(actual["color"], curve["color_input"])
