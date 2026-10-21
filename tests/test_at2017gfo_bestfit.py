import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_color_envelopes import (
    PS1_MW_R,
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

        extincted = at2017gfo_stage_colours(
            curve,
            "no_m5_with_mw",
            "r",
            "i",
            0.105,
            "dereddened",
            "ps1",
        )
        expected_shift = (PS1_MW_R["r"] - PS1_MW_R["i"]) * 0.105
        np.testing.assert_allclose(
            extincted["color"], curve["color_input"] + expected_shift
        )


if __name__ == "__main__":
    unittest.main()
