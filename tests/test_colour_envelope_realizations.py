from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plot_colour_envelope_realizations import summarize_nested_prefixes


class RealizationsTest(unittest.TestCase):
    def test_nested_prefixes_use_finite_values_only(self):
        colours = pd.DataFrame({"event_id":[0,1,2], "time_days":[1.,1.,1.],
                                "colour_mag":[4.,8.,np.nan]})
        result = summarize_nested_prefixes(colours,[1,3],[1.,2.])
        self.assertEqual(result.n_valid.tolist(), [1,0,2,0])
        np.testing.assert_allclose(result.p50, [4.,np.nan,6.,np.nan])
        self.assertEqual(result.envelope_valid.tolist(), [True,False,True,False])
