"""Saved Metzger populations must not be dropped for lacking Bulla parameters."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_color_envelopes import (
    build_parser, canonical_event_id, identify_permanent_exits, prepare_metadata,
)


class MetzgerEnvelopeInputTest(unittest.TestCase):
    def test_metzger_metadata_and_exit_parameters_are_retained(self):
        row = dict(event_id="000000", model="MetzgerFullModel", ebv_mw=0.1,
                   inclination_EM=0.5, luminosity_distance=40.7, redshift=0.009,
                   log10_mej=-2, log10_vej=-1, beta=3, log10_kappa_r=1,
                   log10_mej_dyn=np.nan, log10_mej_wind=np.nan)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([row]).to_csv(root / "summary.csv", index=False)
            args = build_parser().parse_args(["--run-dir", str(root), "--photometric-system", "lsst"])
            event_id = canonical_event_id("000000")
            meta = prepare_metadata(args, {event_id})
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta["log10_mej"].iloc[0], -2)
        rules = pd.DataFrame(dict(bin_index=[0, 1, 2], t_center_days=[0.6, 1, 1.4],
                                  color_min=[0, 0, 0], color_max=[1, 1, 1]))
        exits = identify_permanent_exits(np.array([[0.5, 1.5, 2.0]]), rules, meta, {}, 2)
        self.assertEqual(len(exits), 1)
        self.assertAlmostEqual(exits["M_ej_Msun"].iloc[0], 0.01)
        self.assertAlmostEqual(exits["v_min_c"].iloc[0], 0.1)
        self.assertAlmostEqual(exits["kappa_r_cm2_g"].iloc[0], 10)
        self.assertFalse(exits["inclination_used_by_model"].iloc[0])

    def test_required_foreground_metadata_is_still_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([dict(event_id=0, model="MetzgerModel")]).to_csv(root / "summary.csv", index=False)
            args = build_parser().parse_args(["--run-dir", str(root)])
            with self.assertRaisesRegex(ValueError, "ebv_mw"):
                prepare_metadata(args, {canonical_event_id(0)})


if __name__ == "__main__":
    unittest.main()
