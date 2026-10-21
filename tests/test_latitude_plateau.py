import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kllap.latitude.plateau import (
    allsky_counts, assess_plateau, high_over_low, load_table, pair_counts,
)


class LowHighPlateauTest(unittest.TestCase):
    def test_losses_separate_m5_from_additional_mw(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            pd.DataFrame({
                "event_id": ["a", "b", "c", "d"],
                "abs_b_deg": [1, 2, 3, 4],
                "N_bins_baseline": [2, 2, 2, 0],
                "N_bins_m5_no_mw": [0, 1, 1, 0],
                "N_bins_m5_with_mw": [0, 0, 1, 0],
            }).to_csv(path, index=False)
            _, out, _ = load_table(path)
            np.testing.assert_array_equal(out.sum(axis=0), [3, 1, 1])

    def test_exact_boundary_belongs_to_high_and_bootstrap_preserves_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            pd.DataFrame({
                "event_id": list("abcdef"),
                "abs_b_deg": [1, 5, 9, 10, 15, 80],
                "N_bins_baseline": [2]*6,
                "N_bins_m5_no_mw": [1]*6,
                "N_bins_m5_with_mw": [0, 1, 1, 0, 1, 1],
            }).to_csv(path, index=False)
            counts, boot, _, _ = allsky_counts(path, np.array([5, 10, 15]), 100, np.random.default_rng(2))
            np.testing.assert_array_equal(counts[:, 0, 0], [1, 3, 4])
            np.testing.assert_array_equal(counts.sum(axis=1), [[6,0,2]]*3)
            np.testing.assert_array_equal(boot.sum(axis=2)[:, 0], boot.sum(axis=2)[:, 1])

    def test_zero_loss_reference_ratio_stays_undefined(self):
        q = high_over_low(np.array([[[2000,100,0], [2000,100,2]]]))
        self.assertTrue(np.isnan(q[0]))

    def test_apparent_early_flat_patch_does_not_imply_tail_plateau(self):
        args = SimpleNamespace(min_events=100, min_mw_losses=20,
                               minimum_plateau_points=3, minimum_plateau_span=10,
                               tolerance=.1, confidence=.95)
        cuts = np.array([5,10,15,20,25,30])
        q = np.array([.1,.1,.1,.08,.06,.04])
        counts = np.tile([[[2000,100,50],[2000,100,50]]], (6,1,1))
        selected, _ = assess_plateau(cuts,q,np.tile(q,(100,1)),counts,args)
        self.assertIsNone(selected)

    def test_plateau_requires_uncertainty_to_fit_tolerance(self):
        args = SimpleNamespace(min_events=100, min_mw_losses=20,
                               minimum_plateau_points=3, minimum_plateau_span=10,
                               tolerance=.1, confidence=.95)
        cuts = np.array([5,10,15,20,25,30])
        q = np.array([.5,.3,.2,.1,.1,.1])
        counts = np.tile([[[2000,100,50],[2000,100,50]]], (6,1,1))
        tight = np.tile(q,(100,1))
        selected, _ = assess_plateau(cuts,q,tight,counts,args)
        self.assertEqual(selected,3)
        broad = tight.copy(); broad[:20,-1] = .15
        selected, _ = assess_plateau(cuts,q,broad,counts,args)
        self.assertIsNone(selected)

    def test_sparse_tail_cannot_be_declared_a_plateau(self):
        args = SimpleNamespace(min_events=100, min_mw_losses=20,
                               minimum_plateau_points=3, minimum_plateau_span=10,
                               tolerance=.1, confidence=.95)
        cuts = np.array([10,20,30])
        q = np.array([.1,.1,.1])
        counts = np.tile([[[2000,100,50],[2000,100,50]]], (3,1,1))
        counts[-1,1,2] = 2
        selected, rows = assess_plateau(cuts,q,np.tile(q,(100,1)),counts,args)
        self.assertIsNone(selected)
        self.assertEqual(rows[0][0], "Pertes rares a 30 deg")

    def test_pair_bootstrap_invariant_to_event_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = []
            for cut in [5,10,15]:
                sides = []
                for side in [0,1]:
                    path = Path(directory) / f"{cut}_{side}.csv"
                    df = pd.DataFrame({
                        "event_id": list("abcd"), "abs_b_deg": [cut-1 if side == 0 else cut]*4,
                        "N_bins_baseline": [2]*4, "N_bins_m5_no_mw": [1]*4,
                        "N_bins_m5_with_mw": [0,1,1,1],
                    })
                    if side:
                        df = df.iloc[::-1]
                    df.to_csv(path,index=False); sides.append(str(path))
                pairs.append([str(cut),*sides])
            _, counts, boot, _, _ = pair_counts(pairs,100,np.random.default_rng(1),"paired")
            np.testing.assert_array_equal(counts[:,0], counts[:,1])
            np.testing.assert_array_equal(boot[:,:,0], boot[:,:,1])


if __name__ == "__main__":
    unittest.main()
