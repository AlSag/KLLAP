from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

from plot_kne_colour_parameter_space import (
    build_parser, color_selection, highest_density_regions,
    highest_density_regions_1d, load_event_colours, main, manual_bounds,
    parameter_values, parse_parameter_pairs, parse_times, read_bounds,
    resolve_bounds_path, percentile_pair_and_label,
)


class ColourParameterSpaceTest(unittest.TestCase):
    def test_times_must_be_positive_and_unique(self):
        np.testing.assert_allclose(parse_times("1,3,6,10"), [1., 3., 6., 10.])
        for invalid in ("0,1", "nan", "1,1"):
            with self.assertRaises(ValueError):
                parse_times(invalid)

    def test_manual_intervals_can_be_shared_or_time_dependent(self):
        times = np.array([1., 3.])
        shared = manual_bounds("-.2:.5", times)
        np.testing.assert_allclose(shared.color_min, [-.2, -.2])
        distinct = manual_bounds("-.2:.5,.3:1.1", times)
        np.testing.assert_allclose(distinct.color_max, [.5, 1.1])
        with self.assertRaises(ValueError):
            manual_bounds("1:0", times)

    def test_projection_pairs_keep_x_then_y(self):
        self.assertEqual(parse_parameter_pairs("Ye_wind:log10_mej_dyn,Ye_dyn:log10_mej_wind"), [("Ye_wind", "log10_mej_dyn"), ("Ye_dyn", "log10_mej_wind")])

    def test_colour_cut_is_inclusive_and_requires_availability(self):
        values = np.array([-.2, .5, .2, .8, np.nan])
        available = np.array([True, True, False, True, True])
        np.testing.assert_array_equal(color_selection(values, available, -.2, .5), [True, True, False, False, False])

    def test_parameter_transforms(self):
        metadata = pd.DataFrame({"log10_mej_dyn": [-3., -2.], "inclination_EM": [0., np.pi / 2]})
        np.testing.assert_allclose(parameter_values(metadata, "mej_dyn"), [.001, .01])
        np.testing.assert_allclose(parameter_values(metadata, "inclination_deg"), [0., 90.])

    def test_regions_calibrate_population_coverage_not_grid_height_percentiles(self):
        rng = np.random.default_rng(21)
        x, y = rng.normal(size=(2, 20000))
        _, _, probability, regions = highest_density_regions(x, y, (-6., 6.), (-6., 6.), grid_size=100)
        self.assertAlmostEqual(probability.sum(), 1.)
        for region in regions:
            self.assertGreaterEqual(region["empirical_fraction_inside"], region["target_mass"])
            self.assertLess(region["empirical_fraction_inside"] - region["target_mass"], .001)
            self.assertLess(abs(region["smoothed_mass_inside"] - region["target_mass"]), .04)
        self.assertGreater(regions[0]["threshold_cell_probability"], regions[1]["threshold_cell_probability"])

    def test_multimodal_population_retains_disconnected_regions(self):
        rng = np.random.default_rng(41)
        x = np.r_[rng.normal(-2., .2, 5000), rng.normal(2., .2, 5000)]
        y = rng.normal(0., .2, len(x))
        _, _, probability, regions = highest_density_regions(x, y, (-4., 4.), (-2., 2.), grid_size=120, sigma=1.)
        _, n_regions = label(probability >= regions[1]["threshold_cell_probability"])
        self.assertEqual(n_regions, 2)

    def test_1d_regions_can_be_disconnected_and_cover_selected_events(self):
        rng = np.random.default_rng(42)
        values = np.r_[
            rng.normal(-2., .15, 5000), rng.normal(2., .15, 5000)
        ]
        _, probability, regions = highest_density_regions_1d(
            values, (-4., 4.), grid_size=160, sigma=1., masses=(.9,)
        )
        self.assertAlmostEqual(probability.sum(), 1.)
        self.assertEqual(len(regions[0]["intervals"]), 2)
        self.assertGreaterEqual(regions[0]["empirical_fraction_inside"], .9)

    def test_saved_rule_bins_are_half_open_and_no_extrapolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.csv"
            pd.DataFrame({"stage": ["m5_no_mw", "m5_no_mw"], "t_min_days": [.4, 1.], "t_max_days": [1., 2.], "color_min": [0., .5], "color_max": [1., 1.5]}).to_csv(path, index=False)
            bounds = read_bounds(path, np.array([1., 2., 3.]), "m5_no_mw")
        np.testing.assert_allclose(bounds.color_min[:2], [.5, .5])
        self.assertEqual(bounds.bounds_status.iloc[2], "missing_or_invalid")

    def test_fixed_profile_does_not_use_invalid_percentiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            pd.DataFrame({"repetition": [0, 0, 1, 1], "t_low_days": [.4] * 4, "t_high_days": [1.4] * 4, "percentile": [5., 95., 5., 95.], "value": [0., 1., 3., 4.], "valid": [True, False, True, True]}).to_csv(path, index=False)
            invalid = read_bounds(path, np.array([1.]), "m5_with_mw", repetition=0)
            valid = read_bounds(path, np.array([1.]), "m5_with_mw", repetition=1)
        self.assertEqual(invalid.bounds_status.iloc[0], "missing_or_invalid")
        self.assertEqual(valid.color_min.iloc[0], 3.)
        self.assertEqual(valid.color_max.iloc[0], 4.)

    def test_convergence_directory_and_named_envelope_family(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            profile = directory / "percentile_reference_profiles.csv"
            pd.DataFrame({
                "t_low_days": [.4, .4, .4],
                "t_high_days": [1.4, 1.4, 1.4],
                "percentile": [16., 50., 84.],
                "value": [-.2, .1, .7],
                "valid": [True, True, True],
            }).to_csv(profile, index=False)
            args = build_parser().parse_args([
                "--envelope-family", "p16-p84",
            ])
            resolved = resolve_bounds_path(directory)
            pair, label_name = percentile_pair_and_label(args, resolved)
            bounds = read_bounds(resolved, np.array([1.]), "m5_with_mw", pair)
        self.assertEqual(resolved, profile)
        self.assertEqual(pair, (16., 84.))
        self.assertEqual(label_name, "p16-p84")
        np.testing.assert_allclose(bounds[["color_min", "color_max"]], [[-.2, .7]])

    @staticmethod
    def make_population(root, n_events=120):
        (root / "csv").mkdir()
        metadata = pd.DataFrame({"event_id": np.arange(n_events), "Ye_wind": np.linspace(.2, .4, n_events), "log10_mej_dyn": -3 + np.mod(np.arange(n_events) * 31, n_events) / n_events, "ebv_mw": np.full(n_events, .1), "redshift": np.full(n_events, .1)})
        metadata.to_csv(root / "summary.csv", index=False)
        for index in range(n_events):
            rows = []
            for time in (.5, 1., 3., 6., 10., 12.):
                for band, magnitude in (("g", 20. + index / n_events), ("r", 20.)):
                    rows.append({"t_days": time, "band": band, "mag": magnitude})
            pd.DataFrame(rows).to_csv(root / "csv" / f"lightcurve_{index:04d}.csv", index=False)
        return metadata

    def test_mw_and_depth_selection_match_saved_magnitude_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.make_population(root, n_events=2)
            args = build_parser().parse_args(["--run-dir", str(root), "--stage", "m5_no_mw"])
            colour, supported, available = load_event_colours(args, root, metadata, ("g", "r"), np.array([1., 13.]), {"g": 20.3, "r": 21.})
            from kne_envelope_core import RUBIN_MW_R
            expected = np.array([0., .5]) - .1 * (RUBIN_MW_R["g"] - RUBIN_MW_R["r"])
            np.testing.assert_allclose(colour[:, 0], expected)
            np.testing.assert_array_equal(available[:, 0], [True, True])
            self.assertFalse(supported[:, 1].any())
            args.stage = "m5_with_mw"
            colour, _, available = load_event_colours(args, root, metadata, ("g", "r"), np.array([1.]), {"g": 20.3, "r": 21.})
            np.testing.assert_allclose(colour[:, 0], [0., .5])
            np.testing.assert_array_equal(available[:, 0], [True, False])

    def test_ps1_depth_selection_uses_lsst_not_ps1_magnitudes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.make_population(root, n_events=2)
            for index in range(2):
                path = root / "csv" / f"lightcurve_{index:04d}.csv"
                lsst = pd.read_csv(path)
                lsst["photometric_system"] = "lsst"
                ps1 = lsst.copy()
                ps1["photometric_system"] = "ps1"
                ps1["mag"] = np.where(ps1.band.eq("g"), 17., 16.)
                pd.concat([lsst, ps1], ignore_index=True).to_csv(path, index=False)
            args = build_parser().parse_args(["--run-dir", str(root), "--stage", "m5_with_mw", "--photometric-system", "ps1"])
            colour, _, available = load_event_colours(args, root, metadata, ("g", "r"), np.array([1.]), {"g": 20.3, "r": 21.})
            np.testing.assert_allclose(colour[:, 0], [1., 1.])
            # Both PS1 magnitudes pass 20.3, but LSST g of the second event fails.
            np.testing.assert_array_equal(available[:, 0], [True, False])

    def test_source_time_conversion_and_no_extrapolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.make_population(root, n_events=2)
            args = build_parser().parse_args(["--run-dir", str(root), "--stage", "no_m5_with_mw", "--time-frame", "source"])
            _, supported, _ = load_event_colours(args, root, metadata, ("g", "r"), np.array([1., 11.]), {})
            # t_source=11 maps to t_obs=12.1, outside saved support ending at 12.
            np.testing.assert_array_equal(supported[:, 0], [True, True])
            np.testing.assert_array_equal(supported[:, 1], [False, False])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "PyArrow not installed in this test environment")
    def test_parquet_shards_have_same_colours_as_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.make_population(root, n_events=3)
            frames = []
            for index in range(3):
                frame = pd.read_csv(root / "csv" / f"lightcurve_{index:04d}.csv")
                frame["event_id"] = str(index)
                frame["photometric_system"] = "lsst"
                frames.append(frame)
            (root / "parquet").mkdir()
            pd.concat(frames, ignore_index=True).to_parquet(root / "parquet" / "synthetic_part_0000.parquet", index=False)
            args = build_parser().parse_args(["--run-dir", str(root), "--stage", "m5_with_mw"])
            colour, supported, available = load_event_colours(args, root, metadata, ("g", "r"), np.array([1.]), {"g": 20.5, "r": 21.})
            np.testing.assert_allclose(colour[:, 0], [0., 1. / 3, 2. / 3])
            self.assertTrue(supported.all())
            np.testing.assert_array_equal(available[:, 0], [True, True, False])

    def test_end_to_end_csv_population_and_output_safeguard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_population(root)
            output = root / "output"
            command = ["--run-dir", str(root), "--stage", "no_m5_with_mw", "--color-ranges=0.2:0.7", "--output-dir", str(output), "--min-contour-events", "20"]
            main(command)
            self.assertTrue((output / "parameter_space_Ye_wind_vs_log10_mej_dyn_manual_HDR90.png").is_file())
            self.assertTrue((output / "parameter_distribution_1d_Ye_wind_manual_HDR90.png").is_file())
            self.assertTrue((output / "parameter_distribution_1d_log10_mej_dyn_manual_HDR90.png").is_file())
            self.assertTrue((output / "parameter_1d_regions.csv").is_file())
            selection = pd.read_csv(output / "selection_summary.csv")
            np.testing.assert_array_equal(selection.N_total, [120] * 4)
            self.assertTrue((selection.N_selected > 0).all())
            self.assertEqual(len(pd.read_csv(output / "contour_diagnostics.csv")), 4)
            self.assertTrue(
                np.allclose(
                    pd.read_csv(output / "contour_diagnostics.csv").target_mass,
                    .9,
                )
            )
            self.assertEqual(len(pd.read_csv(output / "event_colour_selection.csv.gz")), 120)
            with self.assertRaises(FileExistsError):
                main(command)


if __name__ == "__main__":
    unittest.main()
