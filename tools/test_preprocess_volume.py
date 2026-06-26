#!/usr/bin/env python3
"""
test_preprocess_volume.py -- unit tests for preprocess_volume.py's
normalization functions and .rawvol round-trip I/O.

Run with:
    cd tools/
    python -m pytest test_preprocess_volume.py -v
or:
    python test_preprocess_volume.py
"""

import os
import tempfile
import unittest

import numpy as np

from preprocess_volume import (
    normalize_window,
    normalize_percentile,
    normalize_minmax,
    save_rawvol,
    load_rawvol,
    to_depth_height_width,
    RAWVOL_MAGIC,
)


class TestNormalizeWindow(unittest.TestCase):
    def test_basic_linear_rescale(self):
        data = np.array([-100.0, 0.0, 400.0], dtype=np.float64)
        out = normalize_window(data, -100, 400)
        np.testing.assert_allclose(out, [0.0, 0.2, 1.0], atol=1e-6)

    def test_clips_below_window(self):
        data = np.array([-2000.0, -100.0, 400.0], dtype=np.float64)
        out = normalize_window(data, -100, 400)
        # -2000 is below target_min, must clip to 0.0, not go negative.
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)

    def test_clips_above_window(self):
        data = np.array([-100.0, 400.0, 5000.0], dtype=np.float64)
        out = normalize_window(data, -100, 400)
        self.assertAlmostEqual(float(out[2]), 1.0, places=6)

    def test_output_dtype_is_float32(self):
        data = np.array([0.0, 1.0], dtype=np.float64)
        out = normalize_window(data, 0, 1)
        self.assertEqual(out.dtype, np.float32)

    def test_invalid_range_raises(self):
        data = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            normalize_window(data, 400, -100)  # max <= min

    def test_equal_bounds_raises(self):
        data = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            normalize_window(data, 100, 100)

    def test_lung_window_realistic_values(self):
        # CT lung window: [-1000, 500]. Air (~-1000 HU) -> 0, soft tissue
        # (~0 HU) -> ~0.667, dense bone-ish (~500 HU) -> 1.0.
        data = np.array([-1000.0, 0.0, 500.0], dtype=np.float64)
        out = normalize_window(data, -1000, 500)
        np.testing.assert_allclose(out, [0.0, 1000 / 1500, 1.0], atol=1e-6)


class TestNormalizePercentile(unittest.TestCase):
    def test_robust_to_outliers_vs_minmax(self):
        # A handful of extreme outliers should compress plain minmax but
        # NOT compress percentile-based normalization nearly as much.
        rng = np.random.default_rng(0)
        core = rng.normal(100, 10, 1000)
        outliers = np.array([-5000.0, 5000.0])  # e.g. metal artifact / air pocket
        data = np.concatenate([core, outliers])

        mm = normalize_minmax(data)
        pct = normalize_percentile(data, 1, 99)

        # The bulk of the (non-outlier) data should occupy much more of
        # the [0,1] range under percentile clipping than under raw minmax.
        core_mm_span = mm[:1000].max() - mm[:1000].min()
        core_pct_span = pct[:1000].max() - pct[:1000].min()
        self.assertGreater(core_pct_span, core_mm_span)

    def test_invalid_percentile_order_raises(self):
        data = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            normalize_percentile(data, 99, 1)  # low > high

    def test_percentile_out_of_bounds_raises(self):
        data = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            normalize_percentile(data, -5, 99)
        with self.assertRaises(ValueError):
            normalize_percentile(data, 1, 105)

    def test_output_in_unit_range(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, 500)
        out = normalize_percentile(data, 2, 98)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class TestNormalizeMinmax(unittest.TestCase):
    def test_basic_rescale(self):
        data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        out = normalize_minmax(data)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)

    def test_constant_data_raises(self):
        data = np.full((5, 5), 42.0)
        with self.assertRaises(ValueError):
            normalize_minmax(data)

    def test_output_dtype_is_float32(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = normalize_minmax(data)
        self.assertEqual(out.dtype, np.float32)

    def test_negative_values_handled(self):
        data = np.array([-50.0, 0.0, 50.0], dtype=np.float64)
        out = normalize_minmax(data)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)


class TestToDepthHeightWidth(unittest.TestCase):
    def test_no_transpose_by_default(self):
        data = np.zeros((4, 8, 16))
        out = to_depth_height_width(data)
        self.assertEqual(out.shape, (4, 8, 16))

    def test_force_transpose_swaps_first_and_last_axis(self):
        data = np.random.default_rng(0).normal(size=(4, 8, 16))
        out = to_depth_height_width(data, force_transpose=True)
        self.assertEqual(out.shape, (16, 8, 4))
        np.testing.assert_allclose(out, np.transpose(data, (2, 1, 0)))

    def test_non_3d_input_exits(self):
        data = np.zeros((4, 8, 16, 2))  # 4D, e.g. multi-channel
        with self.assertRaises(SystemExit):
            to_depth_height_width(data)


class TestRawvolRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_shape_and_values(self):
        rng = np.random.default_rng(5)
        volume = rng.uniform(0, 1, (6, 10, 12)).astype(np.float32)  # (D, H, W)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.rawvol")
            save_rawvol(path, volume)
            loaded = load_rawvol(path)
            self.assertEqual(loaded.shape, volume.shape)
            np.testing.assert_allclose(loaded, volume, atol=1e-7)

    def test_bad_magic_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.rawvol")
            with open(path, "wb") as f:
                f.write((0xDEADBEEF).to_bytes(4, "little"))
                f.write((4).to_bytes(4, "little"))
                f.write((4).to_bytes(4, "little"))
                f.write((4).to_bytes(4, "little"))
            with self.assertRaises(ValueError):
                load_rawvol(path)

    def test_header_magic_is_correct(self):
        import struct
        volume = np.zeros((2, 3, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.rawvol")
            save_rawvol(path, volume)
            with open(path, "rb") as f:
                magic, width, height, depth = struct.unpack("<4I", f.read(16))
            self.assertEqual(magic, RAWVOL_MAGIC)
            self.assertEqual((depth, height, width), volume.shape)


if __name__ == "__main__":
    unittest.main()
