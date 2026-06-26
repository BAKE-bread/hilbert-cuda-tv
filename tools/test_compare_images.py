#!/usr/bin/env python3
"""
test_compare_images.py -- unit tests for compare_images.py (the 2D
gray/color image counterpart to compare_volumes.py). Mirrors
test_compare_volumes.py's structure and coverage philosophy. Requires
Pillow (the same dependency compare_images.py itself requires).

Run with:
    cd tools/
    python -m pytest test_compare_images.py -v
or:
    python test_compare_images.py
"""

import csv
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("SKIPPING test_compare_images.py: Pillow is not installed "
          "(pip install Pillow)", file=sys.stderr)
    sys.exit(0)

from compare_images import load_image, compare_pair, run_batch, save_diff_image


def _smooth_gray(size=64, seed=0):
    """Spatially-correlated synthetic grayscale image"""
    yy, xx = np.meshgrid(np.arange(size), np.arange(size))
    img = 0.5 + 0.3 * np.sin(xx * 0.1) * np.cos(yy * 0.08)
    return np.clip(img, 0, 1)


def _smooth_color(size=64, seed=0):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size))
    r = 0.5 + 0.3 * np.sin(xx * 0.1)
    g = 0.5 + 0.3 * np.cos(yy * 0.08)
    b = 0.5 + 0.2 * np.sin((xx + yy) * 0.05)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)


def _save_gray_png(path, arr01):
    Image.fromarray((arr01 * 255).astype(np.uint8), mode="L").save(path)


def _save_color_png(path, arr01):
    Image.fromarray((arr01 * 255).astype(np.uint8), mode="RGB").save(path)


class TestLoadImage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def test_grayscale_round_trip(self):
        arr = _smooth_gray()
        path = self._path("g.png")
        _save_gray_png(path, arr)
        loaded = load_image(path)
        self.assertEqual(loaded.ndim, 2)
        self.assertEqual(loaded.shape, arr.shape)
        # 8-bit quantization -> compare with quantization-aware tolerance
        np.testing.assert_allclose(loaded, arr, atol=1.0 / 255 + 1e-9)

    def test_color_round_trip(self):
        arr = _smooth_color()
        path = self._path("c.png")
        _save_color_png(path, arr)
        loaded = load_image(path)
        self.assertEqual(loaded.ndim, 3)
        self.assertEqual(loaded.shape, arr.shape)
        np.testing.assert_allclose(loaded, arr, atol=1.0 / 255 + 1e-9)

    def test_16bit_grayscale_is_not_misread_as_color(self):
        # Regression test: an earlier version of load_image() checked for
        # PIL mode "I" but real 16-bit PNGs decode to "I;16"/"I;16B"/etc,
        # which fell through to the RGB-conversion branch and silently
        # produced wrong (and wrongly-shaped) data with no error at all.
        rng = np.random.default_rng(0)
        arr16 = (rng.uniform(0, 1, (32, 32)) * 65535).astype(np.uint16)
        path = self._path("g16.png")
        Image.fromarray(arr16).save(path)  # PIL infers mode "I;16" from the uint16 dtype

        loaded = load_image(path)
        self.assertEqual(loaded.ndim, 2, "16-bit grayscale must load as 2D, not be "
                                          "misinterpreted as a 3-channel color image")
        expected = arr16.astype(np.float64) / 65535.0
        np.testing.assert_allclose(loaded, expected, atol=1e-9)

    def test_rgba_drops_alpha(self):
        arr = _smooth_color()
        rgba = np.concatenate([arr, np.full((*arr.shape[:2], 1), 0.5)], axis=-1)
        path = self._path("rgba.png")
        Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA").save(path)
        loaded = load_image(path)
        self.assertEqual(loaded.shape, arr.shape)  # alpha dropped -> still (H,W,3)

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_image(self._path("does_not_exist.png"))


class TestComparePair(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def test_identical_grayscale_gives_ceiling_metrics(self):
        arr = _smooth_gray()
        pa, pb = self._path("a.png"), self._path("b.png")
        _save_gray_png(pa, arr)
        _save_gray_png(pb, arr)
        r = compare_pair(pa, pb)
        self.assertFalse(r["is_color"])
        self.assertEqual(r["psnr_db"], 100.0)
        self.assertAlmostEqual(r["ssim"], 1.0, places=4)
        self.assertEqual(r["mean_abs_diff"], 0.0)

    def test_identical_color_gives_ceiling_metrics(self):
        arr = _smooth_color()
        pa, pb = self._path("a.png"), self._path("b.png")
        _save_color_png(pa, arr)
        _save_color_png(pb, arr)
        r = compare_pair(pa, pb)
        self.assertTrue(r["is_color"])
        self.assertEqual(r["psnr_db"], 100.0)
        self.assertAlmostEqual(r["ssim"], 1.0, places=4)
        self.assertEqual(len(r["ssim_per_channel"]), 3)

    def test_color_ssim_is_per_channel_average_not_volume_misread(self):
        # Regression-style test for the most important correctness
        # property of this module: a (H,W,3) color image must NOT be
        # passed directly into hctv_metrics.ssim_windowed(), which would
        # silently interpret it as a 3-slice (D=H,H=W,W=3) volume instead
        # of an image and produce a number with no error at all.
        clean = _smooth_color(size=32)
        rng = np.random.default_rng(5)
        noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1)
        pa, pb = self._path("a.png"), self._path("b.png")
        _save_color_png(pa, clean)
        _save_color_png(pb, noisy)
        r = compare_pair(pa, pb)

        # Compare against the SAME on-disk (8-bit quantized) arrays
        # compare_pair() itself reads, not the unquantized float originals
        # -- otherwise this test would fail on harmless PNG quantization
        # noise rather than on the actual property under test.
        a_loaded = load_image(pa)
        b_loaded = load_image(pb)

        from hctv_metrics import ssim_windowed
        manual_per_channel = [
            ssim_windowed(a_loaded[:, :, c], b_loaded[:, :, c]) for c in range(3)
        ]
        self.assertAlmostEqual(r["ssim"], float(np.mean(manual_per_channel)), places=6)
        # And confirm it's NOT the (meaningless, volume-misread) result of
        # calling ssim_windowed on the raw (H,W,3) arrays directly.
        misread_result = ssim_windowed(a_loaded, b_loaded)
        self.assertNotAlmostEqual(r["ssim"], misread_result, places=2)

    def test_shape_mismatch_raises(self):
        a = _smooth_gray(size=64)
        b = _smooth_gray(size=32)
        pa, pb = self._path("a.png"), self._path("b.png")
        _save_gray_png(pa, a)
        _save_gray_png(pb, b)
        with self.assertRaises(ValueError):
            compare_pair(pa, pb)

    def test_color_grayscale_mismatch_raises_with_clear_message(self):
        pa, pb = self._path("a.png"), self._path("b.png")
        _save_gray_png(pa, _smooth_gray())
        _save_color_png(pb, _smooth_color())
        with self.assertRaises(ValueError) as ctx:
            compare_pair(pa, pb)
        self.assertIn("color/grayscale mismatch", str(ctx.exception))

    def test_missing_file_raises(self):
        pa = self._path("exists.png")
        _save_gray_png(pa, _smooth_gray())
        with self.assertRaises(FileNotFoundError):
            compare_pair(pa, self._path("nope.png"))

    def test_noisier_pair_has_lower_psnr(self):
        clean = _smooth_gray(size=64)
        rng = np.random.default_rng(1)
        low_noise = np.clip(clean + rng.normal(0, 0.01, clean.shape), 0, 1)
        high_noise = np.clip(clean + rng.normal(0, 0.10, clean.shape), 0, 1)
        p_clean, p_low, p_high = (self._path(n) for n in ("c.png", "lo.png", "hi.png"))
        _save_gray_png(p_clean, clean)
        _save_gray_png(p_low, low_noise)
        _save_gray_png(p_high, high_noise)
        r_low = compare_pair(p_clean, p_low)
        r_high = compare_pair(p_clean, p_high)
        self.assertGreater(r_low["psnr_db"], r_high["psnr_db"])

    def test_explicit_dynamic_range_changes_result(self):
        a = np.full((16, 16), 0.5)
        b = np.full((16, 16), 0.6)
        pa, pb = self._path("a.png"), self._path("b.png")
        _save_gray_png(pa, a)
        _save_gray_png(pb, b)
        r_small = compare_pair(pa, pb, dynamic_range=1.0)
        r_large = compare_pair(pa, pb, dynamic_range=10.0)
        self.assertNotAlmostEqual(r_small["psnr_db"], r_large["psnr_db"])
        self.assertGreater(r_large["psnr_db"], r_small["psnr_db"])


class TestSaveDiffImage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def test_diff_image_is_white_for_identical_inputs(self):
        arr = _smooth_gray()
        out = self._path("diff.png")
        save_diff_image(arr, arr.copy(), out)
        result = np.asarray(Image.open(out))
        # Identical inputs -> diff is exactly 0 everywhere -> the
        # percentile-based vmax_abs falls back to the 1e-9 floor, and
        # t = 0/1e-9 = 0 everywhere -> pure white.
        np.testing.assert_array_equal(result, 255)

    def test_diff_image_colormap_breakpoints(self):
        # Construct a case where we know the exact normalized t values:
        # a 2-pixel image where pixel 0 has a>b (positive diff) and pixel
        # 1 has a<b (negative diff) by equal magnitude.
        a = np.array([[0.6, 0.4]])
        b = np.array([[0.4, 0.6]])
        out = self._path("diff2.png")
        save_diff_image(a, b, out)
        result = np.asarray(Image.open(out))
        # Pixel 0: diff=+0.2 -> at the 99th percentile of |diff| (which is
        # 0.2 here, since both pixels have |diff|=0.2) -> t=+1 -> pure blue.
        self.assertEqual(tuple(result[0, 0]), (0, 0, 255))
        # Pixel 1: diff=-0.2 -> t=-1 -> pure red.
        self.assertEqual(tuple(result[0, 1]), (255, 0, 0))

    def test_color_diff_image_averages_channels(self):
        a = _smooth_color()
        rng = np.random.default_rng(3)
        b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1)
        out = self._path("diff3.png")
        save_diff_image(a, b, out)  # must not raise on (H,W,3) input
        result = np.asarray(Image.open(out))
        self.assertEqual(result.shape, (*a.shape[:2], 3))


class TestRunBatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.clean = _smooth_gray(size=32)
        rng = np.random.default_rng(10)
        self.noisy = np.clip(self.clean + rng.normal(0, 0.05, self.clean.shape), 0, 1)
        self.mismatched = _smooth_gray(size=16)
        self.color = _smooth_color(size=32)

        self.p_clean = self._path("clean.png")
        self.p_noisy = self._path("noisy.png")
        self.p_mismatched = self._path("mismatched.png")
        self.p_color = self._path("color.png")
        _save_gray_png(self.p_clean, self.clean)
        _save_gray_png(self.p_noisy, self.noisy)
        _save_gray_png(self.p_mismatched, self.mismatched)
        _save_color_png(self.p_color, self.color)

    def _path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def _write_csv(self, rows, name="pairs.csv"):
        path = self._path(name)
        with open(path, "w", newline="") as f:
            for row in rows:
                f.write(row + "\n")
        return path

    def test_all_valid_pairs_produce_output_csv(self):
        csv_path = self._write_csv([
            f"{self.p_clean},{self.p_noisy}",
            f"{self.p_clean},{self.p_clean}",
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertIn("psnr_db", rows[0])
        self.assertIn("is_color", rows[0])

    def test_shape_mismatch_row_is_skipped_not_fatal(self):
        csv_path = self._write_csv([
            f"{self.p_clean},{self.p_mismatched}",
            f"{self.p_clean},{self.p_noisy}",
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_color_grayscale_mismatch_row_is_skipped_not_fatal(self):
        csv_path = self._write_csv([
            f"{self.p_clean},{self.p_color}",
            f"{self.p_clean},{self.p_noisy}",
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_missing_file_row_is_skipped_not_fatal(self):
        csv_path = self._write_csv([
            f"{self.p_clean},{self._path('nope.png')}",
            f"{self.p_clean},{self.p_noisy}",
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_comment_and_blank_lines_are_skipped(self):
        csv_path = self._write_csv([
            "# comment",
            "",
            f"{self.p_clean},{self.p_noisy}",
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_all_rows_failing_raises_systemexit(self):
        csv_path = self._write_csv([
            f"{self._path('nope1.png')},{self._path('nope2.png')}",
        ])
        out_path = self._path("summary.csv")
        with self.assertRaises(SystemExit):
            run_batch(csv_path, out_path)
        self.assertFalse(os.path.exists(out_path))

    def test_empty_csv_raises_systemexit(self):
        csv_path = self._write_csv([])
        out_path = self._path("summary.csv")
        with self.assertRaises(SystemExit):
            run_batch(csv_path, out_path)


if __name__ == "__main__":
    unittest.main()
