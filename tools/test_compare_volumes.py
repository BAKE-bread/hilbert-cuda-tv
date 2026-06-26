#!/usr/bin/env python3
"""
test_compare_volumes.py -- unit tests for compare_volumes.py, focused on
the previously-untested --batch CSV parsing/error-handling path (see
devdocs/DEV_LOG.md section 29, item 1) plus the single-pair comparison
logic it builds on.

Run with:
    cd tools/
    python -m pytest test_compare_volumes.py -v
or:
    python test_compare_volumes.py
"""

import csv
import os
import struct
import tempfile
import unittest

import numpy as np

from compare_volumes import load_rawvol, compare_pair, run_batch, RAWVOL_MAGIC


def _save_rawvol(path, volume_dhw):
    depth, height, width = volume_dhw.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", RAWVOL_MAGIC, width, height, depth))
        volume_dhw.astype(np.float32).tofile(f)


def _smooth_volume(shape, seed):
    """Spatially-correlated synthetic data -- see DEV_LOG section 28 for
    why uncorrelated random data is the wrong choice for this kind of
    test fixture."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    base = 0.5 + 0.3 * np.sin(xx * 0.3) * np.cos(yy * 0.25) * np.cos(zz * 0.2)
    return np.clip(base, 0, 1).astype(np.float32)


class TestLoadRawvol(unittest.TestCase):
    def test_round_trip(self):
        vol = _smooth_volume((4, 6, 8), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v.rawvol")
            _save_rawvol(path, vol)
            loaded = load_rawvol(path)
            self.assertEqual(loaded.shape, vol.shape)
            np.testing.assert_allclose(loaded, vol)

    def test_bad_magic_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.rawvol")
            with open(path, "wb") as f:
                f.write(struct.pack("<4I", 0xDEADBEEF, 4, 4, 4))
                np.zeros(64, dtype=np.float32).tofile(f)
            with self.assertRaises(ValueError):
                load_rawvol(path)

    def test_truncated_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "truncated.rawvol")
            with open(path, "wb") as f:
                f.write(struct.pack("<4I", RAWVOL_MAGIC, 4, 4, 4))
                np.zeros(10, dtype=np.float32).tofile(f)  # expects 64, only 10
            with self.assertRaises(ValueError):
                load_rawvol(path)


class TestComparePair(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def test_identical_files_give_ceiling_metrics(self):
        vol = _smooth_volume((4, 8, 8), 1)
        pa, pb = self._path("a.rawvol"), self._path("b.rawvol")
        _save_rawvol(pa, vol)
        _save_rawvol(pb, vol.copy())
        r = compare_pair(pa, pb)
        self.assertEqual(r["psnr_db"], 100.0)
        self.assertAlmostEqual(r["ssim"], 1.0, places=6)
        self.assertEqual(r["mean_abs_diff"], 0.0)

    def test_shape_mismatch_raises_value_error(self):
        a = _smooth_volume((4, 8, 8), 2)
        b = _smooth_volume((4, 8, 10), 3)
        pa, pb = self._path("a.rawvol"), self._path("b.rawvol")
        _save_rawvol(pa, a)
        _save_rawvol(pb, b)
        with self.assertRaises(ValueError):
            compare_pair(pa, pb)

    def test_noisier_pair_has_lower_psnr(self):
        clean = _smooth_volume((4, 16, 16), 4)
        rng = np.random.default_rng(4)
        low_noise = np.clip(clean + rng.normal(0, 0.01, clean.shape), 0, 1).astype(np.float32)
        high_noise = np.clip(clean + rng.normal(0, 0.10, clean.shape), 0, 1).astype(np.float32)

        p_clean, p_low, p_high = (self._path(n) for n in ("clean.rawvol", "low.rawvol", "high.rawvol"))
        _save_rawvol(p_clean, clean)
        _save_rawvol(p_low, low_noise)
        _save_rawvol(p_high, high_noise)

        r_low = compare_pair(p_clean, p_low)
        r_high = compare_pair(p_clean, p_high)
        self.assertGreater(r_low["psnr_db"], r_high["psnr_db"])

    def test_missing_file_raises(self):
        pa = self._path("exists.rawvol")
        _save_rawvol(pa, _smooth_volume((2, 4, 4), 5))
        with self.assertRaises(FileNotFoundError):
            compare_pair(pa, self._path("does_not_exist.rawvol"))

    def test_explicit_dynamic_range_is_respected(self):
        # Same absolute error, different declared dynamic_range -> different
        # PSNR. Confirms the parameter actually changes the computation
        # rather than being silently ignored.
        a = np.full((2, 4, 4), 0.5, dtype=np.float32)
        b = np.full((2, 4, 4), 0.6, dtype=np.float32)
        pa, pb = self._path("a.rawvol"), self._path("b.rawvol")
        _save_rawvol(pa, a)
        _save_rawvol(pb, b)
        r_small_range = compare_pair(pa, pb, dynamic_range=1.0)
        r_large_range = compare_pair(pa, pb, dynamic_range=10.0)
        self.assertNotAlmostEqual(r_small_range["psnr_db"], r_large_range["psnr_db"])
        self.assertGreater(r_large_range["psnr_db"], r_small_range["psnr_db"])


class TestRunBatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        # Use a fixed cwd-independent absolute path scheme.
        self.clean = _smooth_volume((4, 8, 8), 10)
        rng = np.random.default_rng(10)
        self.noisy = np.clip(self.clean + rng.normal(0, 0.05, self.clean.shape), 0, 1).astype(np.float32)
        self.mismatched = _smooth_volume((4, 8, 10), 11)

        self.p_clean = self._path("clean.rawvol")
        self.p_noisy = self._path("noisy.rawvol")
        self.p_mismatched = self._path("mismatched.rawvol")
        _save_rawvol(self.p_clean, self.clean)
        _save_rawvol(self.p_noisy, self.noisy)
        _save_rawvol(self.p_mismatched, self.mismatched)

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

        self.assertTrue(os.path.exists(out_path))
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertIn("psnr_db", rows[0])

    def test_comment_and_blank_lines_are_skipped(self):
        csv_path = self._write_csv([
            "# this is a comment",
            "",
            f"{self.p_clean},{self.p_noisy}",
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_shape_mismatch_row_is_skipped_not_fatal(self):
        csv_path = self._write_csv([
            f"{self.p_clean},{self.p_mismatched}",   # bad: skip
            f"{self.p_clean},{self.p_noisy}",         # good: keep
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)  # must not raise
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)  # only the good pair survives

    def test_missing_file_row_is_skipped_not_fatal(self):
        csv_path = self._write_csv([
            f"{self.p_clean},{self._path('nope.rawvol')}",  # bad: skip
            f"{self.p_clean},{self.p_noisy}",                 # good: keep
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_malformed_single_column_row_is_skipped(self):
        csv_path = self._write_csv([
            "only_one_column_here",
            f"{self.p_clean},{self.p_noisy}",
        ])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_all_rows_failing_raises_systemexit(self):
        csv_path = self._write_csv([
            f"{self._path('nope1.rawvol')},{self._path('nope2.rawvol')}",
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

    def test_output_csv_round_trips_shape_as_string(self):
        csv_path = self._write_csv([f"{self.p_clean},{self.p_noisy}"])
        out_path = self._path("summary.csv")
        run_batch(csv_path, out_path)
        with open(out_path, newline="") as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row["shape"], "4x8x8")


if __name__ == "__main__":
    unittest.main()
