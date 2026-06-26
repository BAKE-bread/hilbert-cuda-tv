#!/usr/bin/env python3
"""
test_hctv_metrics.py -- real assertion-based test suite for hctv_metrics.py.

This replaces "run it and eyeball the printed numbers" with actual pass/
fail checks. Run with:

    cd tools/
    python -m pytest test_hctv_metrics.py -v

or, without pytest installed:

    python test_hctv_metrics.py

Covers:
  - psnr(): identical arrays, known-MSE cases, shape-mismatch errors, 2D/3D
  - ssim_windowed(): identical arrays (~1.0), shape-mismatch errors, 2D/3D,
    the "block too small to have variance" edge case
  - estimate_noise_sigma(): recovers a known injected sigma on spatially
    correlated synthetic data (2D and 3D) -- explicitly NOT using
    uncorrelated random data as the "clean" baseline, which is the exact
    trap documented in devdocs/DEV_LOG.md section 28 (uncorrelated noise
    has no spatial structure for the Laplacian-based estimator to exploit,
    so it looks "wrong" even though the estimator itself is fine).
"""

import sys
import unittest

import numpy as np

from hctv_metrics import psnr, ssim_windowed, estimate_noise_sigma


def make_smooth_2d(size=32, seed=0):
    """Spatially-correlated synthetic 'clean' test image -- see module
    docstring and DEV_LOG section 28 for why this matters instead of
    np.random.uniform(...)."""
    yy, xx = np.meshgrid(np.arange(size), np.arange(size))
    return (0.5 + 0.3 * np.sin(xx * 0.2) * np.cos(yy * 0.15)).astype(np.float64)


def make_smooth_3d(size=16, seed=0):
    zz, yy, xx = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij")
    return (0.5 + 0.3 * np.sin(xx * 0.3) * np.cos(yy * 0.25) * np.cos(zz * 0.2)).astype(np.float64)


class TestPSNR(unittest.TestCase):
    def test_identical_2d_is_ceiling(self):
        a = make_smooth_2d()
        self.assertEqual(psnr(a, a.copy()), 100.0)

    def test_identical_3d_is_ceiling(self):
        a = make_smooth_3d()
        self.assertEqual(psnr(a, a.copy()), 100.0)

    def test_known_constant_offset(self):
        # a vs a+delta everywhere -> mse = delta^2 exactly -> closed-form PSNR
        a = make_smooth_2d()
        delta = 0.1
        b = a + delta
        expected = 10 * np.log10(1.0 / (delta ** 2))
        self.assertAlmostEqual(psnr(a, b, peak=1.0), expected, places=6)

    def test_custom_peak_scales_correctly(self):
        a = make_smooth_2d() * 255.0
        delta = 5.0
        b = a + delta
        expected = 10 * np.log10((255.0 ** 2) / (delta ** 2))
        self.assertAlmostEqual(psnr(a, b, peak=255.0), expected, places=4)

    def test_shape_mismatch_raises(self):
        a = make_smooth_2d(32)
        b = make_smooth_2d(16)
        with self.assertRaises(ValueError):
            psnr(a, b)

    def test_higher_noise_gives_lower_psnr(self):
        a = make_smooth_2d()
        rng = np.random.default_rng(1)
        b_low_noise = a + rng.normal(0, 0.01, a.shape)
        b_high_noise = a + rng.normal(0, 0.10, a.shape)
        self.assertGreater(psnr(a, b_low_noise), psnr(a, b_high_noise))


class TestSSIM(unittest.TestCase):
    def test_identical_2d_is_near_one(self):
        a = make_smooth_2d()
        self.assertAlmostEqual(ssim_windowed(a, a.copy()), 1.0, places=6)

    def test_identical_3d_is_near_one(self):
        a = make_smooth_3d()
        self.assertAlmostEqual(ssim_windowed(a, a.copy()), 1.0, places=6)

    def test_shape_mismatch_raises(self):
        a = make_smooth_2d(32)
        b = make_smooth_2d(16)
        with self.assertRaises(ValueError):
            ssim_windowed(a, b)

    def test_unsupported_ndim_raises(self):
        a = np.zeros((4, 4, 4, 4))
        b = np.zeros((4, 4, 4, 4))
        with self.assertRaises(ValueError):
            ssim_windowed(a, b)

    def test_more_noise_gives_lower_ssim(self):
        a = make_smooth_2d()
        rng = np.random.default_rng(2)
        b_low_noise = a + rng.normal(0, 0.01, a.shape)
        b_high_noise = a + rng.normal(0, 0.10, a.shape)
        self.assertGreater(ssim_windowed(a, b_low_noise), ssim_windowed(a, b_high_noise))

    def test_tiny_array_smaller_than_one_tile_does_not_crash(self):
        # size 3 with default tile=8 -> a single partial block, pa.size < 2
        # guard must kick in cleanly rather than dividing by zero.
        a = np.array([[0.1, 0.2], [0.3, 0.4]])
        b = a.copy()
        # 2x2 has size 4 >= 2 so this one should actually compute; the real
        # edge case is a 1x1 array (size 1 < 2) which must NOT crash:
        a1 = np.array([[0.5]])
        b1 = np.array([[0.5]])
        result = ssim_windowed(a1, b1)
        self.assertEqual(result, 0.0)  # count stays 0 -> documented fallback
        # sanity: the normal 2x2 case still works and is high since identical
        self.assertAlmostEqual(ssim_windowed(a, b), 1.0, places=6)


class TestEstimateNoiseSigma(unittest.TestCase):
    def test_2d_recovers_known_sigma_on_correlated_data(self):
        clean = make_smooth_2d(size=64)
        rng = np.random.default_rng(42)
        true_sigma = 25.0 / 255.0
        noisy = np.clip(clean + rng.normal(0, true_sigma, clean.shape), 0, 1)
        est = estimate_noise_sigma(noisy)
        # Generous tolerance -- this is a statistical estimator, not exact.
        self.assertAlmostEqual(est, true_sigma, delta=true_sigma * 0.5)

    def test_3d_recovers_known_sigma_on_correlated_data(self):
        clean = make_smooth_3d(size=24)
        rng = np.random.default_rng(43)
        true_sigma = 25.0 / 255.0
        noisy = np.clip(clean + rng.normal(0, true_sigma, clean.shape), 0, 1)
        est = estimate_noise_sigma(noisy)
        self.assertAlmostEqual(est, true_sigma, delta=true_sigma * 0.5)

    def test_zero_noise_gives_small_estimate(self):
        # NOTE: not exactly zero -- a sinusoidal "clean" image still has
        # nonzero curvature (second derivative), so the Laplacian-based
        # estimator picks up a small residual even with no injected noise.
        # The bar here is "much smaller than a real injected sigma", not
        # "exactly zero" -- that distinction is the whole point of using
        # a MAD-based robust estimator instead of raw Laplacian energy.
        clean = make_smooth_2d(size=32)
        est = estimate_noise_sigma(clean)
        true_sigma = 25.0 / 255.0
        self.assertLess(est, true_sigma * 0.5)

    def test_uncorrelated_random_data_is_not_a_valid_clean_baseline(self):
        # Documents the DEV_LOG section 28 trap rather than silently
        # avoiding it: uncorrelated noise as "clean" data makes the
        # estimator read much higher than the injected sigma, because its
        # own high-frequency content swamps the signal. This test asserts
        # that the failure mode is real and reproducible, as a guard
        # against accidentally "fixing" the estimator to hide it (which
        # would actually be breaking it for real, structured data).
        rng = np.random.default_rng(7)
        uncorrelated = rng.uniform(0, 1, (32, 32))
        true_sigma = 25.0 / 255.0
        noisy = np.clip(uncorrelated + rng.normal(0, true_sigma, uncorrelated.shape), 0, 1)
        est = estimate_noise_sigma(noisy)
        # Expect a substantial overestimate (this is the documented trap,
        # not a bug to "fix" here) -- use a loose bound that just confirms
        # the gap is large, not a precise regression pin.
        self.assertGreater(est, true_sigma * 1.5)

    def test_unsupported_ndim_raises(self):
        with self.assertRaises(ValueError):
            estimate_noise_sigma(np.zeros((4, 4, 4, 4)))


if __name__ == "__main__":
    unittest.main()
