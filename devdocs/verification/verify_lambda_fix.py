#!/usr/bin/env python3
"""
verify_lambda_fix.py -- numerical re-verification of the --reference/--demo
lambda fix (DEV_LOG.md sections 21/23/25; src/main.cpp's run_*_mode()
functions). See DEV_LOG.md section 30 for the narrative.

This is a faithful Python port of:
  - include/utils/ImageIO.h's make_synthetic_test_image() (the --demo image)
  - include/utils/ImageIO.h's estimate_noise_sigma() (4-neighbor Laplacian
    + MAD-based robust sigma estimator)
  - include/utils/ImageIO.h's lambda_from_sigma() (lambda = 1.5 * sigma)

Purpose: an earlier development pass claimed "reproduces the old --demo
numbers to within ~0.1%" without a committed, re-runnable check. Running
this independently found that claim was specific to one seed/sigma
combination and does not hold as a general bound -- the real behavior
(documented below) is still good news for the historically-validated
default (sigma=25/255), but the deviation grows substantially at higher
noise levels due to [0,1] clipping. This script is what to re-run if the
demo image generator, the default noise-sigma, or the estimator ever
change, rather than re-deriving this by hand or trusting a stale number.

Run: python3 verify_lambda_fix.py
"""

import numpy as np


def make_synthetic_test_image(W=512, H=512):
    """Exact port of ImageIO.h's make_synthetic_test_image()."""
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    base = 0.5 + 0.3 * np.sin(jj * 0.05) * np.cos(ii * 0.04)
    bright_block = (ii > H // 3) & (ii < 2 * H // 3) & (jj > W // 3) & (jj < 2 * W // 3)
    dark_block = (ii > H // 10) & (ii < H // 4) & (jj > 2 * W // 3) & (jj < 9 * W // 10)
    base = np.where(bright_block, 0.85, base)
    base = np.where(dark_block, 0.15, base)
    return np.clip(base, 0, 1)


def estimate_noise_sigma_2d(data):
    """Exact port of ImageIO.h's estimate_noise_sigma(): 4-neighbor
    Laplacian, MAD-based robust sigma estimator. Stencil gain sqrt(20) is
    analytically exact for the stencil [1,1,1,1,-4] applied to i.i.d.
    Gaussian noise (sum of squared coefficients = 1+1+1+1+16 = 20)."""
    H, W = data.shape
    c = data.astype(np.float64)
    up = np.vstack([c[0:1, :], c[:-1, :]])
    down = np.vstack([c[1:, :], c[-1:, :]])
    left = np.hstack([c[:, 0:1], c[:, :-1]])
    right = np.hstack([c[:, 1:], c[:, -1:]])
    lap = up + down + left + right - 4.0 * c
    median_lap = np.median(lap)
    mad = np.median(np.abs(lap - median_lap))
    stencil_gain = np.sqrt(20.0)
    mad_to_std = 0.6745
    return max(0.0, mad / (mad_to_std * stencil_gain))


def lambda_from_sigma(sigma_normalized, k=1.5):
    return k * sigma_normalized


def main():
    clean = make_synthetic_test_image(512, 512)

    print(f"{'sigma_255':>10} {'mean%diff':>10} {'min%diff':>9} {'max%diff':>9} "
          f"{'frac_clipped':>13} {'estimator_vs_nominal':>22}")

    for noise_sigma_255 in [10.0, 25.0, 40.0, 60.0]:
        pct_diffs = []
        directions = []
        clip_fracs = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            raw_noise = rng.normal(0, noise_sigma_255 / 255.0, clean.shape)
            unclipped = clean + raw_noise
            clip_fracs.append(np.mean((unclipped < 0) | (unclipped > 1)))
            noisy = np.clip(unclipped, 0, 1)

            sigma_old = noise_sigma_255 / 255.0
            sigma_new = estimate_noise_sigma_2d(noisy)
            lambda_old = lambda_from_sigma(sigma_old)
            lambda_new = lambda_from_sigma(sigma_new)

            pct_diffs.append(100.0 * abs(lambda_new - lambda_old) / lambda_old)
            directions.append("lower" if sigma_new < sigma_old else "higher")

        pct_diffs = np.array(pct_diffs)
        mean_clip = np.mean(clip_fracs) * 100
        majority_dir = max(set(directions), key=directions.count)
        print(f"{noise_sigma_255:10.1f} {pct_diffs.mean():10.3f} {pct_diffs.min():9.3f} "
              f"{pct_diffs.max():9.3f} {mean_clip:12.3f}% {majority_dir:>22}")

    print()
    print("Interpretation:")
    print("  - At sigma=25/255 (the historically-validated default), the new")
    print("    estimate-from-data lambda differs from the old fixed-sigma")
    print("    lambda by well under 1% on average -- old --demo numbers at")
    print("    this setting remain a valid sanity check.")
    print("  - The gap grows with sigma because higher noise clips more pixels")
    print("    at the [0,1] boundary (clean image's own range leaves limited")
    print("    headroom before clipping kicks in), which suppresses the")
    print("    EFFECTIVE noise variance below the nominal injected value --")
    print("    a real, expected, well-understood nonlinearity, not a bug in")
    print("    either the noise injection or the estimator.")
    print("  - Do not treat '~0.1%' (an earlier, now-corrected internal claim)")
    print("    as a general bound across all noise-sigma settings.")


if __name__ == "__main__":
    main()
