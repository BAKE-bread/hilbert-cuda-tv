#!/usr/bin/env python3
"""
hctv_metrics.py -- shared PSNR/SSIM implementations used by every Python
tool in tools/ (visualize_volume.py, compare_volumes.py).

This module exists specifically so every tool computes these metrics the
SAME way -- having each script carry its own slightly-different copy of
"PSNR" or "SSIM" would silently reintroduce the exact "can't unify
measurement standards" problem this whole batch of fixes was meant to
solve, just one level up (Python tools disagreeing with each other,
instead of --input vs --reference disagreeing with each other).

The formulas here mirror include/utils/Metrics.h (the C++ implementation
used by HilbertCUDA-TV.exe itself) as closely as practical in Python:
  - psnr(): standard, peak-relative PSNR.
  - ssim_windowed(): non-overlapping-window SSIM (2D tiles or 3D cubes,
    matching the data's dimensionality), NOT the more common 11x11
    Gaussian-window SSIM -- chosen for the same reason Metrics.h chose it
    (fast, dependency-free, good enough as a comparison scalar). This
    means numbers from this script will read differently from e.g.
    scikit-image's SSIM; that's expected, not a bug -- don't directly
    compare numbers across tools without accounting for this, same caveat
    Metrics.h documents.
"""

import numpy as np


def psnr(a, b, peak=1.0):
    """Peak signal-to-noise ratio between two arrays of any matching
    shape (2D images or 3D volumes both work -- this just operates
    elementwise regardless of dimensionality)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"psnr: shape mismatch {a.shape} vs {b.shape}")
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-20:
        return 100.0
    return 10 * np.log10(peak * peak / mse)


def ssim_windowed(a, b, tile=8, dynamic_range=1.0):
    """Windowed SSIM using non-overlapping tiles (2D) or cubes (3D),
    dispatching on a.ndim. See module docstring for why this formulation
    (not the more common Gaussian-window SSIM) was chosen."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"ssim_windowed: shape mismatch {a.shape} vs {b.shape}")

    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2

    if a.ndim == 2:
        H, W = a.shape
        ranges = [(range(0, H, tile), range(0, W, tile))]
        def get_block(y, x):
            by, bx = min(tile, H - y), min(tile, W - x)
            return a[y:y + by, x:x + bx], b[y:y + by, x:x + bx]
        coords = [(y, x) for y in range(0, H, tile) for x in range(0, W, tile)]
    elif a.ndim == 3:
        D, H, W = a.shape
        def get_block(z, y, x):
            bz, by, bx = min(tile, D - z), min(tile, H - y), min(tile, W - x)
            return (a[z:z + bz, y:y + by, x:x + bx], b[z:z + bz, y:y + by, x:x + bx])
        coords = [(z, y, x) for z in range(0, D, tile) for y in range(0, H, tile) for x in range(0, W, tile)]
    else:
        raise ValueError(f"ssim_windowed: expected 2D or 3D array, got {a.ndim}D")

    total = 0.0
    count = 0
    for coord in coords:
        pa, pb = get_block(*coord)
        if pa.size < 2:
            continue
        ma, mb = pa.mean(), pb.mean()
        va, vb = pa.var(ddof=1), pb.var(ddof=1)
        cov = np.mean((pa - ma) * (pb - mb)) * pa.size / (pa.size - 1)
        num = (2 * ma * mb + C1) * (2 * cov + C2)
        den = (ma ** 2 + mb ** 2 + C1) * (va + vb + C2)
        total += num / den
        count += 1

    return total / count if count > 0 else 0.0


def estimate_noise_sigma(data):
    """Robust MAD-based noise sigma estimator, matching include/utils/
    ImageIO.h's estimate_noise_sigma (2D, 4-neighbor Laplacian) or
    include/utils/VolumeIO.h's estimate_noise_sigma_volume (3D, 6-
    neighbor Laplacian), dispatching on data.ndim. Returns sigma in the
    SAME units as the input array (i.e. if your data is in [0,1], you get
    a [0,1]-scale sigma; multiply by 255 yourself for the conventional
    "sigma=25"-style display, exactly as the C++ side does)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        H, W = data.shape
        lap = np.zeros_like(data)
        c = data
        up = np.vstack([c[0:1, :], c[:-1, :]])
        down = np.vstack([c[1:, :], c[-1:, :]])
        left = np.hstack([c[:, 0:1], c[:, :-1]])
        right = np.hstack([c[:, 1:], c[:, -1:]])
        lap = up + down + left + right - 4 * c
        stencil_gain = np.sqrt(20.0)
    elif data.ndim == 3:
        c = data
        up = np.concatenate([c[:, 0:1, :], c[:, :-1, :]], axis=1)
        down = np.concatenate([c[:, 1:, :], c[:, -1:, :]], axis=1)
        left = np.concatenate([c[:, :, 0:1], c[:, :, :-1]], axis=2)
        right = np.concatenate([c[:, :, 1:], c[:, :, -1:]], axis=2)
        back = np.concatenate([c[0:1, :, :], c[:-1, :, :]], axis=0)
        front = np.concatenate([c[1:, :, :], c[-1:, :, :]], axis=0)
        lap = up + down + left + right + back + front - 6 * c
        stencil_gain = np.sqrt(42.0)
    else:
        raise ValueError(f"estimate_noise_sigma: expected 2D or 3D array, got {data.ndim}D")

    median_lap = np.median(lap)
    mad = np.median(np.abs(lap - median_lap))
    mad_to_std = 0.6745
    return max(0.0, mad / (mad_to_std * stencil_gain))


if __name__ == "__main__":
    # Quick self-test when run directly (not a full test suite -- see
    # tools/test_hctv_metrics.py for that). Mirrors the sanity checks used
    # when these formulas were first validated in C++ (devdocs/DEV_LOG.md).
    import sys
    rng = np.random.default_rng(0)

    a2d = rng.uniform(0, 1, (32, 32))
    b2d = a2d.copy()
    print(f"2D identical: PSNR={psnr(a2d, b2d):.2f} (expect ~100) SSIM={ssim_windowed(a2d, b2d):.4f} (expect ~1.0)")

    a3d = rng.uniform(0, 1, (16, 16, 16))
    b3d = a3d.copy()
    print(f"3D identical: PSNR={psnr(a3d, b3d):.2f} (expect ~100) SSIM={ssim_windowed(a3d, b3d):.4f} (expect ~1.0)")

    # IMPORTANT: noise estimation needs spatially-correlated "clean" test
    # data, not pure uncorrelated random noise -- a uniform-random array's
    # OWN Laplacian is large (no spatial structure to exploit), which
    # swamps the injected noise signal and makes the estimator look wrong
    # when it is actually working correctly on realistic data (this exact
    # mistake was caught and documented in devdocs/DEV_LOG.md during
    # earlier estimator validation; repeating the fix here for the Python
    # side's self-test too).
    yy, xx = np.meshgrid(np.arange(32), np.arange(32))
    smooth2d = (0.5 + 0.3 * np.sin(xx * 0.2) * np.cos(yy * 0.15)).astype(np.float64)
    noisy2d = np.clip(smooth2d + rng.normal(0, 25 / 255, smooth2d.shape), 0, 1)
    est = estimate_noise_sigma(noisy2d)
    print(f"2D noise estimate: {est:.4f} (255-scale {est*255:.1f}, expect close to 25)")

    zz, yy3, xx3 = np.meshgrid(np.arange(16), np.arange(16), np.arange(16), indexing="ij")
    smooth3d = (0.5 + 0.3 * np.sin(xx3 * 0.3) * np.cos(yy3 * 0.25) * np.cos(zz * 0.2)).astype(np.float64)
    noisy3d = np.clip(smooth3d + rng.normal(0, 25 / 255, smooth3d.shape), 0, 1)
    est3d = estimate_noise_sigma(noisy3d)
    print(f"3D noise estimate: {est3d:.4f} (255-scale {est3d*255:.1f}, expect close to 25)")

    sys.exit(0)
