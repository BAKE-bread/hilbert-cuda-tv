#!/usr/bin/env python3
"""
verify_metrics_math.py -- independent mathematical cross-check of
tools/hctv_metrics.py's psnr() and ssim_windowed() against textbook
formulas implemented from scratch here (NOT importing or copying any
code from hctv_metrics.py's own implementation), to catch any subtle bug
that might survive hctv_metrics.py's own self-consistency. Comparing a
function to itself proves nothing; comparing it to an independently
written reference does. See devdocs/DEV_LOG.md section 30.

Run from anywhere: python3 devdocs/verification/verify_metrics_math.py
"""
import os
import sys

_TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools")
sys.path.insert(0, _TOOLS_DIR)

import numpy as np
from hctv_metrics import psnr, ssim_windowed

# --- Independent textbook PSNR ---
def textbook_psnr(a, b, peak=1.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((peak ** 2) / mse)

# --- Independent textbook windowed SSIM (standard formula, no numpy tricks shared with the module) ---
def textbook_ssim_2d_tile(pa, pb, dynamic_range=1.0):
    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2
    mu_a, mu_b = pa.mean(), pb.mean()
    var_a = np.sum((pa - mu_a)**2) / (pa.size - 1)
    var_b = np.sum((pb - mu_b)**2) / (pb.size - 1)
    cov = np.sum((pa - mu_a)*(pb - mu_b)) / (pa.size - 1)
    numerator = (2*mu_a*mu_b + C1) * (2*cov + C2)
    denominator = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    return numerator / denominator

def textbook_ssim_windowed_2d(a, b, tile=8, dynamic_range=1.0):
    H, W = a.shape
    total, count = 0.0, 0
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            pa = a[y:y+tile, x:x+tile]
            pb = b[y:y+tile, x:x+tile]
            if pa.size < 2:
                continue
            total += textbook_ssim_2d_tile(pa, pb, dynamic_range)
            count += 1
    return total / count if count > 0 else 0.0

def main():
    rng = np.random.default_rng(99)
    results = []
    for trial in range(20):
        H, W = rng.integers(16, 64, size=2)
        H, W = int(H) - int(H) % 8 + 8, int(W) - int(W) % 8 + 8  # keep multiple-of-8-ish, not required but realistic
        a = rng.uniform(0, 1, (H, W))
        noise_level = rng.uniform(0.01, 0.3)
        b = np.clip(a + rng.normal(0, noise_level, (H, W)), 0, 1)

        psnr_module = psnr(a, b)
        psnr_textbook = textbook_psnr(a, b)
        psnr_diff = abs(psnr_module - psnr_textbook)

        ssim_module = ssim_windowed(a, b)
        ssim_textbook = textbook_ssim_windowed_2d(a, b)
        ssim_diff = abs(ssim_module - ssim_textbook)

        results.append((psnr_diff, ssim_diff))

    psnr_diffs = [r[0] for r in results]
    ssim_diffs = [r[1] for r in results]
    print(f"PSNR: max abs diff across 20 random trials = {max(psnr_diffs):.2e}")
    print(f"SSIM: max abs diff across 20 random trials = {max(ssim_diffs):.2e}")
    print()
    ok = max(psnr_diffs) < 1e-9 and max(ssim_diffs) < 1e-9
    print("PASS" if ok else "FAIL -- formulas disagree!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
