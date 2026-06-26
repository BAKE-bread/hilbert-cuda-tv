#!/usr/bin/env python3
"""
compare_images.py -- independent comparison of two 2D image files
(PNG/JPG/BMP/etc, grayscale or color), for situations where
HilbertCUDA-TV.exe's --reference mode is NOT the right measurement.

This is the 2D (gray/color image) counterpart to compare_volumes.py --
see that script's docstring for the full --reference-vs-direct-comparison
rationale; the short version:

--reference mode (available in ALL THREE of HilbertCUDA-TV.exe's modes --
gray, color, and volume) is a CONTROLLED SELF-TEST: it loads one file,
injects a KNOWN amount of synthetic noise, denoises, and measures how
well that specific injected noise was removed. It does NOT compare two
independently-existing files to each other. That's a different question:

  - "How close is my denoised result to an independently-produced
    ground-truth clean photo?" (no synthetic noise involved at all --
    e.g. comparing clean.png against denoised.png after running
    --input clean.png through the solver)
  - "How does my result compare to another tool's denoised output on the
    same input?"
  - "How different are these two parameter choices' outputs from each
    other?" (doesn't need any ground truth)
  - Batch-comparing many file pairs at once.

This script answers those questions directly: give it any two image
files of matching dimensions, and it reports PSNR, SSIM, and basic pixel
statistics between them -- no noise injection, no assumptions about
which one (if either) is "clean". Works on grayscale OR color images;
mismatched grayscale-vs-color pairs are rejected with a clear error
rather than silently broadcast/misread (see compare_pair()'s docstring).

Usage:
    # Compare a denoised result against independent ground truth
    python compare_images.py --a clean.png --b denoised.png

    # Save a visual diff heatmap alongside the printed numbers
    python compare_images.py --a clean.png --b denoised.png --diff-output diff.png

    # Compare two different denoising runs against each other
    python compare_images.py --a result_lambda_0.05.png --b result_lambda_0.15.png

    # Batch mode: compare many pairs at once, write a CSV summary
    python compare_images.py --batch pairs.csv --output summary.csv

pairs.csv format (batch mode): two columns, no header, one pair per line:
    clean_1.png,denoised_1.png
    clean_2.png,denoised_2.png
    ...
"""

import argparse
import csv
import os
import sys

import numpy as np

from hctv_metrics import psnr, ssim_windowed


def load_image(path):
    """Load an image file as a float64 array in [0,1].
    Grayscale images return shape (H,W). Color images return shape
    (H,W,3) -- standard interleaved RGB, which is how PNG/JPG/etc are
    stored on disk regardless of HilbertCUDA-TV.exe's internal PLANAR
    in-memory layout for ColorImage (see include/utils/ImageIO.h) --
    that planar layout is purely an in-process implementation detail for
    the solver's per-channel kernels, never written to disk, so there is
    nothing to convert here. Any alpha channel is dropped (RGBA -> RGB),
    matching ImageIO.h's load_color(), which also forces 3 channels.

    NOTE: HilbertCUDA-TV.exe itself only ever reads/writes 8-bit PNGs
    (include/utils/ImageIO.h uses stbi_load/stbi_write_png, never the
    16-bit stbi_load_16 variant) -- so files THIS project produces are
    always plain 8-bit "L" or RGB. The 16-bit handling below exists only
    for the case where the user points this tool at some OTHER 16-bit
    PNG (e.g. a scientific camera capture) for comparison; it is not
    something HilbertCUDA-TV.exe's own outputs ever require."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Error: reading image files requires the 'Pillow' package.\n"
                  "Install it with: pip install Pillow")
    try:
        img = Image.open(path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"{path}: failed to load image ({e})")

    mode = img.mode
    if mode == "L":
        return np.asarray(img, dtype=np.float64) / 255.0
    elif mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        # Genuine 16-bit-depth grayscale (e.g. 16-bit PNG) -- divide by
        # the format's true max (65535), the same convention as "L"
        # dividing by 255, NOT by this particular image's own observed
        # min/max (which would silently rescale contrast differently for
        # different files and make PSNR/SSIM numbers incomparable across
        # them).
        return np.asarray(img, dtype=np.float64) / 65535.0
    elif mode == "I":
        # Plain "I" is PIL's 32-bit signed int mode with no single fixed
        # bit-depth convention -- pick 8-bit or 16-bit based on the
        # decoded values actually present, which is the best available
        # signal for what the source really was.
        arr = np.asarray(img, dtype=np.float64)
        return arr / 255.0 if arr.max() <= 255 else arr / 65535.0
    elif mode == "F":
        # 32-bit float pixels, also no fixed convention. If values look
        # like an integer 0-255 range that got stored as float, treat it
        # that way; otherwise assume it's already roughly normalized.
        arr = np.asarray(img, dtype=np.float64)
        return arr / 255.0 if arr.max() > 1.5 else arr
    else:
        # Anything else (RGB, RGBA, P/palette, CMYK, etc) -> force
        # standard RGB, exactly mirroring ImageIO.h's load_color()
        # forcing 3 channels.
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.float64) / 255.0


def is_color(arr):
    return arr.ndim == 3


def compare_pair(path_a, path_b, dynamic_range=None):
    """Returns a dict of comparison metrics between two image files.
    Raises ValueError if dimensions/color-mode don't match or files
    can't be loaded -- caller decides whether to treat that as fatal
    (single-pair mode) or skip-and-continue (batch mode).

    Color images (H,W,3): PSNR is computed over all channels jointly
    (standard definition -- MSE pooled across R,G,B). SSIM is computed
    PER CHANNEL via hctv_metrics.ssim_windowed() (which only natively
    understands 2D images or true 3D volumes) and then averaged across
    the 3 channels -- NOT by passing the (H,W,3) array into
    ssim_windowed() directly, which would silently misinterpret a color
    image as a 3-slice volume (depth=H, height=W, width=3) and produce a
    meaningless number with no error. This per-channel-then-average
    approach is the same convention used by, e.g., scikit-image's
    multichannel SSIM."""
    a = load_image(path_a)
    b = load_image(path_b)

    if a.shape != b.shape:
        if is_color(a) != is_color(b):
            raise ValueError(
                f"color/grayscale mismatch: {path_a} is "
                f"{'color ' + str(a.shape) if is_color(a) else 'grayscale ' + str(a.shape)}, "
                f"{path_b} is "
                f"{'color ' + str(b.shape) if is_color(b) else 'grayscale ' + str(b.shape)} "
                f"-- convert both to the same mode before comparing")
        raise ValueError(f"shape mismatch: {path_a} is {a.shape}, {path_b} is {b.shape}")

    if dynamic_range is None:
        # Use the larger of the two files' own ranges -- works whether or
        # not either file happens to be normalized to exactly [0,1].
        dynamic_range = max(1.0, float(a.max()) - float(a.min()), float(b.max()) - float(b.min()))

    if is_color(a):
        ssim_per_channel = [
            ssim_windowed(a[:, :, c], b[:, :, c], dynamic_range=dynamic_range)
            for c in range(3)
        ]
        ssim_value = float(np.mean(ssim_per_channel))
    else:
        ssim_per_channel = None
        ssim_value = ssim_windowed(a, b, dynamic_range=dynamic_range)

    result = {
        "file_a": path_a,
        "file_b": path_b,
        "shape": a.shape,
        "is_color": is_color(a),
        "a_mean": float(a.mean()),
        "a_std": float(a.std()),
        "a_min": float(a.min()),
        "a_max": float(a.max()),
        "b_mean": float(b.mean()),
        "b_std": float(b.std()),
        "b_min": float(b.min()),
        "b_max": float(b.max()),
        "psnr_db": psnr(a, b, peak=dynamic_range),
        "ssim": ssim_value,
        "ssim_per_channel": ssim_per_channel,  # None for grayscale
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "_a_array": a,  # kept only for --diff-output; stripped before CSV/print
        "_b_array": b,
    }
    return result


def save_diff_image(a, b, output_path):
    """Save a visual diff heatmap (|a-b|, averaged across channels if
    color, mapped through a simple blue-white-red diverging colormap
    matching tools/visualize_volume.py's residual panel convention for
    consistency across the 2D and 3D tools) as a PNG. Does not require
    matplotlib -- built directly with Pillow/numpy so this script's only
    hard dependency stays Pillow, matching load_image()'s dependency."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Error: saving a diff image requires the 'Pillow' package.\n"
                  "Install it with: pip install Pillow")

    diff = a.astype(np.float64) - b.astype(np.float64)
    if diff.ndim == 3:
        diff = diff.mean(axis=2)  # average signed difference across channels

    vmax_abs = np.percentile(np.abs(diff), 99) or 1e-9
    t = np.clip(diff / vmax_abs, -1.0, 1.0)  # normalized to [-1, 1]

    # Simple diverging colormap, white at t=0: blue for t>0 (A brighter
    # than B), red for t<0 (B brighter than A) -- same sign convention as
    # visualize_volume.py's "Original - Denoised" residual panel. At
    # t=+1: (R,G,B)=(0,0,255) pure blue. At t=-1: (255,0,0) pure red.
    # At t=0: (255,255,255) white. Linear in |t| on each side of zero.
    red = np.where(t >= 0, 255 * (1 - t), 255)
    green = np.where(t >= 0, 255 * (1 - t), 255 * (1 + t))
    blue = np.where(t >= 0, 255, 255 * (1 + t))

    rgb = np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)

    Image.fromarray(rgb, mode="RGB").save(output_path)
    print(f"Saved diff heatmap: {output_path} (blue=A brighter, red=B brighter, "
          f"clipped at +/-{vmax_abs:.4f} = 99th percentile |diff|)")


def print_single_result(r):
    mode = "color" if r["is_color"] else "grayscale"
    print(f"=== Comparing {r['file_a']} vs {r['file_b']} ({mode}) ===")
    print(f"Shape: {r['shape']}")
    print(f"A - mean: {r['a_mean']:.6f}  std: {r['a_std']:.6f}  range: [{r['a_min']:.6f}, {r['a_max']:.6f}]")
    print(f"B - mean: {r['b_mean']:.6f}  std: {r['b_std']:.6f}  range: [{r['b_min']:.6f}, {r['b_max']:.6f}]")
    print(f"PSNR (A vs B): {r['psnr_db']:.2f} dB")
    if r["ssim_per_channel"] is not None:
        ch_str = ", ".join(f"{c}={v:.4f}" for c, v in zip("RGB", r["ssim_per_channel"]))
        print(f"SSIM (A vs B, windowed, mean over channels): {r['ssim']:.4f}  [{ch_str}]")
    else:
        print(f"SSIM (A vs B, windowed): {r['ssim']:.4f}")
    print(f"Mean absolute difference: {r['mean_abs_diff']:.6f}")
    print(f"Max absolute difference:  {r['max_abs_diff']:.6f}")
    print()
    print("Interpretation: higher PSNR/SSIM means the two files are more")
    print("similar. If A is independent ground truth and B is this tool's")
    print("denoised output, this IS a valid accuracy measurement (unlike")
    print("--reference mode's self-test numbers, which measure noise")
    print("removal relative to a SYNTHETICALLY noised version of A, not")
    print("genuine accuracy against unrelated ground truth).")


def run_batch(pairs_csv_path, output_csv_path):
    pairs = []
    with open(pairs_csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                print(f"WARNING: skipping malformed row (need 2 columns): {row}", file=sys.stderr)
                continue
            pairs.append((row[0].strip(), row[1].strip()))

    if not pairs:
        sys.exit(f"Error: no valid pairs found in {pairs_csv_path}")

    results = []
    for path_a, path_b in pairs:
        try:
            r = compare_pair(path_a, path_b)
            results.append(r)
            print(f"OK   {path_a} vs {path_b}: PSNR={r['psnr_db']:.2f} dB, SSIM={r['ssim']:.4f}")
        except (ValueError, FileNotFoundError) as e:
            print(f"SKIP {path_a} vs {path_b}: {e}", file=sys.stderr)

    if not results:
        sys.exit("Error: every pair failed -- nothing to write.")

    fieldnames = ["file_a", "file_b", "shape", "is_color", "a_mean", "a_std", "a_min", "a_max",
                  "b_mean", "b_std", "b_min", "b_max", "psnr_db", "ssim",
                  "mean_abs_diff", "max_abs_diff"]
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k in fieldnames}
            row["shape"] = "x".join(str(s) for s in row["shape"])
            writer.writerow(row)

    print(f"\nWrote {len(results)} results to {output_csv_path}")
    psnrs = [r["psnr_db"] for r in results]
    print(f"PSNR summary: min={min(psnrs):.2f}  max={max(psnrs):.2f}  mean={np.mean(psnrs):.2f} dB")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two 2D image files directly (PSNR/SSIM/stats), "
                     "independent of HilbertCUDA-TV.exe's --reference self-test mode. "
                     "This is the gray/color image counterpart to compare_volumes.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--a", help="First image file (PNG/JPG/BMP/etc)")
    parser.add_argument("--b", help="Second image file")
    parser.add_argument("--diff-output", help="Optional: save a visual diff heatmap PNG (single-pair mode only)")
    parser.add_argument("--batch", help="CSV file of path_a,path_b pairs for batch comparison")
    parser.add_argument("--output", help="Output CSV path (required with --batch)")
    args = parser.parse_args()

    if args.batch:
        if not args.output:
            parser.error("--output is required when using --batch")
        if args.diff_output:
            parser.error("--diff-output is only supported in single-pair mode (--a/--b), not --batch")
        run_batch(args.batch, args.output)
    elif args.a and args.b:
        try:
            r = compare_pair(args.a, args.b)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(f"Error: {e}")
        print_single_result(r)
        if args.diff_output:
            save_diff_image(r["_a_array"], r["_b_array"], args.diff_output)
    else:
        parser.error("either provide both --a and --b, or use --batch with --output")


if __name__ == "__main__":
    main()