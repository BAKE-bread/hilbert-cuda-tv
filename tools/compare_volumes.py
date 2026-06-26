#!/usr/bin/env python3
"""
compare_volumes.py -- independent comparison of two .rawvol volume files, 
for situations where HilbertCUDA-TV.exe's --reference mode is NOT the right measurement.

This script answers those questions directly: give it any two .rawvol files 
of matching shape, and it reports PSNR, SSIM, and basic statistics between 
them -- no noise injection, no assumptions about which one (if either) 
is "clean". It also supports batch comparison of many pairs via a CSV file.
"""

import argparse
import csv
import struct
import sys

import numpy as np

from hctv_metrics import psnr, ssim_windowed

RAWVOL_MAGIC = 0x564C4854


def load_rawvol(path):
    with open(path, "rb") as f:
        magic, width, height, depth = struct.unpack("<4I", f.read(16))
        if magic != RAWVOL_MAGIC:
            raise ValueError(f"{path}: bad magic number (not a valid .rawvol file)")
        data = np.fromfile(f, dtype=np.float32, count=width * height * depth)
    if data.size != width * height * depth:
        raise ValueError(f"{path}: truncated (expected {width*height*depth} voxels, got {data.size})")
    return data.reshape(depth, height, width)


def compare_pair(path_a, path_b, dynamic_range=None):
    """Returns a dict of comparison metrics between two .rawvol files.
    Raises ValueError if shapes don't match or files can't be loaded --
    caller decides whether to treat that as fatal (single-pair mode) or
    skip-and-continue (batch mode)."""
    a = load_rawvol(path_a)
    b = load_rawvol(path_b)

    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {path_a} is {a.shape}, {path_b} is {b.shape}")

    if dynamic_range is None:
        # Use the larger of the two files' own ranges -- works whether or
        # not either file happens to be normalized to exactly [0,1] (e.g.
        # comparing two un-normalized files directly is valid input to
        # this tool, unlike --reference mode which assumes [0,1]).
        dynamic_range = max(1.0, float(a.max()) - float(a.min()), float(b.max()) - float(b.min()))

    result = {
        "file_a": path_a,
        "file_b": path_b,
        "shape": a.shape,
        "a_mean": float(a.mean()),
        "a_std": float(a.std()),
        "a_min": float(a.min()),
        "a_max": float(a.max()),
        "b_mean": float(b.mean()),
        "b_std": float(b.std()),
        "b_min": float(b.min()),
        "b_max": float(b.max()),
        "psnr_db": psnr(a, b, peak=dynamic_range),
        "ssim": ssim_windowed(a, b, dynamic_range=dynamic_range),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "max_abs_diff": float(np.max(np.abs(a - b))),
    }
    return result


def print_single_result(r):
    print(f"=== Comparing {r['file_a']} vs {r['file_b']} ===")
    print(f"Shape: {r['shape']}")
    print(f"A - mean: {r['a_mean']:.6f}  std: {r['a_std']:.6f}  range: [{r['a_min']:.6f}, {r['a_max']:.6f}]")
    print(f"B - mean: {r['b_mean']:.6f}  std: {r['b_std']:.6f}  range: [{r['b_min']:.6f}, {r['b_max']:.6f}]")
    print(f"PSNR (A vs B): {r['psnr_db']:.2f} dB")
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

    fieldnames = ["file_a", "file_b", "shape", "a_mean", "a_std", "a_min", "a_max",
                  "b_mean", "b_std", "b_min", "b_max", "psnr_db", "ssim",
                  "mean_abs_diff", "max_abs_diff"]
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["shape"] = "x".join(str(s) for s in row["shape"])
            writer.writerow(row)

    print(f"\nWrote {len(results)} results to {output_csv_path}")
    psnrs = [r["psnr_db"] for r in results]
    print(f"PSNR summary: min={min(psnrs):.2f}  max={max(psnrs):.2f}  mean={np.mean(psnrs):.2f} dB")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two .rawvol files directly (PSNR/SSIM/stats), "
                     "independent of HilbertCUDA-TV.exe's --reference self-test mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--a", help="First .rawvol file")
    parser.add_argument("--b", help="Second .rawvol file")
    parser.add_argument("--batch", help="CSV file of path_a,path_b pairs for batch comparison")
    parser.add_argument("--output", help="Output CSV path (required with --batch)")
    args = parser.parse_args()

    if args.batch:
        if not args.output:
            parser.error("--output is required when using --batch")
        run_batch(args.batch, args.output)
    elif args.a and args.b:
        try:
            r = compare_pair(args.a, args.b)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(f"Error: {e}")
        print_single_result(r)
    else:
        parser.error("either provide both --a and --b, or use --batch with --output")


if __name__ == "__main__":
    main()
