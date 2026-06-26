#!/usr/bin/env python3
"""
visualize_volume.py -- 3D slice-comparison visualization and statistical
analysis for HilbertCUDA-TV's volumetric (.rawvol) outputs.

Generalizes the comparison/plotting prototype from the project's nii.py
script into a reusable, command-line driven, English-language tool with a
few additional diagnostics (PSNR/SSIM when both volumes represent the
same underlying data, a configurable slice index, and a clean error path
when matplotlib's CJK font fallback isn't available rather than silently
rendering boxes for non-ASCII text).

Usage:
    # Compare an original (e.g. noisy or reference) volume against a
    # denoised result -- the most common use case.
    python visualize_volume.py --original heart_003_norm.rawvol --denoised result.rawvol

    # Specify which slice indices to show (default: middle of each axis)
    python visualize_volume.py --original a.rawvol --denoised b.rawvol --slice-z 40 --slice-y 100 --slice-x 150

    # Just look at one volume (no comparison)
    python visualize_volume.py --original scan.rawvol

Outputs (written to --output-dir, default: current directory):
    volume_slices.png      3x3 grid: original / denoised / residual, for
                            axial / coronal / sagittal mid-slices
    volume_histogram.png   intensity histogram comparison (step-line style,
                            so overlapping distributions stay readable)
    (stats are printed to stdout, not saved to a file, by design -- numbers
    you want to keep should go in your own lab notes / experiment log)
"""

import argparse
import struct
import sys

import numpy as np

from hctv_metrics import psnr, ssim_windowed as ssim_windowed_3d

RAWVOL_MAGIC = 0x564C4854


def load_rawvol(path):
    with open(path, "rb") as f:
        magic, width, height, depth = struct.unpack("<4I", f.read(16))
        if magic != RAWVOL_MAGIC:
            sys.exit(f"Error: {path} does not look like a valid .rawvol file "
                      f"(bad magic number).")
        data = np.fromfile(f, dtype=np.float32, count=width * height * depth)
    if data.size != width * height * depth:
        sys.exit(f"Error: {path} is truncated (expected {width*height*depth} "
                  f"voxels, got {data.size}).")
    return data.reshape(depth, height, width)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def setup_matplotlib():
    """Imports and configures matplotlib. Unlike the original prototype,
    this does NOT assume a CJK font is installed -- all labels in this
    script are plain ASCII English specifically so no font fallback is
    needed at all, avoiding the silent "boxes instead of text" failure
    mode entirely rather than trying to paper over it with a font list
    that may or may not be present on the user's system."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe; works whether or not a display is available
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Error: this script requires matplotlib.\nInstall it with: pip install matplotlib")
    return plt


def get_orthogonal_slices(volume, idx_z=None, idx_y=None, idx_x=None):
    """volume: shape (D,H,W). Returns {name: (2D slice, index used)}."""
    d, h, w = volume.shape
    idx_z = idx_z if idx_z is not None else d // 2
    idx_y = idx_y if idx_y is not None else h // 2
    idx_x = idx_x if idx_x is not None else w // 2
    for name, idx, size in [("z (axial)", idx_z, d), ("y (coronal)", idx_y, h), ("x (sagittal)", idx_x, w)]:
        if not (0 <= idx < size):
            sys.exit(f"Error: slice index {idx} for axis {name} is out of range [0, {size}).")
    return {
        "axial": (volume[idx_z, :, :], idx_z),
        "coronal": (volume[:, idx_y, :], idx_y),
        "sagittal": (volume[:, :, idx_x], idx_x),
    }


def plot_slice_comparison(plt, original, denoised, output_path, idx_z=None, idx_y=None, idx_x=None):
    orig_slices = get_orthogonal_slices(original, idx_z, idx_y, idx_x)
    residual = original - denoised
    denoised_slices = get_orthogonal_slices(denoised, idx_z, idx_y, idx_x)
    residual_slices = get_orthogonal_slices(residual, idx_z, idx_y, idx_x)

    # Use GridSpec with a dedicated 4th column for the colorbar to avoid
    # auto-placement overlapping the middle row's title.
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05],
                           left=0.03, right=0.92, top=0.95, bottom=0.03,
                           wspace=0.15, hspace=0.3)
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cax = fig.add_subplot(gs[:, 3])

    plane_names = ["axial", "coronal", "sagittal"]
    vmax_abs = np.percentile(np.abs(residual), 95)

    im = None
    for row, plane in enumerate(plane_names):
        orig_slice, idx = orig_slices[plane]
        denoised_slice, _ = denoised_slices[plane]
        residual_slice, _ = residual_slices[plane]

        axes[row, 0].imshow(orig_slice, cmap="gray")
        axes[row, 0].set_title(f"Original - {plane} (idx={idx})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(denoised_slice, cmap="gray")
        axes[row, 1].set_title(f"Denoised - {plane} (idx={idx})")
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(residual_slice, cmap="RdBu", vmin=-vmax_abs, vmax=vmax_abs)
        axes[row, 2].set_title(f"Residual - {plane} (idx={idx})")
        axes[row, 2].axis("off")

    fig.colorbar(im, cax=cax, label="Original - Denoised")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved slice comparison: {output_path}")


def plot_single_volume(plt, volume, output_path, idx_z=None, idx_y=None, idx_x=None):
    """Single-volume version (no denoised/residual columns) for when only
    --original is given."""
    slices = get_orthogonal_slices(volume, idx_z, idx_y, idx_x)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for col, plane in enumerate(["axial", "coronal", "sagittal"]):
        sl, idx = slices[plane]
        axes[col].imshow(sl, cmap="gray")
        axes[col].set_title(f"{plane} (idx={idx})")
        axes[col].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved volume slices: {output_path}")


def plot_histogram(plt, original, denoised, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(original.flatten(), bins=100, label="Original", density=True,
              histtype="step", linewidth=2, color="blue")
    plt.hist(denoised.flatten(), bins=100, label="Denoised", density=True,
              histtype="step", linewidth=2, color="orange")
    plt.xlabel("Voxel intensity")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Voxel intensity histogram comparison")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histogram: {output_path}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(original, denoised=None):
    print("=== Volume statistics ===")
    print(f"Original shape (D,H,W): {original.shape}")
    print(f"Original  - mean: {original.mean():.6f}  std: {original.std():.6f}  "
          f"range: [{original.min():.6f}, {original.max():.6f}]")

    if denoised is None:
        return

    if denoised.shape != original.shape:
        print(f"WARNING: shape mismatch -- original {original.shape} vs "
              f"denoised {denoised.shape}. Skipping comparison metrics.")
        return

    print(f"Denoised  - mean: {denoised.mean():.6f}  std: {denoised.std():.6f}  "
          f"range: [{denoised.min():.6f}, {denoised.max():.6f}]")

    std_reduction = (1 - denoised.std() / original.std()) * 100 if original.std() > 0 else float("nan")
    print(f"Std-dev reduction: {std_reduction:.2f}%")

    residual = original - denoised
    print(f"Residual (original - denoised) std: {residual.std():.6f}")

    p = psnr(original, denoised, peak=max(1.0, float(original.max())))
    s = ssim_windowed_3d(original, denoised, dynamic_range=max(1.0, float(original.max())))
    print(f"PSNR (original vs denoised): {p:.2f} dB")
    print(f"SSIM (original vs denoised, windowed): {s:.4f}")
    print()
    print("NOTE: if 'original' here is itself a noisy/reference volume (not")
    print("a ground-truth clean scan), these numbers describe how much the")
    print("denoiser changed the data, NOT how close the result is to ground")
    print("truth. For a true accuracy assessment against a known-clean")
    print("baseline, use tools/compare_volumes.py instead.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize and statistically compare HilbertCUDA-TV .rawvol volumes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--original", required=True, help="Path to the original/reference .rawvol file")
    parser.add_argument("--denoised", help="Path to the denoised .rawvol file (optional -- omit to just inspect --original alone)")
    parser.add_argument("--output-dir", default=".", help="Directory to save plots into (default: current directory)")
    parser.add_argument("--slice-z", type=int, help="Axial slice index (default: middle)")
    parser.add_argument("--slice-y", type=int, help="Coronal slice index (default: middle)")
    parser.add_argument("--slice-x", type=int, help="Sagittal slice index (default: middle)")
    parser.add_argument("--no-plots", action="store_true", help="Only print stats, skip generating any image files")
    args = parser.parse_args()

    print(f"Loading: {args.original}")
    original = load_rawvol(args.original)

    denoised = None
    if args.denoised:
        print(f"Loading: {args.denoised}")
        denoised = load_rawvol(args.denoised)

    print()
    print_stats(original, denoised)

    if args.no_plots:
        return

    plt = setup_matplotlib()
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    if denoised is not None and denoised.shape == original.shape:
        plot_slice_comparison(plt, original, denoised,
                               os.path.join(args.output_dir, "volume_slices.png"),
                               args.slice_z, args.slice_y, args.slice_x)
        plot_histogram(plt, original, denoised,
                        os.path.join(args.output_dir, "volume_histogram.png"))
    else:
        plot_single_volume(plt, original,
                            os.path.join(args.output_dir, "volume_slices.png"),
                            args.slice_z, args.slice_y, args.slice_x)


if __name__ == "__main__":
    main()