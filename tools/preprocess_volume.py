#!/usr/bin/env python3
"""
preprocess_volume.py -- load common 3D medical imaging formats (NIfTI,
DICOM series, or a raw numpy .npy array), inspect basic info, normalize to
[0,1] with a clearly logged transform, and save as the project's .rawvol
format for use with HilbertCUDA-TV --mode volume.

HilbertCUDA-TV.exe now auto-normalizes and warns about this itself
as a safety net, but the RIGHT place to normalize medical data is here,
where you can choose a clinically-meaningful window (e.g. a CT lung or
soft-tissue window) rather than relying on naive min-max normalization,
which is sensitive to a handful of extreme outlier voxels (air, metal
artifacts) and can badly compress the useful intensity range.

Dependencies: numpy (required). nibabel (for NIfTI), pydicom (for DICOM)
are imported lazily -- you only need the one matching your actual input
format, not both.
"""

import argparse
import os
import struct
import sys
from glob import glob

import numpy as np

RAWVOL_MAGIC = 0x564C4854  # must match include/utils/VolumeIO.h's kRawVolMagic


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_nifti(path):
    """Load a .nii or .nii.gz file. Returns (data, meta) where data is a
    float32 numpy array and meta is a dict of basic header info for the
    info-only report."""
    try:
        import nibabel as nib
    except ImportError:
        sys.exit("Error: loading NIfTI files requires the 'nibabel' package.\n"
                 "Install it with: pip install nibabel")
    img = nib.load(path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    meta = {
        "format": "NIfTI",
        "affine": img.affine.tolist() if hasattr(img, "affine") else None,
        "voxel_sizes": [float(x) for x in img.header.get_zooms()] if hasattr(img, "header") else None,
    }
    return data, meta


def load_dicom_series(folder_path):
    """Load a folder of .dcm slices as a single 3D volume, sorted by
    ImagePositionPatient[2] (the standard slice-ordering convention).
    Returns (data, meta)."""
    try:
        import pydicom
    except ImportError:
        sys.exit("Error: loading DICOM files requires the 'pydicom' package.\n"
                 "Install it with: pip install pydicom")
    files = sorted(glob(os.path.join(folder_path, "*.dcm")))
    if not files:
        sys.exit(f"Error: no .dcm files found in {folder_path}")
    slices = [pydicom.dcmread(f) for f in files]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        print("Warning: could not sort by ImagePositionPatient[2] (missing "
              "or malformed tag); using filename order instead, which may "
              "not match true anatomical slice order.", file=sys.stderr)
    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)

    # Apply rescale slope/intercept if present (standard DICOM convention
    # for converting raw pixel values to real-world units, e.g. CT
    # Hounsfield units) -- skipping this is a common source of "my CT data
    # doesn't look like it's in HU units" confusion.
    slope = getattr(slices[0], "RescaleSlope", 1.0)
    intercept = getattr(slices[0], "RescaleIntercept", 0.0)
    if slope != 1.0 or intercept != 0.0:
        volume = volume * float(slope) + float(intercept)
        print(f"Applied DICOM RescaleSlope={slope}, RescaleIntercept={intercept}")

    meta = {
        "format": "DICOM series",
        "num_slices": len(files),
        "rescale_slope": float(slope),
        "rescale_intercept": float(intercept),
    }
    return volume, meta


def load_npy(path):
    """Load a plain numpy .npy file -- useful if you've already done your
    own loading/preprocessing in Python and just want this script's
    normalization + .rawvol export."""
    data = np.load(path).astype(np.float32)
    meta = {"format": "numpy .npy"}
    return data, meta


def load_volume_auto(path):
    """Dispatch to the right loader based on the input path."""
    if os.path.isdir(path):
        return load_dicom_series(path)
    lower = path.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return load_nifti(path)
    if lower.endswith(".npy"):
        return load_npy(path)
    sys.exit(f"Error: don't know how to load '{path}' -- expected a directory "
              f"of .dcm files, a .nii/.nii.gz file, or a .npy file.")


# ---------------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------------

def to_depth_height_width(data, force_transpose=False):
    """Returns data in (depth, height, width) order -- the convention this
    project's .rawvol format and README use.

    IMPORTANT: this does NOT guess your data's axis order from its shape.
    An earlier version of this script tried a "if shape[0] != shape[2],
    transpose" heuristic; testing found it guesses WRONG for the common
    case of real medical volumes, where depth (slice count) routinely
    differs from in-plane width regardless of whether the data is already
    in the right order -- shape comparison alone cannot distinguish
    "already correct, just asymmetric" from "needs reordering". There is
    no reliable shape-only heuristic for this.

    Instead: by default, NO reordering is applied (your data is assumed to
    already be in, or you will explicitly request, (depth, height, width)
    order). Pass --transpose to swap axes 0 and 2 if you've checked (e.g.
    via --info-only's printed NIfTI affine matrix, which DOES reliably
    encode true voxel orientation, unlike shape) that your data needs it.
    """
    if data.ndim != 3:
        sys.exit(f"Error: expected a 3D volume, got shape {data.shape} "
                  f"({data.ndim} dimensions). 4D data (e.g. multi-channel "
                  f"or time series) is not supported by this script.")
    if force_transpose:
        data = np.transpose(data, (2, 1, 0))
    return data


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_window(data, target_min, target_max):
    """Clip to [target_min, target_max], then rescale linearly to [0,1].
    Use this for CT data with a known, clinically-meaningful Hounsfield
    window (e.g. lung: [-1000, 500]; soft tissue/abdomen: [-100, 400] or
    [-160, 240]; bone: [-500, 1300] -- exact values vary by convention and
    body region, consult a radiology reference for your specific case)."""
    if target_max <= target_min:
        raise ValueError(f"target_max ({target_max}) must be > target_min ({target_min})")
    clipped = np.clip(data, target_min, target_max)
    normalized = (clipped.astype(np.float64) - target_min) / (target_max - target_min)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def normalize_percentile(data, low_pct, high_pct):
    """Clip to the given percentile range, then rescale to [0,1]. Use this
    when there's no standard intensity-unit window to rely on (e.g. MRI,
    which has no equivalent of CT's Hounsfield units) -- robust to a small
    number of extreme outlier voxels, unlike plain min-max."""
    if not (0 <= low_pct < high_pct <= 100):
        raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, got ({low_pct}, {high_pct})")
    lo, hi = np.percentile(data, (low_pct, high_pct))
    return normalize_window(data, lo, hi)


def normalize_minmax(data):
    """Plain min-max normalization using the data's own observed range.
    WARNING: sensitive to outlier voxels (e.g. air, metal/beam-hardening
    artifacts in CT) -- these can compress the clinically useful intensity
    range into a tiny sliver of [0,1]. Prefer --window or --percentile for
    real medical data; this mode is mainly useful for already-clean,
    synthetic, or non-medical 3D data."""
    vmin, vmax = float(data.min()), float(data.max())
    if vmax <= vmin:
        raise ValueError(f"data is constant (min == max == {vmin}); nothing to normalize")
    return ((data.astype(np.float64) - vmin) / (vmax - vmin)).astype(np.float32)


# ---------------------------------------------------------------------------
# .rawvol I/O (must stay byte-compatible with include/utils/VolumeIO.h)
# ---------------------------------------------------------------------------

def save_rawvol(path, volume_dhw):
    """volume_dhw: 3D numpy array, shape (depth, height, width), assumed
    ALREADY normalized to [0,1] -- this function does not normalize."""
    depth, height, width = volume_dhw.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", RAWVOL_MAGIC, width, height, depth))
        volume_dhw.astype(np.float32).tofile(f)


def load_rawvol(path):
    """Round-trip helper, also used by tools/compare_volumes.py and
    tools/visualize_volume.py -- kept here as the single source of truth
    for this format's Python-side reader so all three scripts agree."""
    with open(path, "rb") as f:
        magic, width, height, depth = struct.unpack("<4I", f.read(16))
        if magic != RAWVOL_MAGIC:
            raise ValueError(f"{path}: bad magic number (not a .rawvol file, "
                              f"or wrong byte order)")
        data = np.fromfile(f, dtype=np.float32, count=width * height * depth)
    return data.reshape(depth, height, width)


# ---------------------------------------------------------------------------
# Info report
# ---------------------------------------------------------------------------

def print_info(data, meta, path):
    print(f"=== {path} ===")
    print(f"Format:        {meta.get('format', 'unknown')}")
    print(f"Shape (as loaded, before any reordering): {data.shape}")
    print(f"Dtype:         {data.dtype}")
    print(f"Value range:   [{data.min():.6f}, {data.max():.6f}]")
    print(f"Mean / std:    {data.mean():.6f} / {data.std():.6f}")
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    if nan_count or inf_count:
        print(f"WARNING: {nan_count} NaN and {inf_count} Inf values present "
              f"-- these MUST be handled before this data can be used (the "
              f"denoiser will refuse data containing NaN/Inf).")
    if meta.get("voxel_sizes"):
        print(f"Voxel sizes (mm, as reported in header): {meta['voxel_sizes']}")
    if meta.get("affine") is not None:
        print("Affine matrix (NIfTI orientation -- check this if your")
        print("converted volume looks transposed or flipped):")
        for row in meta["affine"]:
            print("  ", ["%.3f" % v for v in row])
    looks_normalized = -0.05 <= data.min() and data.max() <= 1.05
    print(f"Looks already normalized to [0,1]? {'Yes' if looks_normalized else 'No -- normalization recommended before use with HilbertCUDA-TV'}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load, inspect, and normalize 3D medical imaging volumes "
                     "for use with HilbertCUDA-TV --mode volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True,
                         help="Path to a .nii/.nii.gz file, a folder of .dcm files, or a .npy file")
    parser.add_argument("--output", help="Path to write the .rawvol file (required unless --info-only)")
    parser.add_argument("--info-only", action="store_true",
                         help="Just print info about the input and exit -- don't normalize or save anything")
    parser.add_argument("--transpose", action="store_true",
                         help="Swap axes 0 and 2 (depth<->width) before normalizing. "
                              "This script does NOT guess your data's axis order -- "
                              "check --info-only's printed shape (and NIfTI affine "
                              "matrix, if applicable) yourself and pass this flag "
                              "only if your data actually needs reordering to reach "
                              "(depth, height, width) order.")

    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--window", nargs=2, type=float, metavar=("MIN", "MAX"),
                             help="Clip to [MIN,MAX] then normalize to [0,1] -- use a clinically "
                                  "meaningful CT Hounsfield window, e.g. --window -1000 500 for lung, "
                                  "--window -100 400 for abdomen/soft-tissue")
    norm_group.add_argument("--percentile", nargs=2, type=float, metavar=("LOW", "HIGH"),
                             help="Clip to the [LOW,HIGH] percentile range then normalize -- use for "
                                  "MRI or other data without a standard intensity-unit window, "
                                  "e.g. --percentile 1 99")
    norm_group.add_argument("--minmax", action="store_true",
                             help="Plain min-max normalization using the data's own range. "
                                  "WARNING: sensitive to outlier voxels -- see script docstring. "
                                  "Mainly useful for already-clean or non-medical data.")

    args = parser.parse_args()

    if not args.info_only and not args.output:
        parser.error("--output is required unless --info-only is given")
    if not args.info_only and not (args.window or args.percentile or args.minmax):
        parser.error("one of --window, --percentile, or --minmax is required unless --info-only is given")

    print(f"Loading: {args.input}")
    data, meta = load_volume_auto(args.input)
    print_info(data, meta, args.input)

    if args.info_only:
        return

    data = to_depth_height_width(data, force_transpose=args.transpose)
    if args.transpose:
        print("Applied --transpose (axis 0<->2 swap).")

    orig_min, orig_max = float(data.min()), float(data.max())

    try:
        if args.window:
            target_min, target_max = args.window
            normalized = normalize_window(data, target_min, target_max)
            transform_desc = f"window [{target_min}, {target_max}]"
        elif args.percentile:
            low_pct, high_pct = args.percentile
            normalized = normalize_percentile(data, low_pct, high_pct)
            lo, hi = np.percentile(data, (low_pct, high_pct))
            transform_desc = f"percentile [{low_pct},{high_pct}] -> clip window [{lo:.2f}, {hi:.2f}]"
        else:
            normalized = normalize_minmax(data)
            transform_desc = "plain min-max"
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # Explicitly logged normalization transform, per the project's standing convention:
    # always state the before/after range so normalization is never a silent, undocumented transformation.
    print(f"Normalization: {transform_desc}")
    print(f"Range: [{orig_min:.6f}, {orig_max:.6f}] -> Scope: [{normalized.min():.6f}, {normalized.max():.6f}]")

    print(f"Final shape (depth, height, width): {normalized.shape}")
    save_rawvol(args.output, normalized)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
