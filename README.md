# HilbertCUDA-TV

A GPU-accelerated **Total Variation (TV) denoiser** for grayscale images, color
images, and 3D volumetric data, built with CUDA and the Chambolle-Pock
primal-dual algorithm.

Built for and tested on: **NVIDIA RTX 4080 Super** (Ada Lovelace, compute
capability 8.9), **CUDA 12.4**, **Windows 11**, **Visual Studio 2022**.

---

## Table of contents

- [HilbertCUDA-TV](#hilbertcuda-tv)
  - [Table of contents](#table-of-contents)
  
  - [What this does, and the math behind it](#what-this-does-and-the-math-behind-it)
  - [**Quick start**](#quick-start--entrance-)
  - [User guide](#user-guide)
    - [**Grayscale** denoising](#grayscale-denoising)
    - [**Color** denoising](#color-denoising)
    - [**3D volume**tric denoising](#3d-volumetric-denoising)
      - [Volume file format (`.rawvol`)](#volume-file-format-rawvol)
    - [`--noise-sigma` vs `--lambda`](#--noise-sigma-vs---lambda-what-each-one-actually-controls)
    - [How lambda is calculated](#how-lambda-is-auto-estimated-when---lambda-is-omitted)
      - [`--no-auto-normalize` and input value ranges](#--no-auto-normalize-and-input-value-ranges)
    - [Profiling](#profiling)
  - [**Evaluation** (tools/)](#tools-python-scripts-for-2d-image-and-3d-volume-comparison)
    - [preprocess\_volume.py](#preprocess_volumepy)
    - [visualize\_volume.py](#visualize_volumepy)
    - [compare\_volumes.py vs `--reference` mode](#compare_volumespy-vs---reference-mode)
    - [compare\_images.py vs `--reference` mode](#compare_imagespy-vs---reference-mode)
    - [hctv\_metrics.py](#hctv_metricspy)
    - [Running the tools/ test suite](#running-the-tools-test-suite)
  - [**Full CLI reference**](#full-cli-reference)
  - [Developer guide](#developer-guide)
    - [Project layout](#project-layout)
    - [Build options](#build-options)
    - [Architecture](#architecture)
    - [Extending this project](#extending-this-project)
    - [Testing and validation](#testing-and-validation)
  - [Known limitations and not-yet-implemented features](#known-limitations-and-not-yet-implemented-features)
  - [**FAQ** / troubleshooting](#faq--troubleshooting)

---

## What this does, and the math behind it

Given a noisy image (or 3D volume) `f`, this tool finds a denoised version
`u` that minimizes:

```
min_u (1/2)‖u - f‖² + λ · TV(u)
```

The first term keeps `u` close to the input; the second term, the **total
variation** `TV(u) = Σ|∇u|`, penalizes how "jagged" `u` is. The balance
between the two is controlled by **λ**: higher λ means more smoothing.

What makes TV denoising distinctive compared to a Gaussian blur is that it
penalizes *gradient magnitude*, not gradient *smoothness* — so it removes
noise (lots of small, random gradients) while preserving sharp edges (one
big, consistent gradient is "cheap" in TV terms, however abrupt).

### How it's solved

This is a convex but non-smooth optimization problem (the TV term isn't
differentiable at zero gradient). It's solved with the **Chambolle-Pock
primal-dual algorithm**, which alternates between two simple steps:

1. **Dual ascent + projection** — update an auxiliary "dual" vector field
   `p` (one 2D, or 3D, vector per pixel/voxel) representing the local
   gradient direction, then project it so its magnitude never exceeds λ.
2. **Primal descent + extrapolation** — use `p`'s divergence to update `u`
   toward the noisy input `f`, then extrapolate to accelerate convergence.

These two steps repeat for a fixed number of iterations (300 by default),
entirely on the GPU — only the initial upload and final result download
cross the PCIe bus.

### The operators, concretely

The gradient operator `K` is a forward finite difference with **Neumann
(zero) boundary conditions**:

```
(Ku)_x[i,j] = u[i,j+1] - u[i,j]   if j < width-1, else 0
(Ku)_y[i,j] = u[i+1,j] - u[i,j]   if i < height-1, else 0
```

Its adjoint `K*` (used in the primal step) is a **gated** backward
divergence — the gating matters, see the box below:

```
K*p [i,j] = (j>0 ? p_x[i,j-1] : 0) + (j<width-1  ? -p_x[i,j] : 0)
          + (i>0 ? p_y[i-1,j] : 0) + (i<height-1 ? -p_y[i,j] : 0)
```

> **Why "gated"?** A common simplified version of this formula omits the
> `j<width-1` / `i<height-1` conditions on the negative terms. That
> simplified version is only correct in the image interior — at the
> boundary it breaks the adjoint identity `⟨Ku,p⟩ = ⟨u,K*p⟩`, which this
> algorithm's correctness depends on. This was caught empirically during
> development (an independent CPU double-precision implementation, run
> before any GPU code was written, found the adjoint identity failing by
> O(1) with the simplified formula and by ~1e-13 — pure floating-point
> roundoff — with the gated version). All gradient/divergence kernels in
> this codebase use the gated form.

The step sizes `τ, σ` in the algorithm must satisfy `τσ‖K‖² ≤ 1` for
convergence. For the 2D operator above, `‖K‖² ≤ 8`, so `τ=σ=1/√8`. For the
**3D** volumetric case (an extra z-direction term), the bound becomes
`‖K‖² ≤ 12`, so the 3D solver uses `τ=σ=1/√12` — confirmed by direct
eigenvalue computation, not just pattern-matched from the 2D case.

### Color images: coupled vectorial TV

For color images, this tool does **not** run three independent grayscale
solves (one per R/G/B channel). Independent per-channel TV causes color
fringing — edges in different channels can land at slightly different
places after denoising. Instead, it uses **coupled vectorial TV**:

```
TV(u) = Σ_pixel sqrt(Σ_channel |∇u_channel|²)
```

The gradient magnitude is computed *jointly* across all channels before
the TV penalty is applied, so all three channels are smoothed (or not
smoothed) together at every edge.

### Reference

This implementation follows Chambolle & Pock, *"A First-Order Primal-Dual
Algorithm for Convex Problems with Applications to Imaging"* (2011), and
Rudin, Osher & Fatemi, *"Nonlinear total variation based noise removal
algorithms"* (1992) for the underlying ROF model.

---

## Quick start (!!! ENTRANCE !!!)

```powershell
# 1. Build (stb_image headers are already vendored in third_party/ -- no separate download step)
.\scripts\build_windows.ps1

# 2. Verify it works on your hardware
.\build\Release\test_adjoint.exe
.\build\Release\test_denoise.exe

# 3. Try it
.\build\Release\HilbertCUDA-TV.exe --demo --output result.png
```

### Setup (one-time)

This project uses the single-header **stb_image** / **stb_image_write**
libraries for PNG/JPG I/O, and they're already vendored in
`third_party/` — no download needed for the default build.

(If you'd rather use OpenCV instead, build with `-DHCTV_USE_OPENCV=ON`;
see `third_party/README.md` if you ever need to re-fetch or update the
vendored stb headers yourself.)

### Build

```powershell
.\scripts\build_windows.ps1
```

This configures and builds with CMake + Visual Studio 2022 + CUDA 12.4,
targeting `sm_89`. If you're on a different GPU generation, override the
architecture: `.\scripts\build_windows.ps1` uses `CMAKE_CUDA_ARCHITECTURES=89`
by default; pass a different value via `cmake -DCMAKE_CUDA_ARCHITECTURES=<cc>`
if you build manually (86 for RTX 30-series, 75 for RTX 20-series, 90 for
H100, etc).

This produces, under `build\Release\`:

| Executable | Purpose |
|---|---|
| `HilbertCUDA-TV.exe` | Main CLI tool (all three modes) |
| `test_adjoint.exe` | Grayscale GPU correctness check |
| `test_denoise.exe` | Grayscale end-to-end PSNR/SSIM check |
| `test_color_adjoint.exe` | Color GPU correctness check |
| `test_color_denoise.exe` | Color end-to-end PSNR check |
| `test_volume_adjoint.exe` | 3D volumetric GPU correctness check |
| `test_volume_denoise.exe` | 3D volumetric end-to-end PSNR check |
| `test_range_check.exe` | `RangeCheck.h` unit test (plain C++, no GPU actually used) |

**Run the adjoint tests first** after any build, before trusting results —
they verify the core math is wired correctly on your specific GPU/driver.

---

## User guide

### Grayscale denoising

```powershell
# Denoise an existing noisy image
.\HilbertCUDA-TV.exe --input my_photo.jpg --output result.png

# Try it on a built-in synthetic test pattern (no input file needed)
.\HilbertCUDA-TV.exe --demo --width 512 --height 512 --output result.png

# Denoise a clean image with synthetic noise added, and see before/after metrics
.\HilbertCUDA-TV.exe --reference my_clean_photo.png --output result.png --noise-sigma 25
```

### Color denoising

Add `--mode color`. Otherwise the same options apply:

```powershell
.\HilbertCUDA-TV.exe --mode color --input my_photo.jpg --output result.png
.\HilbertCUDA-TV.exe --mode color --demo --output result.png
.\HilbertCUDA-TV.exe --mode color --reference my_clean_photo.png --noise-sigma 25
```

### 3D volumetric denoising

Add `--mode volume`. Volume data uses a minimal custom binary format
(`.rawvol`) instead of PNG/JPG — see [Volume file format](#volume-file-format-rawvol)
below for why, and how to create one from your own data (e.g. a numpy array,
a DICOM/NIfTI volume you've already loaded elsewhere, a stack of TIFF slices).

```powershell
.\HilbertCUDA-TV.exe --mode volume --input scan.rawvol --output result.rawvol
.\HilbertCUDA-TV.exe --mode volume --demo --width 64 --height 64 --depth 64 --output result.rawvol
```

#### Volume file format (`.rawvol`)

A `.rawvol` file is: a 16-byte header (4 little-endian `uint32`s: a magic
number, width, height, depth), followed by `width*height*depth` raw
`float32` voxel values in `[0,1]`, row-major with z slowest-varying
(`index = z*height*width + y*width + x`). No compression, no metadata.

This was chosen over DICOM or NIfTI deliberately: those are real, heavy
formats with clinical metadata, multi-file series, and orientation/affine
semantics that this denoiser has no use for — it only needs a 3D array of
numbers. If you have a DICOM series, NIfTI file, or numpy array, converting
it to `.rawvol` is a few lines:

```python
import numpy as np
import struct

def save_rawvol(path, volume):
    """volume: 3D numpy array, shape (depth, height, width), any numeric
    dtype -- will be cast to float32 and assumed already normalized to
    [0,1] (rescale your data yourself first if it isn't, e.g. divide a
    12-bit CT volume by 4095.0)."""
    depth, height, width = volume.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<4I', 0x564C4854, width, height, depth))
        volume.astype(np.float32).tofile(f)

# Example: convert a stack of DICOM slices loaded via pydicom/SimpleITK
# into a single normalized numpy array, then save:
#   volume = my_loaded_dicom_array / my_loaded_dicom_array.max()
#   save_rawvol("scan.rawvol", volume)
```

Reading a `.rawvol` file back in Python (e.g. to inspect a result):

```python
import numpy as np
import struct

def load_rawvol(path):
    with open(path, 'rb') as f:
        magic, width, height, depth = struct.unpack('<4I', f.read(16))
        data = np.fromfile(f, dtype=np.float32, count=width*height*depth)
    return data.reshape(depth, height, width)
```

### `--noise-sigma` vs `--lambda`: what each one actually controls

These two flags are easy to conflate because they're both "noise-related"
numbers on the command line, but **they do completely different jobs and
neither one sets the other**. Mixing them up is the single most common
source of confusing results with this tool, so read this before tuning
either one.

| | `--noise-sigma` | `--lambda` |
|---|---|---|
| **What it controls** | How much *synthetic* noise to manufacture and inject | How aggressively the solver smooths whatever noise is *actually present* |
| **Used by which modes** | `--demo` and `--reference` ONLY | All modes |
| **Used by `--input` mode?** | **No effect at all** — there's no synthetic noise to inject; `--input` solves on the file exactly as loaded | Yes |
| **Scale** | 0–255 (e.g. `25` ≈ moderate noise, matching the common "σ=25" benchmark convention in the denoising literature) | Solver-internal weight, typically `0.001`–`0.5` for `[0,1]`-scale data |
| **Default** | `25.0` | None — auto-estimated from the actual data if omitted (see below) |
| **Does setting it set the other?** | **No.** It only changes how much noise gets injected; lambda is then estimated from (or overridden independently of) whatever noise level actually resulted | **No.** Explicit `--lambda` overrides auto-estimation entirely, in every mode, regardless of `--noise-sigma` |

**The calculation order, every time, in every mode:**

1. Get `solve_input` — the array the solver will actually run on. For
   `--input`, this is just the loaded file. For `--demo`/`--reference`,
   this is the loaded/generated clean image **after** `--noise-sigma`
   worth of synthetic Gaussian noise has been injected into it.
2. Decide `lambda`:
   - If you passed `--lambda <value>` explicitly: use exactly that value.
     `--noise-sigma` is irrelevant to this decision — it already did its
     only job in step 1 (or had no job at all, in `--input` mode).
   - Otherwise: estimate the noise level **by measuring `solve_input`
     directly** (the array from step 1, not the `--noise-sigma` value you
     typed), then set `lambda = 1.5 × estimated_sigma`. This applies
     uniformly in all three modes — see "How lambda is auto-estimated"
     below for why it's measured rather than assumed.
3. Solve, using whatever `lambda` step 2 produced.

**Why this trips people up — three concrete combinations:**

- **`--input photo.png` with no `--lambda`:** `--noise-sigma` is never
  even read. Lambda is estimated from `photo.png`'s own actual noise
  level. This is the right choice for "I have a real noisy photo and
  want it cleaned up" — there's no clean ground truth to inject noise
  into, so `--reference`'s self-test mechanism doesn't apply here at all.
- **`--reference clean.png --noise-sigma 25` with no `--lambda`:** 25/255
  worth of noise is injected into `clean.png`, lambda is then estimated
  from *that noisy result* (not assumed to be exactly 25/255 — see
  below), and the printed "improvement" numbers tell you how well the
  solver removed noise of *that specific, known severity*. This is a
  controlled self-test, not a measurement of `clean.png`'s real-world
  noise (it has none — that's the point).
- **`--reference clean.png --noise-sigma 0.01` with no `--lambda`:** this
  is the case that produced a *negative* "improvement" in testing (PSNR
  noisy-vs-clean: 88 dB; PSNR denoised-vs-clean: 47 dB). This is not a
  bug: `--noise-sigma` is on a **0–255** scale, so `0.01` injects an
  almost imperceptible amount of noise — the "noisy" image is already
  99.99%+ identical to clean. The solver still measures *some* residual
  noise (there's always a little, even in a clean image, from the
  estimator's own ~1–4% baseline uncertainty — see section 15 of
  `devdocs/DEV_LOG.md`) and applies a correspondingly tiny but nonzero
  `lambda`, which **smooths real detail that was never noise in the
  first place**. Lower PSNR after "denoising" an already-clean image is
  the expected, correct outcome of doing this, not a sign anything is
  broken. If you actually wanted a *light* noise injection on the
  0–255 scale, something like `--noise-sigma 2` to `--noise-sigma 5` is
  "barely visible," not `0.01`.

**If your result is over- or under-smoothed and you don't know why,**
check in this order: (1) confirm whether `--lambda` was actually passed —
if not, the auto-estimate is doing the deciding, not anything you typed
for `--noise-sigma`; (2) for `--demo`/`--reference`, check whether
`--noise-sigma` is the value you actually intended on the 0–255 scale;
(3) only then consider overriding `--lambda` directly.

### How lambda is auto-estimated (when `--lambda` is omitted)

If you don't specify `--lambda`, this tool picks one automatically using
`lambda = 1.5 × (estimated_noise_sigma / 255)`.

**The noise sigma is always estimated directly from the actual array the
solver is about to run on** — in every mode (`--input`, `--demo`,
`--reference`), not just plain `--input`. Concretely:

- `--input`: estimated from the loaded input image/volume itself.
- `--demo`: estimated from the generated synthetic image *after* its
  built-in noise has been added (not assumed equal to `--noise-sigma`).
- `--reference`: estimated from the noisy image *after* synthetic noise is
  injected into the clean reference (again, not assumed equal to
  `--noise-sigma`).

This used to be inconsistent: `--demo`/`--reference` previously trusted
`--noise-sigma` directly instead of measuring the data, which silently
assumed the file being noised was already clean [0,1] data — true for
`--demo`'s own synthetic image, but not guaranteed for a user-supplied
`--reference` file (e.g. unnormalized medical data, or a real photo that
already had its own noise before injection). Estimating from the actual
array in every mode fixes that inconsistency.

This was numerically re-verified with a Python replica of the exact
estimator and the built-in synthetic test image (see
`devdocs/DEV_LOG.md` section 30 for the full derivation): at the
historically-validated default `--noise-sigma 25`, the resulting lambda
differs from the old behavior by under ~1% on average, so old `--demo`
PSNR/SSIM numbers at that setting remain a reasonable sanity check. **This
is not a tight bound at every setting**, though — the gap grows with
`--noise-sigma` (up to ~5% by `--noise-sigma 60` in the same test),
because heavier injected noise clips more pixels at the `[0,1]` boundary,
a real and expected nonlinearity, not an estimator bug: clipping
suppresses the *effective* noise variance the estimator measures, so its
estimate increasingly undershoots the nominal injected value as more
pixels saturate. An earlier internal note claimed a flat "~0.1%" bound;
that number turned out to be specific to one seed/sigma combination and
didn't hold up under broader re-verification, so it's been corrected
here and in the code comments.

Why this matters in practice: a fixed `lambda` tuned for one noise level
can badly over-smooth a cleaner image — early testing found that a
default tuned for heavy synthetic noise (σ=25/255) turned a normal,
lightly-noisy photograph into a posterized, "cartoon-like" result,
because it was smoothing away real texture (fur, grass, fabric) along
with the (mostly absent) noise. Auto-estimation from the real data fixes
this for typical photos; if a specific image still looks over- or
under-smoothed, override with `--lambda <value>` directly (try halving or
doubling it and compare).

#### `--no-auto-normalize` and input value ranges

Lambda estimation (and the solver itself) assumes `[0,1]`-scale data. If
loaded data falls outside that range (e.g. raw CT Hounsfield values, or
an unnormalized `.npy` array), `HilbertCUDA-TV.exe` auto-detects this and
auto-normalizes it via min-max rescaling before solving, printing a
warning so this isn't silent. Pass `--no-auto-normalize` to disable this
safety net and solve on the raw values as-is — lambda will still be
estimated relative to whatever scale the data is actually on, but results
will be harder to interpret or compare across datasets than if you
normalize consistently yourself first.

For medical/scientific volumes specifically, **the auto-normalizer is a
safety net, not the recommended workflow** — plain min-max normalization
is sensitive to a handful of extreme outlier voxels (air, metal
artifacts) and can badly compress the clinically useful intensity range.
Use `tools/preprocess_volume.py` to normalize with a clinically-meaningful
window or robust percentile clipping *before* running the solver — see
[tools/](#tools-python-scripts-for-2d-image-and-3d-volume-comparison) below.

### Profiling

```powershell
.\scripts\run_nsight_profile.ps1 -Mode timeline   # Nsight Systems: overall iteration loop
.\scripts\run_nsight_profile.ps1 -Mode kernel     # Nsight Compute: per-kernel occupancy/memory metrics
```

---

## tools/ (Python scripts for 2D image and 3D volume comparison)

Five Python scripts live here. Four support the volumetric (`--mode
volume`) workflow: converting real medical data into `.rawvol`,
inspecting results visually, and measuring accuracy in ways `--reference`
mode doesn't cover. The fifth, `compare_images.py`, is the 2D (gray/color
PNG/JPG) counterpart to `compare_volumes.py`, for the same
"`--reference` mode can't directly compare two existing files" gap in
the `--mode gray`/`--mode color` workflow.

All five need only `numpy`; `preprocess_volume.py` additionally needs
`nibabel` (NIfTI) and/or `pydicom` (DICOM) — only the one matching your
input format, imported lazily; `compare_images.py` additionally needs
`Pillow` to read/write PNG/JPG files.

```bash
pip install numpy nibabel pydicom matplotlib Pillow   # matplotlib for visualize_volume.py, Pillow for compare_images.py
```

### *preprocess_volume.py*

Loads NIfTI (`.nii`/`.nii.gz`), a DICOM series (a folder of `.dcm`
slices), or a raw `.npy` array; normalizes it to `[0,1]`; saves it as
`.rawvol`.

```bash
# CT, soft-tissue window (clinically meaningful, not raw min-max)
python tools/preprocess_volume.py --input liver_001.nii.gz --output liver_001.rawvol --window -100 400

# CT, lung window
python tools/preprocess_volume.py --input scan.nii.gz --output scan.rawvol --window -1000 500

# MRI (no standard Hounsfield-style units -- use robust percentile clipping instead)
python tools/preprocess_volume.py --input brain.nii.gz --output brain.rawvol --percentile 1 99

# DICOM series folder
python tools/preprocess_volume.py --input ./dicom_series/ --output scan.rawvol --window -1000 500

# Just inspect a file without converting anything
python tools/preprocess_volume.py --input scan.nii.gz --info-only
```

**Why windowing/percentile clipping instead of plain min-max?** Real
medical volumes routinely contain a handful of extreme outlier voxels
(air pockets, metal/beam-hardening artifacts). Plain min-max
normalization stretches those outliers to fill the `[0,1]` range, which
compresses the clinically useful intensity values into a tiny sliver of
that range. `--window` (a known, clinically-meaningful intensity range)
or `--percentile` (robust clipping when there's no standard unit, e.g.
MRI) avoid this. Plain min-max (no `--window`/`--percentile` flag) is
still available and is fine for already-clean, synthetic, or non-medical
volumes — just be deliberate about which mode you're using for real
clinical data.

This script does **not** guess your data's axis order from its shape —
an earlier version tried a shape-comparison heuristic and it guessed
wrong for typical asymmetric medical volumes (depth routinely differs
from in-plane width regardless of whether the data is already correctly
ordered, so shape alone can't tell "correct but asymmetric" from "needs
reordering"). By default no reordering happens; pass `--transpose` only
if you've separately confirmed (e.g. via `--info-only`'s printed NIfTI
affine matrix) that your data actually needs axes 0 and 2 swapped.

### *visualize_volume.py*

Generates a 3×3 grid of orthogonal slices (original / denoised /
residual, axial/coronal/sagittal) plus a histogram comparison, for
visually sanity-checking a denoising result.

```bash
python tools/visualize_volume.py --original scan.rawvol --denoised result.rawvol --output comparison.png
```

### *compare_volumes.py* vs `--reference` mode

These answer **different questions** — use the right one:

| | `HilbertCUDA-TV.exe --reference` | `tools/compare_volumes.py` |
|---|---|---|
| What it measures | How well a *known, synthetically injected* noise level was removed | How similar any two `.rawvol` files actually are |
| Needs ground truth? | No (it manufactures its own noisy/clean pair from one file) | Only if you want an accuracy interpretation — you supply both files either way |
| Use it for | A controlled self-test of the solver/lambda on a clean reference image | Comparing your result against independent ground truth, comparing two parameter choices' outputs against each other, or batch-comparing many file pairs at once |

Concretely: `--reference` loads **one** file, adds synthetic noise itself,
denoises, and reports how much of *that specific injected noise* was
removed. It cannot tell you how your denoised output compares to an
independently-acquired clean scan, or how two different `--lambda`
choices' outputs differ from each other — those need two independently
produced files compared directly, which is what `compare_volumes.py`
does instead.

```bash
# Compare a denoised result against independent ground truth
python tools/compare_volumes.py --a ground_truth.rawvol --b result.rawvol

# Compare two different denoising runs against each other (no ground truth needed)
python tools/compare_volumes.py --a result_lambda_0.05.rawvol --b result_lambda_0.15.rawvol

# Batch mode: compare many pairs at once, write a CSV summary.
# pairs.csv: two columns, no header, one pair per line (# starts a comment,
# malformed/missing/shape-mismatched rows are skipped with a warning, not fatal):
#   ground_truth_1.rawvol,result_1.rawvol
#   ground_truth_2.rawvol,result_2.rawvol
python tools/compare_volumes.py --batch pairs.csv --output summary.csv
```

### *compare_images.py* vs `--reference` mode

This is the 2D (gray/color PNG/JPG) counterpart to `compare_volumes.py`
above — same rationale, same answer to the same gap: `--reference` mode
(available in `--mode gray` and `--mode color`, same as `--mode volume`)
is a controlled self-test that injects its own synthetic noise; it
cannot directly tell you how close an existing denoised PNG is to an
existing clean PNG. `compare_images.py` does that directly, for ordinary
image files instead of `.rawvol` volumes:

```bash
# Compare a denoised result against independent ground truth
python tools/compare_images.py --a clean.png --b denoised.png

# Also save a visual diff heatmap (blue = A brighter, red = B brighter)
python tools/compare_images.py --a clean.png --b denoised.png --diff-output diff.png

# Compare two different denoising runs against each other (no ground truth needed)
python tools/compare_images.py --a result_lambda_0.05.png --b result_lambda_0.15.png

# Batch mode: same pairs.csv format as compare_volumes.py
python tools/compare_images.py --batch pairs.csv --output summary.csv
```

Works on grayscale OR color images (PNG, JPG, BMP, or anything else
Pillow reads), including a 16-bit grayscale PNG if you happen to have
one — though note HilbertCUDA-TV.exe itself only ever reads/writes 8-bit
PNGs (`include/utils/ImageIO.h` uses `stb_image`'s 8-bit loader, never
its 16-bit variant), so 16-bit support here is only for comparing against
*other* sources, not anything this project's own `--output` will produce.

**Color images get a real per-channel SSIM, not a misread.** A color
image loaded as a `(H,W,3)` array could, if passed directly into
`hctv_metrics.ssim_windowed()`, be silently misinterpreted as a 3-slice
*volume* (depth=H, height=W, width=3) instead of an image — that function
only natively understands 2D images or true 3D volumes, with no way to
tell a `(H,W,3)` color image apart from a 3-slice volume by shape alone.
`compare_images.py` avoids this by computing SSIM independently on each
of the R/G/B channels (each a genuine 2D image) and reporting both the
per-channel breakdown and their average — the same convention
scikit-image uses for multichannel SSIM.

### *hctv_metrics.py*

Not a CLI tool — a shared PSNR/SSIM/noise-estimator module imported by
`visualize_volume.py`, `compare_volumes.py`, and `compare_images.py`, so
none of the three scripts can silently compute "PSNR" or "SSIM" slightly
differently from each other. Its windowed-SSIM formulation
(non-overlapping tiles/cubes, not the more common 11×11 Gaussian-window
SSIM) deliberately mirrors `include/utils/Metrics.h`'s C++ implementation
as closely as practical — numbers from these Python tools will read
differently from e.g. scikit-image's SSIM; that's expected, not a bug.

### Running the tools/ test suite

```bash
cd tools/
python -m pytest test_hctv_metrics.py test_preprocess_volume.py test_compare_volumes.py test_compare_images.py -v
```

---

## Full CLI reference

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--mode gray\|color\|volume` | `gray` | all | Which denoiser to run |
| `--input <path>` | — | all | Load this file and denoise it as-is |
| `--demo` | — | all | Generate a built-in synthetic test case |
| `--reference <path>` | — | all | Load a clean file, add synthetic noise, denoise, report PSNR/SSIM |
| `--output <path>` | `denoised.png` / `denoised.rawvol` | all | Where to save the result |
| `--lambda <float>` | auto-estimated | all | TV regularization weight; higher = smoother (see [`--noise-sigma` vs `--lambda`](#--noise-sigma-vs---lambda-what-each-one-actually-controls)) |
| `--iterations <int>` | `300` | all | Chambolle-Pock iteration count |
| `--naive` | off | gray only | Use non-shared-memory kernels (for A/B comparison) |
| `--no-auto-normalize` | off (auto-normalize is on) | all | Disable automatic [0,1] rescaling of out-of-range input (see [How lambda is auto-estimated](#how-lambda-is-auto-estimated-when---lambda-is-omitted)) |
| `--noise-sigma <float>` | `25.0` | `--demo`/`--reference` | Synthetic noise std-dev, 0–255 scale ONLY (see [`--noise-sigma` vs `--lambda`](#--noise-sigma-vs---lambda-what-each-one-actually-controls) — does not affect `--input` mode and does not directly set `--lambda`) |
| `--width <int>` | `512` | `--demo` (gray/color) | Demo image width |
| `--height <int>` | `512` | `--demo` (gray/color) | Demo image height |
| `--depth <int>` | `64` | `--demo` (volume) | Demo volume depth |

---

## Developer guide

### Project layout

```
HilbertCUDA-TV/
├── CMakeLists.txt
├── .gitignore
├── README.md                       <- you are here
├── include/
│   ├── core/
│   │   ├── HilbertOperator.cuh         abstract operator interface
│   │   ├── GradientOp.cuh              scalar 2D gradient/divergence
│   │   ├── DivergenceOp.cuh            thin spec-name-compat wrapper
│   │   ├── ColorGradientOp.cuh         multi-channel 2D gradient/divergence
│   │   └── VolumeGradientOp.cuh        3D gradient/divergence
│   ├── solvers/
│   │   ├── ROFSolver.cuh               scalar 2D solver
│   │   ├── ColorROFSolver.cuh          coupled vectorial color solver
│   │   └── VolumeROFSolver.cuh         3D volumetric solver
│   ├── distributed/
│   │   └── MultiGpuStub.cuh            interface-only multi-GPU placeholder
│   └── utils/
│       ├── CudaCheck.cuh               CUDA error-checking macros
│       ├── ImageIO.h                   PNG/JPG I/O, noise estimation, synthetic test images
│       ├── VolumeIO.h                  .rawvol I/O, 3D noise estimation, synthetic test volumes
│       ├── Metrics.h                   PSNR / SSIM
│       └── RangeCheck.h                value-range validation + auto-normalization (header-only, no CUDA dep)
├── src/
│   ├── core/{GradientOp,ColorGradientOp,VolumeGradientOp}.cu
│   ├── solvers/{ROFSolver,ColorROFSolver,VolumeROFSolver}.cu
│   ├── main.cpp                        CLI tool (all 3 modes)
│   └── dummy.cu                        build-system workaround, see comments in the file
├── tests/
│   ├── test_adjoint.cu / test_denoise.cu               (grayscale, needs CUDA)
│   ├── test_color_adjoint.cu / test_color_denoise.cu   (color, needs CUDA)
│   ├── test_volume_adjoint.cu / test_volume_denoise.cu (3D, needs CUDA)
│   └── test_range_check.cpp                            (RangeCheck.h, plain C++, no CUDA needed)
├── tools/
│   ├── preprocess_volume.py            NIfTI/DICOM/npy -> .rawvol, with clinical normalization
│   ├── visualize_volume.py             orthogonal-slice + histogram comparison grid
│   ├── compare_volumes.py              independent two-file (or batch) .rawvol comparison, no noise injection
│   ├── compare_images.py               same as above, for 2D PNG/JPG (gray or color)
│   ├── hctv_metrics.py                 shared PSNR/SSIM/noise-estimator module
│   ├── test_preprocess_volume.py       unit tests for preprocess_volume.py
│   ├── test_compare_volumes.py         unit tests for compare_volumes.py (incl. --batch)
│   ├── test_compare_images.py          unit tests for compare_images.py (incl. --batch)
│   └── test_hctv_metrics.py            unit tests for hctv_metrics.py
├── scripts/
│   ├── build_windows.ps1
│   └── run_nsight_profile.ps1
├── third_party/
│   ├── README.md                       vendored stb_image notes (files are already here, no download needed)
│   ├── stb_image.h                     vendored (v2.30, public domain)
│   └── stb_image_write.h               vendored (v1.16, public domain)
└── devdocs/
    ├── DEV_LOG.md                       running development log (see below)
    ├── verification/                    committed, re-runnable numerical-claim checks (see Testing and validation)
    │   ├── verify_lambda_fix.py
    │   ├── verify_metrics_math.py
    │   ├── verify_rangecheck_math.py
    │   ├── verify_batch_stats.py
    │   └── verify_operator_norm.py
    └── cpu_reference/
        └── cpu_reference.cpp            double-precision CPU reference + validator
```

### Build options

CMake options (pass as `-D<OPTION>=ON/OFF`):

| Option | Default | Effect |
|---|---|---|
| `HCTV_USE_OPENCV` | `OFF` | Use OpenCV for image I/O instead of stb_image |
| `HCTV_BUILD_TESTS` | `ON` | Build the `tests/` executables |
| `HCTV_DEBUG_CHECKS` | `OFF` | Sync + check for errors after every kernel launch (slower, more precise error locations) |
| `HCTV_FAST_MATH` | `OFF` | Enable `--use_fast_math`. **Caution**: this affects every division/sqrt in every kernel, not just the obvious ones — re-run the test suite after enabling to confirm nothing regressed for your case |

### Architecture

**Why three separate solver implementations (gray/color/volume) instead of
one generic templated solver?** The scalar 2D path was validated on real
hardware before the color/volume code was written. Generalizing the
existing kernels into one templated implementation would have required
re-deriving and re-verifying that already-working code from scratch, with
no GPU available during that development to re-confirm it still worked
afterward. Keeping the proven path untouched and adding new, separate
(structurally similar, but independent) kernel families for color and
volume was the lower-risk choice. A future unification is possible once
all three paths are independently hardware-validated, but wasn't done here.

**Why a custom `.rawvol` format instead of DICOM/NIfTI?** See
[Volume file format](#volume-file-format-rawvol) above.

**Why does `dummy.cu` exist?** A Windows + Visual Studio 2022 + CMake +
CUDA 12.4 quirk: a target with zero `.cu` files of its own, but which links
a static library built with relocatable device code, was found (on actual
hardware, not theoretically) to misconfigure the generated project's
device-link step. Adding one trivial `.cu` file to the main executable's
source list fixes it. See the comment at the top of `src/dummy.cu` and
`CMakeLists.txt`.

**Why is `--use_fast_math` opt-in, not default?** It doesn't just speed up
`sqrtf` in the TV projection step — it makes *every* division in *every*
kernel approximate, including ones on the convergence-correctness path
(the primal update's `/(1+τ)`, the projection's `q/scale`). Since this
could change numerical behavior in ways that are easy to get wrong, it's
opt-in with an explicit warning rather than a silent default.

### Extending this project

**Adding a new channel count for color** (e.g. RGBA): `kMaxColorChannels`
in `ColorROFSolver.cuh` is currently `4`; the kernels already support up
to that many channels via the `channels` field in `ColorROFParams`. Raise
the constant if you need more — the projection kernel uses fixed-size local
arrays sized to it, so there's a small, fixed register cost per increment.

**Multi-GPU**: `include/distributed/MultiGpuStub.cuh` defines the intended
future API (`MultiGpuROFSolver`, `MultiGpuConfig`) but every method
currently throws — it's an interface placeholder, not a working
implementation. The intended design (documented in comments in that file)
partitions the image/volume into row-bands or z-slabs across GPUs, with a
halo exchange between devices each iteration — conceptually the same halo
concept the existing shared-memory tiled kernels already use internally,
just exchanged over NVLink/PCIe instead of on-chip shared memory.

**Adding a new operator** (e.g. a wavelet transform): derive from
`HilbertOperator<T>` (`include/core/HilbertOperator.cuh`), implementing
`apply()`/`applyAdjoint()`. This was the explicit extension point the base
interface was designed around.

### Testing and validation

Every piece of math in this project was validated **before** being written
as a CUDA kernel — first as a derivation, then numerically (in Python
during development, and in a standalone C++ CPU reference implementation
that's still in this repo). This mattered: the process caught two real
bugs in the original algorithm specification this project was built from
(a boundary-condition gap in the divergence operator, and a sign error in
the primal update) — both would have silently produced a "denoiser" that
doesn't actually minimize the energy it's supposed to minimize, while
still running without crashing and producing *some* output.

**Run the CPU reference** (no GPU/CUDA needed, just a C++17 compiler):

```bash
g++ -O2 -std=c++17 devdocs/cpu_reference/cpu_reference.cpp -o cpu_reference
./cpu_reference
```

Expected: all adjoint trials pass at ~1e-15 relative error, the denoise
test shows a clear PSNR improvement, and the energy-monotonicity check
reports zero upward violations. If you ever doubt a GPU result, this is
your independent, hardware-independent oracle for the underlying math.

**Run the `RangeCheck.h` unit test** (also no GPU/CUDA needed — it's
plain host C++ with zero CUDA dependency):

```bash
g++ -O2 -std=c++17 -I include tests/test_range_check.cpp -o test_range_check
./test_range_check
```

This is built automatically as part of the normal CMake build too (target
`test_range_check`, wired into `ctest`) — the standalone g++ command above
is for when you want to check it in isolation, e.g. on a machine without
CUDA at all.

**Run the GPU test suite** after any build:

```powershell
.\test_adjoint.exe          # grayscale
.\test_color_adjoint.exe    # color
.\test_volume_adjoint.exe   # 3D volumetric
.\test_denoise.exe          # grayscale end-to-end
.\test_color_denoise.exe    # color end-to-end
.\test_volume_denoise.exe   # 3D volumetric end-to-end
.\test_range_check.exe      # RangeCheck.h, no GPU actually exercised
```

Or run everything CMake knows about at once: `ctest` from the build
directory.

**Run the Python tools/ test suite** (see [tools/](#tools-python-scripts-for-2d-image-and-3d-volume-comparison) above):

```bash
cd tools/
python -m pytest test_hctv_metrics.py test_preprocess_volume.py test_compare_volumes.py test_compare_images.py -v
```

**Re-run a specific numerical claim** made in this README or in
`devdocs/DEV_LOG.md` (e.g. "the lambda fix differs by under ~1% at the
default noise level", or "‖K‖² ≤ 8 for the 2D gradient operator"): see
`devdocs/verification/` — each script there is a committed, standalone,
re-runnable check for one specific claim (the lambda-estimation fix, the
PSNR/SSIM formulas vs an independent textbook implementation,
`RangeCheck.h`'s normalization algebra, the batch-mode summary
statistics, and the Chambolle-Pock step-size operator-norm bounds),
rather than something you'd have to re-derive by hand to double-check.
Run any of them directly:

```bash
python3 devdocs/verification/verify_lambda_fix.py
python3 devdocs/verification/verify_metrics_math.py
python3 devdocs/verification/verify_rangecheck_math.py
python3 devdocs/verification/verify_batch_stats.py
python3 devdocs/verification/verify_operator_norm.py
```

See `devdocs/DEV_LOG.md` if you want the full derivations, the specific
numerical evidence for every design decision, and a chronological account
of what was tried, verified, and (twice) found to be wrong before being
fixed. It's kept up to date as this project evolves and is the first place
to look before re-deriving something that may already be settled.

---

## Known limitations and not-yet-implemented features

- **Multi-GPU**: interface-only stub, not implemented (see above).
- **Color + 3D combined** (e.g. a color volumetric scan): not implemented.
  Each axis (color, 3D) is implemented independently; combining them would
  need a new solver generalizing both at once.
- **`--naive` flag**: only affects grayscale mode; color and volume modes
  always use the shared-memory tiled kernels.
- Run `test_color_adjoint.exe` / `test_volume_adjoint.exe`
  first and treat any failure as a real bug report, not a fluke.

---

## FAQ / troubleshooting

**My denoised image looks like a cartoon / posterized blob.** Lambda is
too high for this image's actual noise level. If you used `--lambda`
explicitly, try a smaller value. If you didn't, the auto-estimator may
have overestimated the noise (rare, but possible on images with very
strong, regular texture patterns) — pass an explicit `--lambda` (try
`0.01`–`0.05` for typical photos) to override it.

**My denoised image looks barely different from the input.** Lambda may
be too low, or `--iterations` too few for full convergence. Try raising
`--lambda` or `--iterations`.

**Build fails with a CUDA device-link error on Windows.** Make sure
`src/dummy.cu` is present and listed in `CMakeLists.txt`'s
`add_executable(HilbertCUDA-TV ...)` call — see the Architecture section
above.

**Build fails complaining it can't find `stb_image.h`.** You need to
download it once — see [Setup](#quick-start) above.

**`test_adjoint.exe` (or any adjoint test) fails on my GPU.** This means
the gradient/divergence kernels aren't computing what they should on your
specific hardware/driver combination — please don't trust any denoising
results until this passes. Check you're not running `HCTV_FAST_MATH=ON`
(that changes division precision project-wide); if you are, rebuild
without it and re-test.
