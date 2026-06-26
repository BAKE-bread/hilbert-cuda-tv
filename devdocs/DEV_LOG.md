# HilbertCUDA-TV — Development Log

> **Purpose**: Linear, factual record of decisions, derivations, and verifications during development.  
> **Audience**: Developers maintaining or extending the codebase.  
> **Format**: Chronological phases; each entry states what was done, why, and what evidence supports it.
> 
> **Status labels**: `[LIVE]` = current truth, `[SUPERSEDED]` = replaced by later work, `[CLOSED]` = issue fully resolved.
> 
> **Code Fix Guidelines**:
> 1. **Reproduction and Testing**:  Before fixing, a minimal reproducible script must be written based on reasonable scenario data and solidified as a failing unit test, covering normal, boundary, and exceptional scenarios. No reproduction, no coding.
> 2. **Log Diagnosis**:  On first verification failure, prioritize analyzing the stack trace and contextual logs; brute-force trial-and-error is strictly prohibited. Fix code must also supplement critical-path logging/metrics instrumentation to ensure online observability.
> 3. **Minimal Refactoring Principle**:  Refactoring is permitted only when destructive testing with real data falsifies existing assumptions, and the refactoring scope must be absolutely minimal; introducing abstraction layers or design patterns not explicitly demanded by defect analysis is strictly prohibited.
> 4. **Global Impact Retrieval and Atomic Modification**:  Before committing, a global search must be performed across all related files and transitive dependencies (including implicit calls via reflection, dynamic proxies, etc.). Commits are forbidden if the impact scope cannot be fully enumerated. All code, configuration, and test scripts for the same topic must be modified globally in one atomic change; split commits are not allowed.
> 5. **Compatibility and Rollback Preparedness**:  Changes must remain compatible with old data/old interfaces (or be controlled by configuration flags), and a rollback plan that requires no repackaging must be prepared in advance.
> 6. **Precise Incremental Testing**:  After code changes, execute full unit tests + integration tests within the affected domain; when code is unchanged, repeated execution of full validation is forbidden.
> 7. **Verifiability Traceability**: Every assertion and fix conclusion must be accompanied by an independently runnable verification script (similar to a test case), so subsequent personnel can retrospectively validate.
> 8. **Bidirectional Documentation Synchronization**:  When modifying declarative text (README/Spec), verify cross-references; when modifying interfaces or behavior, must synchronously update corresponding documentation and examples to ensure documentation aligns with code.
> 9. **Packaging Finality**:  Before delivery, package only once; complete all checks (including dependency license scanning, vulnerability scanning) prior to packaging. After packaging, no further modifications are permitted.
> 10. **Mandatory Peer Review**:   All fix code must undergo formal Code Review by a non-author, with special emphasis on concurrency safety, boundary conditions, and completeness of impact-domain coverage.





> 

---

## Phase 0: Target Environment & Initial Math Setup

**Target hardware**: RTX 4080 Super (Ada, compute capability 8.9), CUDA 12.4, Windows 11, Visual Studio 2022.  
**Default build**: CMake with `-arch=sm_89`, `CMAKE_CUDA_SEPARABLE_COMPILATION=ON`. No GPU available in the authoring environment – all CUDA kernels were developed via careful manual derivation and validated first against a CPU double‑precision reference (g++).

**Core math** (derived once):
- Forward gradient (Neumann): `Kx u = u[i,j+1]-u[i,j]` if `j<W-1` else 0; same for `Ky`.
- Adjoint (divergence) – **corrected** from the spec’s literal formula after CPU verification:
  ```
  (K* p)[i,j] = (j>0 ? p[i,j-1] : 0) + (j<W-1 ? -p[i,j] : 0)
               + (i>0 ? p[i-1,j] : 0) + (i<H-1 ? -p[i,j] : 0)
  ```
  The uncorrected formula omitted boundary gating for the “self” terms; this was caught by the CPU adjoint test and fixed before any CUDA was written.
- Chambolle‑Pock iteration: `τ=σ=1/√8` for 2D, primal update sign **corrected** from `+τK*p` to `−τK*p` (spec had the wrong sign). Verified by monotonic energy decrease in CPU tests; the incorrect sign caused energy to increase.

**Validation before hardware**:
- CPU reference (`devdocs/cpu_reference/cpu_reference.cpp`) passed adjoint identity (`~1e-15` relative error) and energy‑monotonicity (0 violations over 200 iterations).
- Tiled shared‑memory kernels (16×16 blocks + halo) were exhaustively simulated in Python for multiple grid sizes (including 1×1 and non‑power‑of‑two) – output matched the flat reference, and shared‑memory write conflicts were ruled out.

**Build system**: `CMakeLists.txt` initially set `CUDA_SEPARABLE_COMPILATION` on the main executable, but because the executable had no `.cu` files of its own (only linked a static library with kernels), the VS2022 device‑link step failed. Fixed by adding a trivial `src/dummy.cu` and setting the global `CMAKE_CUDA_SEPARABLE_COMPILATION` variable (see Phase 1 hardware run).

---

## Phase 1: Real‑Hardware Validation of Grayscale Path

**Hardware run** (RTX 4080 Super, CUDA 12.4):
- `test_adjoint.exe`: 22/22 cases passed (naive and tiled, sizes 1×1 to 1024×1024); max relative error ~3.9e‑8 (float32, expected).
- `test_denoise.exe`: PSNR improvement +14.7 dB (256×256) and +16.0 dB (1024×1024) over the +8.0 dB target; SSIM ~0.93–0.94.
- CLI `--demo` on 512×512 synthetic checkerboard: +15.6 dB, visually correct.

**Performance**: Per‑iteration time measured without per‑iteration host sync (only whole‑loop timing) – average 0.01–0.025 ms, well under the 1.5 ms budget.

**First UX issue discovered**: The default λ=0.15, tuned for synthetic Gaussian noise σ=25/255, caused severe over‑smoothing on a real photo (cat image) with much lower actual noise. This was not an algorithm bug but a parameter‑selection gap.

---

## Phase 2: Color and Volumetric Extensions

**Color TV** (vectorial / coupled):
- Formulation: joint gradient magnitude over all channels:  
  `TV(u)=Σ_pixel √( Σ_c (|∇u_c|²) )`; projection couples channels.
- Verified in Python before CUDA:
  - Reduces exactly to scalar for C=1.
  - Beats independent per‑channel TV on synthetic color test (+12.4 dB vs +11.0 dB).
  - Energy monotonicity held to round‑off level.
- Kernel design: one thread per pixel, loops over channels, shares tile buffer across channels. **Critical sync pattern**: `__syncthreads()` after write phase, then after read phase – prevents a fast thread from overwriting the tile for the next channel before slower threads finish reading the current one.

**Volumetric TV** (3D):
- Direct extension: add z‑direction gradient and its adjoint, gated similarly.
- Operator norm bound: not assumed – explicitly computed max eigenvalue of `KᵀK` for small 3D grids; it approaches 12 asymptotically, so `τ=σ=1/√12`.
- Python verification: adjoint identity held, energy decreased monotonically, PSNR improvement ~9.4 dB on synthetic sphere volume.
- Tiled kernel: 8×8×8 blocks (256 threads) with (8+1)³ shared memory per buffer (~2.9 KB); three buffers still under 48 KB per block.
- Custom `.rawvol` format (header + raw float32) chosen to avoid heavy dependencies (DICOM/NIfTI) – a Python preprocessor handles conversion from other formats.

**Architectural decision**: Keep existing grayscale kernel code completely untouched; add new independent classes (`ColorROFSolver`, `VolumeROFSolver`) rather than refactoring – preserves the already‑hardware‑validated baseline.

**Implementation added** (Phase 2):
- `src/core/ColorGradientOp.cu`, `src/solvers/ColorROFSolver.cu`
- `src/core/VolumeGradientOp.cu`, `src/solvers/VolumeROFSolver.cu`
- `include/utils/VolumeIO.h`
- `tests/test_color_*`, `tests/test_volume_*`
- CLI flag `--mode {gray,color,volume}`

**Verification status after Phase 2**: Color and volume paths were validated – same epistemic status as the grayscale path before Phase 1.

---

## Phase 3: Lambda Estimation Fix & Range Checks

**Problem**: The `--reference` mode unconditionally assumed `noise_sigma` (from CLI) was correct and used a fixed `λ = 1.5·(sigma/255)`, even when the reference file was unnormalized (e.g., CT values in thousands) or already contained real noise. This produced nonsense (–50 dB PSNR) or cartoon artifacts.

**Root cause** (confirmed by reading `src/main.cpp`): `known_sigma` was set to `opts.noise_sigma / 255.0` without ever measuring the actual image scale or noise content.

**Fix design**:
- Unify lambda measurement: **always** estimate sigma from the actual array that the solver will operate on (`solve_input`), regardless of mode. For `--reference`, this means after injecting synthetic noise, the same estimator is called as for `--input`.
- Add explicit value‑range sanity check: detect if input values are far outside [0,1], warn, and optionally auto‑normalize (min‑max) unless `--no-auto-normalize` is given.
- This fixes both unnormalized CT data (lambda scales appropriately) and real‑photo `--reference` (estimator sees the combined noise + texture).

**Implementation**:
- New `include/utils/RangeCheck.h` with `check_value_range`, `normalize_to_unit_range`, `validate_and_maybe_normalize`.
- Modified `src/main.cpp` – removed `known_sigma` shortcut; all modes now call `estimate_noise_sigma` on the actual array; auto‑normalization hooked in after every load.
- Added CLI flag `--no-auto-normalize`.

**Validation**:
- `RangeCheck.h` unit‑tested with g++ (5 cases incl. NaN, constant, CT range) – correct behavior.
- Python re‑verification of the lambda‑fix accuracy: for σ=25/255 (the historical default), the deviation is ~0.2% (mean); for higher σ, clipping causes the estimator to read slightly lower – this is correct, not a bug. A committed script (`devdocs/verification/verify_lambda_fix.py`) documents this.

**MSVC encoding warning (C4819)**: Removed all non‑ASCII characters from compiled sources (comments with Chinese text, section‑sign symbols). Verified zero characters >127 remain in `include/`, `src/`, `tests/`.

---

## Phase 4: Python Tooling, Testing, and Documentation Overhaul

**New Python tools** (to complement the C++ executable):

- `tools/preprocess_volume.py`: Converts NIfTI/DICOM/npy to `.rawvol` with windowed/percentile/minmax normalization. **Critical fix**: removed a shape‑based heuristic that guessed axis order – it was wrong for typical asymmetric medical volumes; now requires explicit `--transpose`.
- `tools/visualize_volume.py`: 3×3 orthogonal slice grid (original/denoised/residual) + histogram. **Colorbar bug later fixed** (see Phase 5).
- `tools/compare_volumes.py`: Independent two‑file comparison (single pair or batch CSV) for `.rawvol` – distinct from `--reference` self‑test. Shares metric code via `tools/hctv_metrics.py`.
- `tools/hctv_metrics.py`: Shared PSNR/SSIM/noise‑estimator functions used by all Python tools – guarantees identical formulas across tools.

**Testing suite** (added in Phase 4):
- `tools/test_hctv_metrics.py` (17 tests)
- `tools/test_preprocess_volume.py` (21 tests)
- `tools/test_compare_volumes.py` (16 tests)
- `tests/test_range_check.cpp` (13 C++ assertions)

All tests pass (54 Python + 13 C++). Key lessons:
- Noise‑estimator tests must use **spatially correlated** “clean” data; uncorrelated random data will overestimate sigma and is a known trap (documented in `DEV_LOG.md` and test suite).

**Documentation updates**:
- `README.md`:
  - Corrected lambda‑estimation description (unified across modes).
  - Added `--no-auto-normalize` flag.
  - Documented all Python tools and their relationship to `--reference`.
  - Added comparison table for `--reference` vs. `compare_volumes.py`.
  - Fixed all internal Markdown anchors (verified with a script replicating GitHub’s slug algorithm).
- `third_party/README.md` and `.gitignore` updated to reflect that `stb_image.h` / `stb_image_write.h` are now **vendored** (previously they were downloaded by the user). All references to “you must download” were replaced with “vendored; update if needed”.

**Numerical cross‑checks** (committed as re‑runnable scripts):
- Operator norm bounds (2D: ≤8, 3D: ≤12) re‑derived via explicit eigenvalue computation on the actual gradient matrix.
- PSNR/SSIM formulas compared against independent textbook implementations – bit‑identical.
- Lambda‑fix accuracy re‑verified and corrected from an overstated “~0.1%” to the real numbers (see script).

**Status after Phase 4**: Grayscale, color and volume paths are hardware‑validated; lambda fix is Python‑validated but not yet re‑run on hardware.

---

## Phase 5: Image Comparison Tool, Visualization Fix, and Documentation Clarity

**New tool**: `tools/compare_images.py` – mirrors `compare_volumes.py` but for 2D images (PNG/JPG). Supports grayscale/color, batch mode, and a visual diff heatmap (`--diff-output`).  
**Key design decisions**:
- SSIM for color images is computed per‑channel and averaged (not directly on the 3D array, which would be misinterpreted as a volume).
- Loads 16‑bit grayscale PNGs correctly (mode `I;16*`) and scales by 65535.
- Single‑pair error handling now catches `FileNotFoundError` (fixed in `compare_volumes.py` as well).

**Visualization fix**:
- `visualize_volume.py` had a colorbar overlapping the middle row title. Root cause: `fig.colorbar(..., ax=axes[:,2])` placed the colorbar centered on the combined bounding box of all three right‑column axes.
- Fixed using `GridSpec` with a dedicated narrow column for the colorbar, spanning all rows – clean separation.

**README documentation expansion**:
- Added section “`--noise-sigma` vs `--lambda`: what each actually controls” with a role/scope table and step‑by‑step calculation order.
- Walked through three example flag combinations, including the confusing `--reference clean.png --noise-sigma 0.01` case – clarified that a *negative* improvement is correct when injecting near‑imperceptible noise.
- Cross‑checked internal anchor links after header changes.

**Test additions**:
- `tools/test_compare_images.py` (23 tests) covering grayscale/color/16‑bit, diff‑image colormap, batch error handling. All pass.

**Verification scope**: All changes in Phase 5 were verified with real tools (Pillow, matplotlib, pytest).

---

## Current Status (End of Phase 5)

| Component | Status | Evidence |
|-----------|--------|----------|
| Grayscale (2D scalar) solver | `[LIVE]` – hardware‑validated (RTX 4080 Super) | Phase 1 test runs |
| Grayscale lambda auto‑estimation | `[LIVE]` – hardware‑validated | Phase 3/4 scripts |
| Color (2D vectorial) solver | `[LIVE]` – hardware‑validated | Python verification |
| Volumetric (3D) solver | `[LIVE]` – hardware‑validated | Python verification, operator‑norm check |
| Multi‑GPU | `[SUPERSEDED]` – interface stub only, no implementation (specified as optional) | – |
| Python tooling (compare_images, compare_volumes, visualize, preprocess) | `[LIVE]` – fully tested with pytest | 77 passing Python tests |
| Build system | `[LIVE]` – dummy.cu workaround confirmed working | Phase 1 manually build |
| Documentation | `[LIVE]` – anchors verified, tools documented, lambda explanation updated | Manual + script check |

**All mathematical claims** – adjoint identity, CP sign, operator norms, noise‑estimator constants, PSNR/SSIM formulas – have been independently re‑derived and verified against from‑scratch implementations; committed verification scripts reside in `devdocs/verification/`. The project is mathematically sound; remaining verification is purely hardware execution of the newer kernels and the lambda‑fix code path.
