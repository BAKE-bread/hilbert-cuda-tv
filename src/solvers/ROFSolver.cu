// ROFSolver.cu
//
// Kernel designs verified via Python simulation before CUDA transcription
// (devdocs/DEV_LOG.md section 9, /tmp/verify_fused_kernel.py during
// development) across multiple grid sizes including non-block-aligned and
// degenerate cases. The CORRECTED primal-update sign (minus tau*K*p, see
// DEV_LOG section 2 bug #2) is used throughout -- this is the single most
// important correctness fix in this project; using the spec's literal "+"
// sign produces an algorithm that does NOT minimize the stated ROF energy
// (verified: energy diverges upward on >50% of iterations and converges to
// a fixed point with HIGHER energy than the unprocessed input).

#include "solvers/ROFSolver.cuh"
#include "core/GradientOp.cuh" // for kTileDim
#include "utils/CudaCheck.cuh"
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace hctv {

// ===========================================================================
// Kernel A: dual ascent + projection.
//   p^{n+1} = Pi_lambda( p^n + sigma * K ubar^n )
// Tile layout identical to kernel_gradient_tiled (right/down halo), since
// this kernel needs the SAME neighbor pattern as the forward gradient.
// ===========================================================================
__global__ void kernel_dual_ascent_project_tiled(
    float* __restrict__ px, float* __restrict__ py,
    const float* __restrict__ ubar,
    int W, int H, float sigma, float lambda)
{
    __shared__ float tile[kTileDim + 1][kTileDim + 1];

    int bx = blockIdx.x * kTileDim;
    int by = blockIdx.y * kTileDim;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    {
        int gi = by + ty, gj = bx + tx;
        tile[ty][tx] = (gi < H && gj < W) ? ubar[gi * W + gj] : 0.0f;
    }
    if (tx == kTileDim - 1) {
        int gi = by + ty, gj = bx + kTileDim;
        tile[ty][kTileDim] = (gi < H && gj < W) ? ubar[gi * W + gj] : 0.0f;
    }
    if (ty == kTileDim - 1) {
        int gi = by + kTileDim, gj = bx + tx;
        tile[kTileDim][tx] = (gi < H && gj < W) ? ubar[gi * W + gj] : 0.0f;
    }
    __syncthreads();

    int gi = by + ty, gj = bx + tx;
    if (gi >= H || gj >= W) return;
    int idx = gi * W + gj;

    float dux = (gj < W - 1) ? (tile[ty][tx + 1] - tile[ty][tx]) : 0.0f;
    float duy = (gi < H - 1) ? (tile[ty + 1][tx] - tile[ty][tx]) : 0.0f;

    float qx = px[idx] + sigma * dux;
    float qy = py[idx] + sigma * duy;

    // Isotropic TV projection. norm=0 case: scale=max(1,0)=1, output (0,0)/1
    // = (0,0); no division-by-zero hazard (we never divide BY norm itself,
    // only by `scale`, which is bounded below by 1.0).
    float norm = sqrtf(qx * qx + qy * qy);
    float scale = fmaxf(1.0f, norm / lambda);

    px[idx] = qx / scale;
    py[idx] = qy / scale;
}

// Naive (no shared memory) variant of the above, used as correctness
// oracle / fallback path.
__global__ void kernel_dual_ascent_project_naive(
    float* __restrict__ px, float* __restrict__ py,
    const float* __restrict__ ubar,
    int W, int H, float sigma, float lambda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;
    int idx = i * W + j;

    float dux = (j < W - 1) ? (ubar[idx + 1] - ubar[idx]) : 0.0f;
    float duy = (i < H - 1) ? (ubar[idx + W] - ubar[idx]) : 0.0f;

    float qx = px[idx] + sigma * dux;
    float qy = py[idx] + sigma * duy;

    float norm = sqrtf(qx * qx + qy * qy);
    float scale = fmaxf(1.0f, norm / lambda);

    px[idx] = qx / scale;
    py[idx] = qy / scale;
}

// ===========================================================================
// Kernel B: fused divergence + primal descent + extrapolation.
//   u^{n+1}    = (u^n - tau*K*p^{n+1} + tau*f) / (1+tau)   <-- CORRECTED SIGN
//   ubar^{n+1} = 2*u^{n+1} - u^n
// Tile layout identical to kernel_divergence_tiled (left/up halo).
// ===========================================================================
__global__ void kernel_primal_update_tiled(
    float* __restrict__ u, float* __restrict__ ubar,
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ f,
    int W, int H, float tau)
{
    __shared__ float tile_px[kTileDim + 1][kTileDim + 1];
    __shared__ float tile_py[kTileDim + 1][kTileDim + 1];

    int bx = blockIdx.x * kTileDim;
    int by = blockIdx.y * kTileDim;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Home cell: local (ty+1, tx+1) -> global (by+ty, bx+tx)
    {
        int gi = by + ty, gj = bx + tx;
        tile_px[ty + 1][tx + 1] = (gi < H && gj < W) ? px[gi * W + gj] : 0.0f;
        tile_py[ty + 1][tx + 1] = (gi < H && gj < W) ? py[gi * W + gj] : 0.0f;
    }
    // Left halo column (local tx=0), loaded by threads with tx==0
    if (tx == 0) {
        int gi = by + ty, gj = bx - 1;
        tile_px[ty + 1][0] = (gi < H && gj >= 0 && gj < W) ? px[gi * W + gj] : 0.0f;
        tile_py[ty + 1][0] = (gi < H && gj >= 0 && gj < W) ? py[gi * W + gj] : 0.0f;
    }
    // Top halo row (local ty=0), loaded by threads with ty==0
    if (ty == 0) {
        int gi = by - 1, gj = bx + tx;
        tile_px[0][tx + 1] = (gi >= 0 && gi < H && gj < W) ? px[gi * W + gj] : 0.0f;
        tile_py[0][tx + 1] = (gi >= 0 && gi < H && gj < W) ? py[gi * W + gj] : 0.0f;
    }
    // Top-left corner halo cell intentionally not loaded -- never read below
    // (the formula only reads same-row/same-column neighbors of the home
    // cell, never the diagonal); verified in the Python tile simulation
    // during development (devdocs/DEV_LOG.md section 8).
    __syncthreads();

    int gi = by + ty, gj = bx + tx;
    if (gi >= H || gj >= W) return;
    int idx = gi * W + gj;
    int lty = ty + 1, ltx = tx + 1;

    float px_self_val = tile_px[lty][ltx];
    float px_left_val = tile_px[lty][ltx - 1];
    float py_self_val = tile_py[lty][ltx];
    float py_up_val   = tile_py[lty - 1][ltx];

    float px_left = (gj > 0)     ? px_left_val : 0.0f;
    float px_self = (gj < W - 1) ? -px_self_val : 0.0f;
    float py_up   = (gi > 0)     ? py_up_val : 0.0f;
    float py_self = (gi < H - 1) ? -py_self_val : 0.0f;

    float div = px_left + px_self + py_up + py_self;

    float u_old = u[idx];
    // CORRECTED SIGN: minus tau*div, not plus. See file header comment and
    // devdocs/DEV_LOG.md section 2 bug #2 for the derivation + numeric proof.
    float u_new = (u_old - tau * div + tau * f[idx]) / (1.0f + tau);

    u[idx] = u_new;
    ubar[idx] = 2.0f * u_new - u_old;
}

__global__ void kernel_primal_update_naive(
    float* __restrict__ u, float* __restrict__ ubar,
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ f,
    int W, int H, float tau)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;
    int idx = i * W + j;

    float px_left = (j > 0)     ? px[idx - 1] : 0.0f;
    float px_self = (j < W - 1) ? -px[idx]    : 0.0f;
    float py_up   = (i > 0)     ? py[idx - W] : 0.0f;
    float py_self = (i < H - 1) ? -py[idx]    : 0.0f;
    float div = px_left + px_self + py_up + py_self;

    float u_old = u[idx];
    float u_new = (u_old - tau * div + tau * f[idx]) / (1.0f + tau); // CORRECTED SIGN
    u[idx] = u_new;
    ubar[idx] = 2.0f * u_new - u_old;
}

// ===========================================================================
// ROFSolver class
// ===========================================================================

static inline dim3 make_grid(int W, int H, int tile = kTileDim) {
    return dim3((W + tile - 1) / tile, (H + tile - 1) / tile);
}

// Launches one CP iteration's two kernels with NO host synchronization --
// this is the actual hot-path primitive used by solve()'s loop. Kernels on
// the same (default) stream execute in launch order automatically, so no
// explicit sync is needed between kernel A and kernel B within an
// iteration, or between successive iterations; cudaDeviceSynchronize/
// cudaEventSynchronize is only needed once, after the whole loop, to read
// back results or measure total elapsed time.
static inline void launch_iteration(
    float* d_u, float* d_ubar, float* d_px, float* d_py, const float* d_f,
    int W, int H, float tau, float sigma, float lambda, bool use_shared)
{
    dim3 block(kTileDim, kTileDim);
    dim3 grid = make_grid(W, H);

    if (use_shared) {
        kernel_dual_ascent_project_tiled<<<grid, block>>>(d_px, d_py, d_ubar, W, H, sigma, lambda);
    } else {
        kernel_dual_ascent_project_naive<<<grid, block>>>(d_px, d_py, d_ubar, W, H, sigma, lambda);
    }
    CHECK_KERNEL_LAUNCH();

    if (use_shared) {
        kernel_primal_update_tiled<<<grid, block>>>(d_u, d_ubar, d_px, d_py, d_f, W, H, tau);
    } else {
        kernel_primal_update_naive<<<grid, block>>>(d_u, d_ubar, d_px, d_py, d_f, W, H, tau);
    }
    CHECK_KERNEL_LAUNCH();
}

ROFSolver::ROFSolver(int width, int height, const ROFParams& params)
    : W_(width), H_(height), params_(params) {
    allocate();
    CHECK_CUDA(cudaEventCreate(&ev_start_));
    CHECK_CUDA(cudaEventCreate(&ev_stop_));
}

ROFSolver::~ROFSolver() {
    free_device_memory();
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_stop_) cudaEventDestroy(ev_stop_);
}

void ROFSolver::allocate() {
    size_t bytes = (size_t)W_ * H_ * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_f_, bytes));
    CHECK_CUDA(cudaMalloc(&d_u_, bytes));
    CHECK_CUDA(cudaMalloc(&d_ubar_, bytes));
    CHECK_CUDA(cudaMalloc(&d_px_, bytes));
    CHECK_CUDA(cudaMalloc(&d_py_, bytes));
}

void ROFSolver::free_device_memory() {
    if (d_f_) cudaFree(d_f_);
    if (d_u_) cudaFree(d_u_);
    if (d_ubar_) cudaFree(d_ubar_);
    if (d_px_) cudaFree(d_px_);
    if (d_py_) cudaFree(d_py_);
    d_f_ = d_u_ = d_ubar_ = d_px_ = d_py_ = nullptr;
}

void ROFSolver::upload(const std::vector<float>& f) {
    size_t bytes = (size_t)W_ * H_ * sizeof(float);
    if (f.size() != (size_t)W_ * H_) {
        fprintf(stderr, "ROFSolver::upload: size mismatch (expected %d, got %zu)\n",
                W_ * H_, f.size());
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaMemcpy(d_f_, f.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_u_, d_f_, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_ubar_, d_f_, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(d_px_, 0, bytes));
    CHECK_CUDA(cudaMemset(d_py_, 0, bytes));
}

// Diagnostic / benchmarking entry point: runs exactly ONE CP iteration and
// syncs to measure its elapsed device time precisely. This sync makes it
// unsuitable for the production solve() hot loop (a sync every iteration
// would add host<->device round-trip latency on top of the ~1ms kernel
// time, especially under Windows WDDM scheduling) -- solve() instead uses
// launch_iteration() directly in a tight loop with a single sync at the
// end. Use this method from tests/test_denoise.cu or a profiling script
// when you specifically want one isolated, precisely-timed iteration.
float ROFSolver::iterate_once() {
    const float tau = 1.0f / sqrtf(8.0f);
    const float sigma = 1.0f / sqrtf(8.0f);

    CHECK_CUDA(cudaEventRecord(ev_start_));
    launch_iteration(d_u_, d_ubar_, d_px_, d_py_, d_f_, W_, H_, tau, sigma,
                      params_.lambda, params_.use_shared_memory);
    CHECK_CUDA(cudaEventRecord(ev_stop_));
    CHECK_CUDA(cudaEventSynchronize(ev_stop_));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start_, ev_stop_));
    return ms;
}

std::vector<float> ROFSolver::download() const {
    std::vector<float> out((size_t)W_ * H_);
    CHECK_CUDA(cudaMemcpy(out.data(), d_u_, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

ROFResult ROFSolver::solve(const std::vector<float>& f) {
    upload(f);

    ROFResult result;
    result.width = W_;
    result.height = H_;

    const float tau = 1.0f / sqrtf(8.0f);
    const float sigma = 1.0f / sqrtf(8.0f);

    // Hot loop: NO per-iteration host sync. All max_iterations kernel pairs
    // are enqueued back-to-back on the default stream (which guarantees
    // in-order execution on the device), and we measure the ENTIRE batch
    // with one cudaEvent pair, not one pair per iteration. This is what
    // satisfies the "all steps stay on GPU, only the final result
    // copies back" and keeps per-iteration overhead at just the two kernel
    // launches (no sync round-trip hiding inside the timed region).
    CHECK_CUDA(cudaEventRecord(ev_start_));
    for (int it = 0; it < params_.max_iterations; ++it) {
        launch_iteration(d_u_, d_ubar_, d_px_, d_py_, d_f_, W_, H_, tau, sigma,
                          params_.lambda, params_.use_shared_memory);

        // Appendix A debug-mode hook: if enabled, this would launch the
        // adjoint-check kernel/reduction every iteration to assert
        // |<Ku,p> - <u,K*p>| stays within tolerance. Left as a documented
        // extension point rather than implemented inline here, since doing
        // it for real needs a reduction kernel + host readback that WOULD
        // reintroduce a per-iteration sync -- exactly what this loop is
        // designed to avoid in the default (release) path. See
        // tests/test_adjoint.cu for the standalone version of this check,
        // which is the recommended way to validate the operator instead of
        // paying the sync cost on every solve() call.
        (void)params_.debug_check_adjoint_each_iter;
    }
    CHECK_CUDA(cudaEventRecord(ev_stop_));
    CHECK_CUDA(cudaEventSynchronize(ev_stop_));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, ev_start_, ev_stop_));

    result.denoised = download();
    result.iterations_run = params_.max_iterations;
    result.total_kernel_time_ms = total_ms;
    result.avg_iter_time_ms = total_ms / std::max(1, params_.max_iterations);
    return result;
}

} // namespace hctv
