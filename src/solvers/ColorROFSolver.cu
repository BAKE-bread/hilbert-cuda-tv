// ColorROFSolver.cu
//
// Two-pass-per-block projection kernel design verified in Python simulation
// before CUDA transcription across multiple grid sizes.
// Inherits BOTH bug fixes from the scalar solver: gated divergence boundary 
// terms, and the corrected MINUS sign on tau*K*p in the primal update.

#include "solvers/ColorROFSolver.cuh"
#include "core/ColorGradientOp.cuh" // for kColorTileDim
#include "utils/CudaCheck.cuh"
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace hctv {

// ===========================================================================
// Kernel A: dual ascent + JOINT projection across channels.
//   q_c = p_c + sigma * K ubar_c                (per channel, independent)
//   norm = sqrt( sum_c |q_c|^2 )                 (joint over ALL channels)
//   scale = max(1, norm/lambda)
//   p_c = q_c / scale                            (per channel, same scale)
//
// Two passes within the same kernel: pass 1 computes q_c for every channel
// at this thread's pixel and accumulates the joint norm; pass 2 (after all
// channels' tiles have been loaded -- note the tile is reused/aliased
// across channels, requiring careful sync placement, see file comment in
// ColorGradientOp.cu) scales and writes.
// Small fixed-size local arrays (size kMaxColorChannels) hold each
// thread's per-channel q values between the two passes -- these live in
// registers/local memory, not shared memory, so they don't add to the
// shared-memory budget.
// ===========================================================================
__global__ void kernel_color_dual_ascent_project_tiled(
    float* __restrict__ px, float* __restrict__ py,
    const float* __restrict__ ubar,
    int W, int H, int C, float sigma, float lambda)
{
    __shared__ float tile[kColorTileDim + 1][kColorTileDim + 1];

    int bx = blockIdx.x * kColorTileDim;
    int by = blockIdx.y * kColorTileDim;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gi = by + ty, gj = bx + tx;
    bool valid = (gi < H && gj < W);
    int out_idx = valid ? (gi * W + gj) : 0;
    size_t plane = (size_t)W * H;

    float qx_local[kMaxColorChannels];
    float qy_local[kMaxColorChannels];
    float norm_sq = 0.0f;

    // Pass 1: compute q_c for every channel, accumulate joint norm.
    for (int c = 0; c < C; ++c) {
        const float* ubar_c = ubar + (size_t)c * plane;

        tile[ty][tx] = valid ? ubar_c[out_idx] : 0.0f;
        if (tx == kColorTileDim - 1) {
            int gj2 = bx + kColorTileDim;
            tile[ty][kColorTileDim] = (gi < H && gj2 < W) ? ubar_c[gi * W + gj2] : 0.0f;
        }
        if (ty == kColorTileDim - 1) {
            int gi2 = by + kColorTileDim;
            tile[kColorTileDim][tx] = (gi2 < H && gj < W) ? ubar_c[gi2 * W + gj] : 0.0f;
        }
        __syncthreads();

        if (valid) {
            float dux = (gj < W - 1) ? (tile[ty][tx + 1] - tile[ty][tx]) : 0.0f;
            float duy = (gi < H - 1) ? (tile[ty + 1][tx] - tile[ty][tx]) : 0.0f;
            const float* px_c = px + (size_t)c * plane;
            const float* py_c = py + (size_t)c * plane;
            float qx = px_c[out_idx] + sigma * dux;
            float qy = py_c[out_idx] + sigma * duy;
            qx_local[c] = qx;
            qy_local[c] = qy;
            norm_sq += qx * qx + qy * qy;
        }
        __syncthreads(); // required before next channel overwrites tile
    }

    // Pass 2: scale and write (no further tile reads needed, so no extra
    // sync required here beyond what the loop above already did).
    if (valid) {
        float norm = sqrtf(norm_sq);
        float scale = fmaxf(1.0f, norm / lambda);
        for (int c = 0; c < C; ++c) {
            float* px_c = px + (size_t)c * plane;
            float* py_c = py + (size_t)c * plane;
            px_c[out_idx] = qx_local[c] / scale;
            py_c[out_idx] = qy_local[c] / scale;
        }
    }
}

// ===========================================================================
// Kernel B: per-channel divergence + primal update + extrapolation.
// Independent per channel (the projection above is the only coupling
// point) -- structurally identical to the scalar kernel_primal_update_tiled
// in ROFSolver.cu, just looped over channels with a reused tile.
// ===========================================================================
__global__ void kernel_color_primal_update_tiled(
    float* __restrict__ u, float* __restrict__ ubar,
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ f,
    int W, int H, int C, float tau)
{
    __shared__ float tile_px[kColorTileDim + 1][kColorTileDim + 1];
    __shared__ float tile_py[kColorTileDim + 1][kColorTileDim + 1];

    int bx = blockIdx.x * kColorTileDim;
    int by = blockIdx.y * kColorTileDim;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gi = by + ty, gj = bx + tx;
    bool valid = (gi < H && gj < W);
    int out_idx = valid ? (gi * W + gj) : 0;
    size_t plane = (size_t)W * H;

    for (int c = 0; c < C; ++c) {
        const float* px_c = px + (size_t)c * plane;
        const float* py_c = py + (size_t)c * plane;

        {
            int gi0 = by + ty, gj0 = bx + tx;
            tile_px[ty + 1][tx + 1] = (gi0 < H && gj0 < W) ? px_c[gi0 * W + gj0] : 0.0f;
            tile_py[ty + 1][tx + 1] = (gi0 < H && gj0 < W) ? py_c[gi0 * W + gj0] : 0.0f;
        }
        if (tx == 0) {
            int gi0 = by + ty, gj0 = bx - 1;
            tile_px[ty + 1][0] = (gi0 < H && gj0 >= 0 && gj0 < W) ? px_c[gi0 * W + gj0] : 0.0f;
            tile_py[ty + 1][0] = (gi0 < H && gj0 >= 0 && gj0 < W) ? py_c[gi0 * W + gj0] : 0.0f;
        }
        if (ty == 0) {
            int gi0 = by - 1, gj0 = bx + tx;
            tile_px[0][tx + 1] = (gi0 >= 0 && gi0 < H && gj0 < W) ? px_c[gi0 * W + gj0] : 0.0f;
            tile_py[0][tx + 1] = (gi0 >= 0 && gi0 < H && gj0 < W) ? py_c[gi0 * W + gj0] : 0.0f;
        }
        __syncthreads();

        if (valid) {
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

            float* u_c = u + (size_t)c * plane;
            float* ubar_c = ubar + (size_t)c * plane;
            const float* f_c = f + (size_t)c * plane;

            float u_old = u_c[out_idx];
            float u_new = (u_old - tau * div + tau * f_c[out_idx]) / (1.0f + tau); // CORRECTED SIGN
            u_c[out_idx] = u_new;
            ubar_c[out_idx] = 2.0f * u_new - u_old;
        }
        __syncthreads(); // required before next channel overwrites tile
    }
}

// ===========================================================================
// ColorROFSolver class
// ===========================================================================

static inline dim3 make_color_grid(int W, int H, int tile = kColorTileDim) {
    return dim3((W + tile - 1) / tile, (H + tile - 1) / tile);
}

static inline void launch_color_iteration(
    float* d_u, float* d_ubar, float* d_px, float* d_py, const float* d_f,
    int W, int H, int C, float tau, float sigma, float lambda)
{
    dim3 block(kColorTileDim, kColorTileDim);
    dim3 grid = make_color_grid(W, H);

    kernel_color_dual_ascent_project_tiled<<<grid, block>>>(d_px, d_py, d_ubar, W, H, C, sigma, lambda);
    CHECK_KERNEL_LAUNCH();

    kernel_color_primal_update_tiled<<<grid, block>>>(d_u, d_ubar, d_px, d_py, d_f, W, H, C, tau);
    CHECK_KERNEL_LAUNCH();
}

ColorROFSolver::ColorROFSolver(int width, int height, const ColorROFParams& params)
    : W_(width), H_(height), C_(params.channels), params_(params) {
    if (C_ < 1 || C_ > kMaxColorChannels) {
        fprintf(stderr, "ColorROFSolver: channels=%d out of supported range [1,%d]\n",
                C_, kMaxColorChannels);
        exit(EXIT_FAILURE);
    }
    allocate();
    CHECK_CUDA(cudaEventCreate(&ev_start_));
    CHECK_CUDA(cudaEventCreate(&ev_stop_));
}

ColorROFSolver::~ColorROFSolver() {
    free_device_memory();
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_stop_) cudaEventDestroy(ev_stop_);
}

void ColorROFSolver::allocate() {
    size_t bytes = (size_t)C_ * W_ * H_ * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_f_, bytes));
    CHECK_CUDA(cudaMalloc(&d_u_, bytes));
    CHECK_CUDA(cudaMalloc(&d_ubar_, bytes));
    CHECK_CUDA(cudaMalloc(&d_px_, bytes));
    CHECK_CUDA(cudaMalloc(&d_py_, bytes));
}

void ColorROFSolver::free_device_memory() {
    if (d_f_) cudaFree(d_f_);
    if (d_u_) cudaFree(d_u_);
    if (d_ubar_) cudaFree(d_ubar_);
    if (d_px_) cudaFree(d_px_);
    if (d_py_) cudaFree(d_py_);
    d_f_ = d_u_ = d_ubar_ = d_px_ = d_py_ = nullptr;
}

void ColorROFSolver::upload(const std::vector<float>& f) {
    size_t n = (size_t)C_ * W_ * H_;
    size_t bytes = n * sizeof(float);
    if (f.size() != n) {
        fprintf(stderr, "ColorROFSolver::upload: size mismatch (expected %zu, got %zu)\n", n, f.size());
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaMemcpy(d_f_, f.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_u_, d_f_, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_ubar_, d_f_, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(d_px_, 0, bytes));
    CHECK_CUDA(cudaMemset(d_py_, 0, bytes));
}

float ColorROFSolver::iterate_once() {
    const float tau = 1.0f / sqrtf(8.0f);
    const float sigma = 1.0f / sqrtf(8.0f);

    CHECK_CUDA(cudaEventRecord(ev_start_));
    launch_color_iteration(d_u_, d_ubar_, d_px_, d_py_, d_f_, W_, H_, C_, tau, sigma, params_.lambda);
    CHECK_CUDA(cudaEventRecord(ev_stop_));
    CHECK_CUDA(cudaEventSynchronize(ev_stop_));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start_, ev_stop_));
    return ms;
}

std::vector<float> ColorROFSolver::download() const {
    std::vector<float> out((size_t)C_ * W_ * H_);
    CHECK_CUDA(cudaMemcpy(out.data(), d_u_, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

ColorROFResult ColorROFSolver::solve(const std::vector<float>& f) {
    upload(f);

    ColorROFResult result;
    result.width = W_;
    result.height = H_;
    result.channels = C_;

    const float tau = 1.0f / sqrtf(8.0f);
    const float sigma = 1.0f / sqrtf(8.0f);

    CHECK_CUDA(cudaEventRecord(ev_start_));
    for (int it = 0; it < params_.max_iterations; ++it) {
        launch_color_iteration(d_u_, d_ubar_, d_px_, d_py_, d_f_, W_, H_, C_, tau, sigma, params_.lambda);
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
