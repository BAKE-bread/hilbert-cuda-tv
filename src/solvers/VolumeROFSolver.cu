// VolumeROFSolver.cu
//
// Kernel designs verified via Python simulation before CUDA transcription
// (devdocs/DEV_LOG.md section 13). CORRECTED primal-update sign (minus,
// not plus -- see devdocs/DEV_LOG.md section 2 bug #2) and gated divergence
// boundary terms (section 2 bug #1) are both used throughout, same as the
// proven 2D/color solvers. Step size uses the INDEPENDENTLY VERIFIED 3D
// operator norm bound tau=sigma=1/sqrt(12) -- NOT 1/sqrt(8) -- confirmed
// by direct eigenvalue computation in devdocs/DEV_LOG.md section 13, not
// just pattern-matched from the 2D case.

#include "solvers/VolumeROFSolver.cuh"
#include "core/VolumeGradientOp.cuh" // for kVolumeTileDim
#include "utils/CudaCheck.cuh"
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace hctv {

// ===========================================================================
// Kernel A: dual ascent + projection (3D analog of kernel_dual_ascent_
// project_tiled in ROFSolver.cu). Same halo convention as
// kernel_volume_gradient_tiled.
// ===========================================================================
__global__ void kernel_volume_dual_ascent_project_tiled(
    float* __restrict__ px, float* __restrict__ py, float* __restrict__ pz,
    const float* __restrict__ ubar,
    int W, int H, int D, float sigma, float lambda)
{
    __shared__ float tile[kVolumeTileDim + 1][kVolumeTileDim + 1][kVolumeTileDim + 1];

    int bx = blockIdx.x * kVolumeTileDim;
    int by = blockIdx.y * kVolumeTileDim;
    int bz = blockIdx.z * kVolumeTileDim;
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gx = bx + tx, gy = by + ty, gz = bz + tz;

    tile[tz][ty][tx] = (gx < W && gy < H && gz < D) ? ubar[(size_t)gz * H * W + (size_t)gy * W + gx] : 0.0f;
    if (tx == kVolumeTileDim - 1) {
        int gx2 = bx + kVolumeTileDim;
        tile[tz][ty][kVolumeTileDim] = (gx2 < W && gy < H && gz < D) ? ubar[(size_t)gz * H * W + (size_t)gy * W + gx2] : 0.0f;
    }
    if (ty == kVolumeTileDim - 1) {
        int gy2 = by + kVolumeTileDim;
        tile[tz][kVolumeTileDim][tx] = (gx < W && gy2 < H && gz < D) ? ubar[(size_t)gz * H * W + (size_t)gy2 * W + gx] : 0.0f;
    }
    if (tz == kVolumeTileDim - 1) {
        int gz2 = bz + kVolumeTileDim;
        tile[kVolumeTileDim][ty][tx] = (gx < W && gy < H && gz2 < D) ? ubar[(size_t)gz2 * H * W + (size_t)gy * W + gx] : 0.0f;
    }
    __syncthreads();

    if (gx >= W || gy >= H || gz >= D) return;
    size_t idx = (size_t)gz * H * W + (size_t)gy * W + gx;

    float dux = (gx < W - 1) ? (tile[tz][ty][tx + 1] - tile[tz][ty][tx]) : 0.0f;
    float duy = (gy < H - 1) ? (tile[tz][ty + 1][tx] - tile[tz][ty][tx]) : 0.0f;
    float duz = (gz < D - 1) ? (tile[tz + 1][ty][tx] - tile[tz][ty][tx]) : 0.0f;

    float qx = px[idx] + sigma * dux;
    float qy = py[idx] + sigma * duy;
    float qz = pz[idx] + sigma * duz;

    float norm = sqrtf(qx * qx + qy * qy + qz * qz);
    float scale = fmaxf(1.0f, norm / lambda);

    px[idx] = qx / scale;
    py[idx] = qy / scale;
    pz[idx] = qz / scale;
}

// ===========================================================================
// Kernel B: fused divergence + primal update + extrapolation (3D analog
// of kernel_primal_update_tiled). Same halo convention as
// kernel_volume_divergence_tiled. CORRECTED SIGN: minus tau*div.
// ===========================================================================
__global__ void kernel_volume_primal_update_tiled(
    float* __restrict__ u, float* __restrict__ ubar,
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ f,
    int W, int H, int D, float tau)
{
    __shared__ float tile_px[kVolumeTileDim + 1][kVolumeTileDim + 1][kVolumeTileDim + 1];
    __shared__ float tile_py[kVolumeTileDim + 1][kVolumeTileDim + 1][kVolumeTileDim + 1];
    __shared__ float tile_pz[kVolumeTileDim + 1][kVolumeTileDim + 1][kVolumeTileDim + 1];

    int bx = blockIdx.x * kVolumeTileDim;
    int by = blockIdx.y * kVolumeTileDim;
    int bz = blockIdx.z * kVolumeTileDim;
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int lz = tz + 1, ly = ty + 1, lx = tx + 1;

    {
        int gx0 = bx + tx, gy0 = by + ty, gz0 = bz + tz;
        bool in = (gx0 < W && gy0 < H && gz0 < D);
        size_t gidx = in ? ((size_t)gz0 * H * W + (size_t)gy0 * W + gx0) : 0;
        tile_px[lz][ly][lx] = in ? px[gidx] : 0.0f;
        tile_py[lz][ly][lx] = in ? py[gidx] : 0.0f;
        tile_pz[lz][ly][lx] = in ? pz[gidx] : 0.0f;
    }
    if (tx == 0) {
        int gx0 = bx - 1, gy0 = by + ty, gz0 = bz + tz;
        bool in = (gx0 >= 0 && gx0 < W && gy0 < H && gz0 < D);
        size_t gidx = in ? ((size_t)gz0 * H * W + (size_t)gy0 * W + gx0) : 0;
        tile_px[lz][ly][0] = in ? px[gidx] : 0.0f;
        tile_py[lz][ly][0] = in ? py[gidx] : 0.0f;
        tile_pz[lz][ly][0] = in ? pz[gidx] : 0.0f;
    }
    if (ty == 0) {
        int gx0 = bx + tx, gy0 = by - 1, gz0 = bz + tz;
        bool in = (gx0 < W && gy0 >= 0 && gy0 < H && gz0 < D);
        size_t gidx = in ? ((size_t)gz0 * H * W + (size_t)gy0 * W + gx0) : 0;
        tile_px[lz][0][lx] = in ? px[gidx] : 0.0f;
        tile_py[lz][0][lx] = in ? py[gidx] : 0.0f;
        tile_pz[lz][0][lx] = in ? pz[gidx] : 0.0f;
    }
    if (tz == 0) {
        int gx0 = bx + tx, gy0 = by + ty, gz0 = bz - 1;
        bool in = (gx0 < W && gy0 < H && gz0 >= 0 && gz0 < D);
        size_t gidx = in ? ((size_t)gz0 * H * W + (size_t)gy0 * W + gx0) : 0;
        tile_px[0][ly][lx] = in ? px[gidx] : 0.0f;
        tile_py[0][ly][lx] = in ? py[gidx] : 0.0f;
        tile_pz[0][ly][lx] = in ? pz[gidx] : 0.0f;
    }
    __syncthreads();

    int gx = bx + tx, gy = by + ty, gz = bz + tz;
    if (gx >= W || gy >= H || gz >= D) return;
    size_t idx = (size_t)gz * H * W + (size_t)gy * W + gx;

    float px_self_v = tile_px[lz][ly][lx];
    float px_left_v = tile_px[lz][ly][lx - 1];
    float py_self_v = tile_py[lz][ly][lx];
    float py_up_v   = tile_py[lz][ly - 1][lx];
    float pz_self_v = tile_pz[lz][ly][lx];
    float pz_back_v = tile_pz[lz - 1][ly][lx];

    float px_left = (gx > 0)     ? px_left_v : 0.0f;
    float px_self = (gx < W - 1) ? -px_self_v : 0.0f;
    float py_up   = (gy > 0)     ? py_up_v : 0.0f;
    float py_self = (gy < H - 1) ? -py_self_v : 0.0f;
    float pz_back = (gz > 0)     ? pz_back_v : 0.0f;
    float pz_self = (gz < D - 1) ? -pz_self_v : 0.0f;

    float div = px_left + px_self + py_up + py_self + pz_back + pz_self;

    float u_old = u[idx];
    float u_new = (u_old - tau * div + tau * f[idx]) / (1.0f + tau); // CORRECTED SIGN
    u[idx] = u_new;
    ubar[idx] = 2.0f * u_new - u_old;
}

// ===========================================================================
// VolumeROFSolver class
// ===========================================================================

static inline dim3 make_volume_grid(int W, int H, int D, int tile = kVolumeTileDim) {
    return dim3((W + tile - 1) / tile, (H + tile - 1) / tile, (D + tile - 1) / tile);
}

static inline void launch_volume_iteration(
    float* d_u, float* d_ubar, float* d_px, float* d_py, float* d_pz, const float* d_f,
    int W, int H, int D, float tau, float sigma, float lambda)
{
    dim3 block(kVolumeTileDim, kVolumeTileDim, kVolumeTileDim);
    dim3 grid = make_volume_grid(W, H, D);

    kernel_volume_dual_ascent_project_tiled<<<grid, block>>>(d_px, d_py, d_pz, d_ubar, W, H, D, sigma, lambda);
    CHECK_KERNEL_LAUNCH();

    kernel_volume_primal_update_tiled<<<grid, block>>>(d_u, d_ubar, d_px, d_py, d_pz, d_f, W, H, D, tau);
    CHECK_KERNEL_LAUNCH();
}

VolumeROFSolver::VolumeROFSolver(int width, int height, int depth, const VolumeROFParams& params)
    : W_(width), H_(height), D_(depth), params_(params) {
    allocate();
    CHECK_CUDA(cudaEventCreate(&ev_start_));
    CHECK_CUDA(cudaEventCreate(&ev_stop_));
}

VolumeROFSolver::~VolumeROFSolver() {
    free_device_memory();
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_stop_) cudaEventDestroy(ev_stop_);
}

void VolumeROFSolver::allocate() {
    size_t bytes = (size_t)W_ * H_ * D_ * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_f_, bytes));
    CHECK_CUDA(cudaMalloc(&d_u_, bytes));
    CHECK_CUDA(cudaMalloc(&d_ubar_, bytes));
    CHECK_CUDA(cudaMalloc(&d_px_, bytes));
    CHECK_CUDA(cudaMalloc(&d_py_, bytes));
    CHECK_CUDA(cudaMalloc(&d_pz_, bytes));
}

void VolumeROFSolver::free_device_memory() {
    if (d_f_) cudaFree(d_f_);
    if (d_u_) cudaFree(d_u_);
    if (d_ubar_) cudaFree(d_ubar_);
    if (d_px_) cudaFree(d_px_);
    if (d_py_) cudaFree(d_py_);
    if (d_pz_) cudaFree(d_pz_);
    d_f_ = d_u_ = d_ubar_ = d_px_ = d_py_ = d_pz_ = nullptr;
}

void VolumeROFSolver::upload(const std::vector<float>& f) {
    size_t n = (size_t)W_ * H_ * D_;
    size_t bytes = n * sizeof(float);
    if (f.size() != n) {
        fprintf(stderr, "VolumeROFSolver::upload: size mismatch (expected %zu, got %zu)\n", n, f.size());
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaMemcpy(d_f_, f.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_u_, d_f_, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_ubar_, d_f_, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(d_px_, 0, bytes));
    CHECK_CUDA(cudaMemset(d_py_, 0, bytes));
    CHECK_CUDA(cudaMemset(d_pz_, 0, bytes));
}

float VolumeROFSolver::iterate_once() {
    // 3D operator norm bound: ||K||^2 <= 12 (NOT 8 -- independently
    // verified via eigenvalue computation, see devdocs/DEV_LOG.md section 13).
    const float tau = 1.0f / sqrtf(12.0f);
    const float sigma = 1.0f / sqrtf(12.0f);

    CHECK_CUDA(cudaEventRecord(ev_start_));
    launch_volume_iteration(d_u_, d_ubar_, d_px_, d_py_, d_pz_, d_f_, W_, H_, D_, tau, sigma, params_.lambda);
    CHECK_CUDA(cudaEventRecord(ev_stop_));
    CHECK_CUDA(cudaEventSynchronize(ev_stop_));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start_, ev_stop_));
    return ms;
}

std::vector<float> VolumeROFSolver::download() const {
    std::vector<float> out((size_t)W_ * H_ * D_);
    CHECK_CUDA(cudaMemcpy(out.data(), d_u_, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

VolumeROFResult VolumeROFSolver::solve(const std::vector<float>& f) {
    upload(f);

    VolumeROFResult result;
    result.width = W_;
    result.height = H_;
    result.depth = D_;

    const float tau = 1.0f / sqrtf(12.0f);
    const float sigma = 1.0f / sqrtf(12.0f);

    CHECK_CUDA(cudaEventRecord(ev_start_));
    for (int it = 0; it < params_.max_iterations; ++it) {
        launch_volume_iteration(d_u_, d_ubar_, d_px_, d_py_, d_pz_, d_f_, W_, H_, D_, tau, sigma, params_.lambda);
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
