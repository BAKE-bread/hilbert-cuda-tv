// GradientOp.cu
//
// See include/core/GradientOp.cuh for the adjoint-formula correction note.
// Tile layouts below were derived and exhaustively verified in Python
// simulation against a reference implementation BEFORE being transcribed
// here (devdocs/DEV_LOG.md section 2, /tmp/tile_design*.py during
// development) -- this sandbox has no nvcc/GPU, so that simulation is the
// closest available substitute for compiling and running the actual
// kernels. The index arithmetic below is a direct transcription of the
// verified Python logic; nothing here is "new" math.

#include "core/GradientOp.cuh"
#include "utils/CudaCheck.cuh"
#include <cstdio>

namespace hctv {

// ===========================================================================
// Naive kernels: 1 thread per pixel, all boundary handling via ternary gates
// directly on global memory. No shared memory. Used as correctness oracle
// and as the M2-milestone baseline before the M3 shared-memory optimization.
// ===========================================================================

__global__ void kernel_gradient_naive(const float* __restrict__ u,
                                       float* __restrict__ px,
                                       float* __restrict__ py,
                                       int W, int H) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;

    int idx = i * W + j;
    px[idx] = (j < W - 1) ? (u[idx + 1] - u[idx]) : 0.0f;
    py[idx] = (i < H - 1) ? (u[idx + W] - u[idx]) : 0.0f;
}

// Adjoint (negative divergence). See header comment: both self-terms are
// gated by the SAME boundary condition as the corresponding forward
// difference, which is the fix for the spec's unconditional-self-term bug.
__global__ void kernel_divergence_naive(const float* __restrict__ px,
                                         const float* __restrict__ py,
                                         float* __restrict__ div,
                                         int W, int H) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;

    int idx = i * W + j;
    float px_left = (j > 0)     ? px[idx - 1] : 0.0f;
    float px_self = (j < W - 1) ? -px[idx]    : 0.0f; // gated, NOT unconditional
    float py_up   = (i > 0)     ? py[idx - W] : 0.0f;
    float py_self = (i < H - 1) ? -py[idx]    : 0.0f; // gated, NOT unconditional
    div[idx] = px_left + px_self + py_up + py_self;
}

// ===========================================================================
// Tiled (shared-memory ghost-zone) kernels: 16x16 blocks, 1-pixel halo.
//
// Gradient tile layout: tile is (kTileDim+1) x (kTileDim+1). Local index
// (ty,tx) maps to global pixel (by*kTileDim+ty, bx*kTileDim+tx) -- i.e. the
// tile's origin coincides with the block's home pixel, and the EXTRA row
// (ty=kTileDim) / column (tx=kTileDim) is the "look-ahead" halo needed for
// the forward-difference neighbor. Threads with local ty,tx < kTileDim
// compute one output pixel each; the halo cells are loaded by an extra
// strided pass since (kTileDim+1)^2 > kTileDim^2 (289 vs 256 cells to load
// with 256 threads).
//
// Divergence tile layout: tile is ALSO (kTileDim+1)x(kTileDim+1), but
// shifted the OTHER way: local (ty,tx) maps to global
// (by*kTileDim+ty-1, bx*kTileDim+tx-1) -- i.e. the halo is at ty=0/tx=0
// (top-left, "look-behind"), and a thread's own home pixel is local index
// (ty+1, tx+1). This matches what backward-difference-style divergence
// needs (left/up neighbors) instead of the gradient's right/down need.
// ===========================================================================

__global__ void kernel_gradient_tiled(const float* __restrict__ u,
                                       float* __restrict__ px,
                                       float* __restrict__ py,
                                       int W, int H) {
    __shared__ float tile[kTileDim + 1][kTileDim + 1];

    int bx = blockIdx.x * kTileDim;
    int by = blockIdx.y * kTileDim;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each thread loads its "home" cell (ty,tx) -> global (by+ty, bx+tx).
    {
        int gi = by + ty;
        int gj = bx + tx;
        tile[ty][tx] = (gi < H && gj < W) ? u[gi * W + gj] : 0.0f;
    }
    // Load the extra halo column (tx == kTileDim) using the threads in the
    // last column of the block (tx == kTileDim-1 take on the extra work),
    // and the extra halo row (ty == kTileDim) similarly. We use the first
    // row/column of threads to do this strided extra load so every halo
    // cell is covered by exactly one thread, avoiding redundant work and
    // bank-conflict-prone overlapping writes.
    if (tx == kTileDim - 1) {
        int gi = by + ty;
        int gj = bx + kTileDim;
        tile[ty][kTileDim] = (gi < H && gj < W) ? u[gi * W + gj] : 0.0f;
    }
    if (ty == kTileDim - 1) {
        int gi = by + kTileDim;
        int gj = bx + tx;
        tile[kTileDim][tx] = (gi < H && gj < W) ? u[gi * W + gj] : 0.0f;
    }
    // Corner cell (kTileDim, kTileDim) -- only needed if both a right AND
    // bottom neighbor tile exist; not actually read by any output pixel in
    // this tile (px needs (ty,tx+1) same row; py needs (ty+1,tx) same col;
    // neither reads the diagonal corner), so we skip loading it.

    __syncthreads();

    int gi = by + ty;
    int gj = bx + tx;
    if (gi >= H || gj >= W) return;

    int idx = gi * W + gj;
    px[idx] = (gj < W - 1) ? (tile[ty][tx + 1] - tile[ty][tx]) : 0.0f;
    py[idx] = (gi < H - 1) ? (tile[ty + 1][tx] - tile[ty][tx]) : 0.0f;
}

__global__ void kernel_divergence_tiled(const float* __restrict__ px,
                                         const float* __restrict__ py,
                                         float* __restrict__ div,
                                         int W, int H) {
    // +1 padding on the leading dimension is a common bank-conflict
    // mitigation (a known risk for this access pattern); since our
    // access pattern here is already row-major contiguous within a row and
    // the halo column avoids power-of-two stride issues for this tile size,
    // we keep the array un-padded but flag this as the first thing to
    // profile with Nsight if NCU reports bank conflicts (see scripts/).
    __shared__ float tile_px[kTileDim + 1][kTileDim + 1];
    __shared__ float tile_py[kTileDim + 1][kTileDim + 1];

    int bx = blockIdx.x * kTileDim;
    int by = blockIdx.y * kTileDim;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Local (ty,tx) maps to global (by+ty-1, bx+tx-1): home pixel for this
    // thread is local (ty+1, tx+1).
    auto load_at = [&](int lty, int ltx, float* tile_dst, const float* src) {
        int gi = by + lty - 1;
        int gj = bx + ltx - 1;
        tile_dst[lty * (kTileDim + 1) + ltx] =
            (gi >= 0 && gi < H && gj >= 0 && gj < W) ? src[gi * W + gj] : 0.0f;
    };

    // Home cell: local (ty+1, tx+1)
    load_at(ty + 1, tx + 1, &tile_px[0][0], px);
    load_at(ty + 1, tx + 1, &tile_py[0][0], py);

    // Left halo column (ltx = 0), loaded by threads with tx == 0
    if (tx == 0) {
        load_at(ty + 1, 0, &tile_px[0][0], px);
        load_at(ty + 1, 0, &tile_py[0][0], py);
    }
    // Top halo row (lty = 0), loaded by threads with ty == 0
    if (ty == 0) {
        load_at(0, tx + 1, &tile_px[0][0], px);
        load_at(0, tx + 1, &tile_py[0][0], py);
    }
    // Top-left corner halo cell (0,0) -- not actually read by the formula
    // below (we only ever read (lty,ltx-1) and (lty-1,ltx) relative to the
    // home cell, never the diagonal), so it is intentionally left
    // unloaded, matching the gradient kernel's symmetric omission above.

    __syncthreads();

    int gi = by + ty;
    int gj = bx + tx;
    if (gi >= H || gj >= W) return;

    int idx = gi * W + gj;
    int lty = ty + 1, ltx = tx + 1; // home position in the shared tile

    float px_self_val = tile_px[lty][ltx];
    float px_left_val = tile_px[lty][ltx - 1];
    float py_self_val = tile_py[lty][ltx];
    float py_up_val   = tile_py[lty - 1][ltx];

    float px_left = (gj > 0)     ? px_left_val : 0.0f;
    float px_self = (gj < W - 1) ? -px_self_val : 0.0f;
    float py_up   = (gi > 0)     ? py_up_val : 0.0f;
    float py_self = (gi < H - 1) ? -py_self_val : 0.0f;

    div[idx] = px_left + px_self + py_up + py_self;
}

// ===========================================================================
// GradientOperator class wiring
// ===========================================================================

static inline dim3 make_grid(int W, int H, int tile = kTileDim) {
    return dim3((W + tile - 1) / tile, (H + tile - 1) / tile);
}

void GradientOperator::gradient(const float* u, float* px, float* py, cudaStream_t stream) const {
    dim3 block(kTileDim, kTileDim);
    dim3 grid = make_grid(W_, H_);
    if (use_shared_) {
        kernel_gradient_tiled<<<grid, block, 0, stream>>>(u, px, py, W_, H_);
    } else {
        kernel_gradient_naive<<<grid, block, 0, stream>>>(u, px, py, W_, H_);
    }
    CHECK_KERNEL_SYNC();
}

void GradientOperator::divergence(const float* px, const float* py, float* div, cudaStream_t stream) const {
    dim3 block(kTileDim, kTileDim);
    dim3 grid = make_grid(W_, H_);
    if (use_shared_) {
        kernel_divergence_tiled<<<grid, block, 0, stream>>>(px, py, div, W_, H_);
    } else {
        kernel_divergence_naive<<<grid, block, 0, stream>>>(px, py, div, W_, H_);
    }
    CHECK_KERNEL_SYNC();
}

void GradientOperator::apply(const float* in, float* out, cudaStream_t stream) {
    // out[0..W*H) = px, out[W*H..2*W*H) = py
    size_t n = (size_t)W_ * H_;
    gradient(in, out, out + n, stream);
}

void GradientOperator::applyAdjoint(const float* in, float* out, cudaStream_t stream) {
    // in[0..W*H) = px, in[W*H..2*W*H) = py
    size_t n = (size_t)W_ * H_;
    divergence(in, in + n, out, stream);
}

} // namespace hctv
