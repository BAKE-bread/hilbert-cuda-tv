// ColorGradientOp.cu
//
// Per-channel-looped tiled kernels, verified in Python simulation before
// CUDA transcription.
// Design: loop over channels INSIDE the kernel, reusing the SAME shared
// memory tile buffer for each channel sequentially -- shared memory usage
// stays identical to the scalar 2D kernel regardless of channel count C,
// at the cost of C sequential load+sync phases per block instead of 1.
// This is the safer choice for an unverified-on-real-hardware-yet feature:
// it reuses the EXACT tile-loading logic already proven for the scalar
// case, per channel, rather than deriving a new wider-tile layout.

#include "core/ColorGradientOp.cuh"
#include "utils/CudaCheck.cuh"

namespace hctv {

__global__ void kernel_color_gradient_tiled(const float* __restrict__ u,
                                             float* __restrict__ px,
                                             float* __restrict__ py,
                                             int W, int H, int C) {
    __shared__ float tile[kColorTileDim + 1][kColorTileDim + 1];

    int bx = blockIdx.x * kColorTileDim;
    int by = blockIdx.y * kColorTileDim;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gi = by + ty, gj = bx + tx;
    bool valid = (gi < H && gj < W);
    int out_idx = valid ? (gi * W + gj) : 0;
    size_t plane = (size_t)W * H;

    for (int c = 0; c < C; ++c) {
        const float* u_c = u + (size_t)c * plane;

        tile[ty][tx] = valid ? u_c[out_idx] : 0.0f;
        if (tx == kColorTileDim - 1) {
            int gj2 = bx + kColorTileDim;
            tile[ty][kColorTileDim] = (gi < H && gj2 < W) ? u_c[gi * W + gj2] : 0.0f;
        }
        if (ty == kColorTileDim - 1) {
            int gi2 = by + kColorTileDim;
            tile[kColorTileDim][tx] = (gi2 < H && gj < W) ? u_c[gi2 * W + gj] : 0.0f;
        }
        __syncthreads();

        if (valid) {
            float* px_c = px + (size_t)c * plane;
            float* py_c = py + (size_t)c * plane;
            px_c[out_idx] = (gj < W - 1) ? (tile[ty][tx + 1] - tile[ty][tx]) : 0.0f;
            py_c[out_idx] = (gi < H - 1) ? (tile[ty + 1][tx] - tile[ty][tx]) : 0.0f;
        }
        // Second syncthreads before the NEXT channel's loads overwrite the
        // shared tile -- without this, a fast thread could start writing
        // channel c+1's tile while a slow thread in the SAME block is
        // still reading channel c's tile values above. Necessary precisely
        // because the tile buffer is being reused/aliased across loop
        // iterations (the scalar kernel's single-channel version never
        // needed this second sync, since it has no loop).
        __syncthreads();
    }
}

__global__ void kernel_color_divergence_tiled(const float* __restrict__ px,
                                               const float* __restrict__ py,
                                               float* __restrict__ div,
                                               int W, int H, int C) {
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

            float* div_c = div + (size_t)c * plane;
            div_c[out_idx] = px_left + px_self + py_up + py_self;
        }
        __syncthreads(); // see comment in kernel_color_gradient_tiled
    }
}

static inline dim3 make_color_grid(int W, int H, int tile = kColorTileDim) {
    return dim3((W + tile - 1) / tile, (H + tile - 1) / tile);
}

void ColorGradientOperator::gradient(const float* u, float* px, float* py, cudaStream_t stream) const {
    dim3 block(kColorTileDim, kColorTileDim);
    dim3 grid = make_color_grid(W_, H_);
    kernel_color_gradient_tiled<<<grid, block, 0, stream>>>(u, px, py, W_, H_, C_);
    CHECK_KERNEL_SYNC();
}

void ColorGradientOperator::divergence(const float* px, const float* py, float* div, cudaStream_t stream) const {
    dim3 block(kColorTileDim, kColorTileDim);
    dim3 grid = make_color_grid(W_, H_);
    kernel_color_divergence_tiled<<<grid, block, 0, stream>>>(px, py, div, W_, H_, C_);
    CHECK_KERNEL_SYNC();
}

} // namespace hctv
