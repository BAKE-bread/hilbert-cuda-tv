// VolumeGradientOp.cu
//
// Direct transcription of the 3D tile designs verified in Python
// simulation BEFORE writing this CUDA code (devdocs/DEV_LOG.md section 13,
// /tmp/verify_3d_tile_design.py and /tmp/verify_3d_div_tile.py during
// development): output matched the flat reference exactly across 6 volume
// sizes including non-block-aligned (17x9x5) and degenerate (1x1x1) cases,
// with zero shared-memory write conflicts.

#include "core/VolumeGradientOp.cuh"
#include "utils/CudaCheck.cuh"

namespace hctv {

// ===========================================================================
// Gradient: tile is (kVolumeTileDim+1)^3, local (tz,ty,tx) maps to global
// (bz*tile+tz, by*tile+ty, bx*tile+tx) -- home cell coincides with tile
// origin, extra plane/row/column at tz=tile/ty=tile/tx=tile holds the
// "look-ahead" halo for the forward difference, mirroring the proven 2D
// gradient kernel's halo convention exactly (just one more dimension).
// ===========================================================================
__global__ void kernel_volume_gradient_tiled(const float* __restrict__ u,
                                              float* __restrict__ px,
                                              float* __restrict__ py,
                                              float* __restrict__ pz,
                                              int W, int H, int D) {
    __shared__ float tile[kVolumeTileDim + 1][kVolumeTileDim + 1][kVolumeTileDim + 1];

    int bx = blockIdx.x * kVolumeTileDim;
    int by = blockIdx.y * kVolumeTileDim;
    int bz = blockIdx.z * kVolumeTileDim;
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    int gx = bx + tx, gy = by + ty, gz = bz + tz;

    tile[tz][ty][tx] = (gx < W && gy < H && gz < D) ? u[(size_t)gz * H * W + (size_t)gy * W + gx] : 0.0f;
    if (tx == kVolumeTileDim - 1) {
        int gx2 = bx + kVolumeTileDim;
        tile[tz][ty][kVolumeTileDim] = (gx2 < W && gy < H && gz < D) ? u[(size_t)gz * H * W + (size_t)gy * W + gx2] : 0.0f;
    }
    if (ty == kVolumeTileDim - 1) {
        int gy2 = by + kVolumeTileDim;
        tile[tz][kVolumeTileDim][tx] = (gx < W && gy2 < H && gz < D) ? u[(size_t)gz * H * W + (size_t)gy2 * W + gx] : 0.0f;
    }
    if (tz == kVolumeTileDim - 1) {
        int gz2 = bz + kVolumeTileDim;
        tile[kVolumeTileDim][ty][tx] = (gx < W && gy < H && gz2 < D) ? u[(size_t)gz2 * H * W + (size_t)gy * W + gx] : 0.0f;
    }
    __syncthreads();

    if (gx >= W || gy >= H || gz >= D) return;
    size_t idx = (size_t)gz * H * W + (size_t)gy * W + gx;

    px[idx] = (gx < W - 1) ? (tile[tz][ty][tx + 1] - tile[tz][ty][tx]) : 0.0f;
    py[idx] = (gy < H - 1) ? (tile[tz][ty + 1][tx] - tile[tz][ty][tx]) : 0.0f;
    pz[idx] = (gz < D - 1) ? (tile[tz + 1][ty][tx] - tile[tz][ty][tx]) : 0.0f;
}

// ===========================================================================
// Divergence: tile is ALSO (kVolumeTileDim+1)^3, but shifted the other way
// -- local (tz,ty,tx) maps to global (bz+tz-1, by+ty-1, bx+tx-1), home
// cell is local (tz+1,ty+1,tx+1). Mirrors the proven 2D divergence
// kernel's halo convention with a third "look-behind" (z) plane.
// ===========================================================================
__global__ void kernel_volume_divergence_tiled(const float* __restrict__ px,
                                                const float* __restrict__ py,
                                                const float* __restrict__ pz,
                                                float* __restrict__ div,
                                                int W, int H, int D) {
    __shared__ float tile_px[kVolumeTileDim + 1][kVolumeTileDim + 1][kVolumeTileDim + 1];
    __shared__ float tile_py[kVolumeTileDim + 1][kVolumeTileDim + 1][kVolumeTileDim + 1];
    __shared__ float tile_pz[kVolumeTileDim + 1][kVolumeTileDim + 1][kVolumeTileDim + 1];

    int bx = blockIdx.x * kVolumeTileDim;
    int by = blockIdx.y * kVolumeTileDim;
    int bz = blockIdx.z * kVolumeTileDim;
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    int lz = tz + 1, ly = ty + 1, lx = tx + 1;

    // Home cell
    {
        int gx0 = bx + tx, gy0 = by + ty, gz0 = bz + tz;
        bool in = (gx0 < W && gy0 < H && gz0 < D);
        size_t gidx = in ? ((size_t)gz0 * H * W + (size_t)gy0 * W + gx0) : 0;
        tile_px[lz][ly][lx] = in ? px[gidx] : 0.0f;
        tile_py[lz][ly][lx] = in ? py[gidx] : 0.0f;
        tile_pz[lz][ly][lx] = in ? pz[gidx] : 0.0f;
    }
    // Left halo (x-1), loaded by tx==0 threads
    if (tx == 0) {
        int gx0 = bx - 1, gy0 = by + ty, gz0 = bz + tz;
        bool in = (gx0 >= 0 && gx0 < W && gy0 < H && gz0 < D);
        size_t gidx = in ? ((size_t)gz0 * H * W + (size_t)gy0 * W + gx0) : 0;
        tile_px[lz][ly][0] = in ? px[gidx] : 0.0f;
        tile_py[lz][ly][0] = in ? py[gidx] : 0.0f;
        tile_pz[lz][ly][0] = in ? pz[gidx] : 0.0f;
    }
    // Up halo (y-1), loaded by ty==0 threads
    if (ty == 0) {
        int gx0 = bx + tx, gy0 = by - 1, gz0 = bz + tz;
        bool in = (gx0 < W && gy0 >= 0 && gy0 < H && gz0 < D);
        size_t gidx = in ? ((size_t)gz0 * H * W + (size_t)gy0 * W + gx0) : 0;
        tile_px[lz][0][lx] = in ? px[gidx] : 0.0f;
        tile_py[lz][0][lx] = in ? py[gidx] : 0.0f;
        tile_pz[lz][0][lx] = in ? pz[gidx] : 0.0f;
    }
    // Back halo (z-1), loaded by tz==0 threads
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

    div[idx] = px_left + px_self + py_up + py_self + pz_back + pz_self;
}

// ===========================================================================
// VolumeGradientOperator class wiring
// ===========================================================================

static inline dim3 make_volume_grid(int W, int H, int D, int tile = kVolumeTileDim) {
    return dim3((W + tile - 1) / tile, (H + tile - 1) / tile, (D + tile - 1) / tile);
}

void VolumeGradientOperator::gradient(const float* u, float* px, float* py, float* pz, cudaStream_t stream) const {
    dim3 block(kVolumeTileDim, kVolumeTileDim, kVolumeTileDim);
    dim3 grid = make_volume_grid(W_, H_, D_);
    kernel_volume_gradient_tiled<<<grid, block, 0, stream>>>(u, px, py, pz, W_, H_, D_);
    CHECK_KERNEL_SYNC();
}

void VolumeGradientOperator::divergence(const float* px, const float* py, const float* pz, float* div, cudaStream_t stream) const {
    dim3 block(kVolumeTileDim, kVolumeTileDim, kVolumeTileDim);
    dim3 grid = make_volume_grid(W_, H_, D_);
    kernel_volume_divergence_tiled<<<grid, block, 0, stream>>>(px, py, pz, div, W_, H_, D_);
    CHECK_KERNEL_SYNC();
}

} // namespace hctv
