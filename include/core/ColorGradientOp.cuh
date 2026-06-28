// ColorGradientOp.cuh
//
// Vector-valued (coupled) gradient/divergence for C-channel color images. 
// The gradient and divergence per channel are IDENTICAL to the scalar 2D case
// (include/core/GradientOp.cuh) -- only the projection step (in 
// ColorROFSolver) couples channels through a joint norm. This header only
// adds the per-channel gradient/divergence kernels generalized to operate
// on C contiguous planar channel blocks in one launch, rather than 
// coupling anything itself.
//
// Deliberately NOT reusing/modifying include/core/GradientOp.cuh or its kernels.
#pragma once

#include <cuda_runtime.h>

namespace hctv {

constexpr int kColorTileDim = 16; // matches the scalar 2D tile size

// Computes the gradient for ALL C channels in one launch. Channels are
// planar (each channel is a contiguous W*H block), matching ColorImage's
// memory layout. u, px, py each point to C*W*H floats (channel c's block
// starts at offset c*W*H). One thread per (pixel, all channels) -- each
// thread loops over C inside itself, since C is small (3 typically) and
// looping keeps the kernel launch configuration identical to the scalar
// case (same grid/block dims as the proven 2D kernels).
__global__ void kernel_color_gradient_tiled(const float* __restrict__ u,
                                             float* __restrict__ px,
                                             float* __restrict__ py,
                                             int W, int H, int C);

__global__ void kernel_color_divergence_tiled(const float* __restrict__ px,
                                               const float* __restrict__ py,
                                               float* __restrict__ div,
                                               int W, int H, int C);

class ColorGradientOperator {
public:
    ColorGradientOperator(int width, int height, int channels)
        : W_(width), H_(height), C_(channels) {}

    // u, px, py: each C*W*H floats, planar layout.
    void gradient(const float* u, float* px, float* py, cudaStream_t stream) const;
    void divergence(const float* px, const float* py, float* div, cudaStream_t stream) const;

    int width() const { return W_; }
    int height() const { return H_; }
    int channels() const { return C_; }

private:
    int W_, H_, C_;
};

} // namespace hctv
