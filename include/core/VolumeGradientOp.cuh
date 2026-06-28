// VolumeGradientOp.cuh
//
// 3D forward-difference gradient/divergence (Neumann boundary). 
// Direct generalization of the proven 2D
// scalar operator (include/core/GradientOp.cuh) with a third (z)
// direction; same gating pattern, same correctness argument, independently
// re-verified (adjoint identity + operator norm bound) rather than just
// assumed to generalize.
//
// Deliberately a SEPARATE implementation from GradientOp.cuh/.cu, not a 
// modification of it.
#pragma once

#include <cuda_runtime.h>

namespace hctv {

constexpr int kVolumeTileDim = 8; // 8x8x8 blocks = 256 threads, matching the 2D 16x16=256 choice

__global__ void kernel_volume_gradient_tiled(const float* __restrict__ u,
                                              float* __restrict__ px,
                                              float* __restrict__ py,
                                              float* __restrict__ pz,
                                              int W, int H, int D);

__global__ void kernel_volume_divergence_tiled(const float* __restrict__ px,
                                                const float* __restrict__ py,
                                                const float* __restrict__ pz,
                                                float* __restrict__ div,
                                                int W, int H, int D);

class VolumeGradientOperator {
public:
    VolumeGradientOperator(int width, int height, int depth)
        : W_(width), H_(height), D_(depth) {}

    void gradient(const float* u, float* px, float* py, float* pz, cudaStream_t stream) const;
    void divergence(const float* px, const float* py, const float* pz, float* div, cudaStream_t stream) const;

    int width() const { return W_; }
    int height() const { return H_; }
    int depth() const { return D_; }

private:
    int W_, H_, D_;
};

} // namespace hctv
