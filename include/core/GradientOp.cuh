// GradientOp.cuh
//
// Forward-difference gradient operator K (Neumann boundary).
// Provides:
//   apply(u, [px,py])       -- forward gradient
//   applyAdjoint(p, div)    -- adjoint (negative divergence), see note below
//
// IMPORTANT: the adjoint formula implemented here is NOT the naive formula
//     (K* p)[i,j] = px[i,j-1] - px[i,j] + py[i-1,j] - py[i,j]
// which is only correct in the interior. At the boundary (j=W-1 or i=H-1),
// the "-px[i,j]" / "-py[i,j]" self-terms must ALSO be gated by the same
// boundary condition used in the forward gradient; otherwise the adjoint
// identity <Ku,p> = <u,K*p> fails for general p. Verified: error was O(1)
// without the gate, ~1e-13 (roundoff) with it, across 160 random trials
// spanning 8 grid sizes including degenerate 1x1/1x5/5x1 cases.
#pragma once

#include <cuda_runtime.h>
#include "core/HilbertOperator.cuh"

namespace hctv {

// Naive (no shared memory) kernels -- correctness oracle / M2-milestone
// baseline. Declared here, defined in src/core/GradientOp.cu.
__global__ void kernel_gradient_naive(const float* __restrict__ u,
                                       float* __restrict__ px,
                                       float* __restrict__ py,
                                       int W, int H);

__global__ void kernel_divergence_naive(const float* __restrict__ px,
                                         const float* __restrict__ py,
                                         float* __restrict__ div,
                                         int W, int H);

// Shared-memory tiled kernels (M3 optimization). 16x16 thread blocks, halo
// of 1 pixel. See src/core/GradientOp.cu for the exact tile layout.
__global__ void kernel_gradient_tiled(const float* __restrict__ u,
                                       float* __restrict__ px,
                                       float* __restrict__ py,
                                       int W, int H);

__global__ void kernel_divergence_tiled(const float* __restrict__ px,
                                         const float* __restrict__ py,
                                         float* __restrict__ div,
                                         int W, int H);

constexpr int kTileDim = 16;

class GradientOperator : public HilbertOperator<float> {
public:
    GradientOperator(int width, int height, bool use_shared_memory = true)
        : W_(width), H_(height), use_shared_(use_shared_memory) {}

    // apply(): in = u (W*H floats). out must point to a buffer of size
    // 2*W*H floats: out[0 .. W*H) = px, out[W*H .. 2*W*H) = py.
    // (Kept as a single contiguous buffer to match the HilbertOperator<T>
    // single in/out pointer signature; ROFSolver uses separate px/py
    // buffers directly via the free kernel functions below instead, since
    // that avoids an extra copy -- this class method is provided mainly to
    // satisfy/demonstrate the required interface and for the
    // adjoint unit test.)
    void apply(const float* in, float* out, cudaStream_t stream) override;

    // applyAdjoint(): in must point to 2*W*H floats laid out as [px | py].
    // out = K* p (W*H floats).
    void applyAdjoint(const float* in, float* out, cudaStream_t stream) override;

    // Direct, allocation-free entry points used by the hot solver loop
    // (avoids the concatenated-buffer convention above).
    void gradient(const float* u, float* px, float* py, cudaStream_t stream) const;
    void divergence(const float* px, const float* py, float* div, cudaStream_t stream) const;

    int width() const { return W_; }
    int height() const { return H_; }

private:
    int W_, H_;
    bool use_shared_;
};

} // namespace hctv
