// DivergenceOp.cuh
//
// A `DivergenceOperator` class distinct from `GradientOperator`.
// We implement the actual kernels jointly in GradientOp.cu/.cuh (they're
// an adjoint pair sharing tile machinery. This header provides a
// thin wrapper with the exact class name and HilbertOperator<float>
// interface, for anyone integrating against that name directly. Internally
// it just forwards to GradientOperator's divergence()/gradient() methods,
// so there's exactly one implementation of the kernels to keep in sync.
#pragma once

#include "core/HilbertOperator.cuh"
#include "core/GradientOp.cuh"

namespace hctv {

class DivergenceOperator : public HilbertOperator<float> {
public:
    DivergenceOperator(int width, int height, bool use_shared_memory = true)
        : grad_(width, height, use_shared_memory) {}

    // apply(): in = [px | py] (2*W*H floats), out = div(p) (W*H floats).
    // This is K* applied directly (divergence is the "forward" direction
    // for this wrapper class, matching its name).
    void apply(const float* in, float* out, cudaStream_t stream) override {
        size_t n = (size_t)grad_.width() * grad_.height();
        grad_.divergence(in, in + n, out, stream);
    }

    // applyAdjoint(): the adjoint of divergence is (negative of) the
    // gradient operator: in = u (W*H floats), out = [px | py] (2*W*H).
    void applyAdjoint(const float* in, float* out, cudaStream_t stream) override {
        size_t n = (size_t)grad_.width() * grad_.height();
        grad_.gradient(in, out, out + n, stream);
    }

    int width() const { return grad_.width(); }
    int height() const { return grad_.height(); }

private:
    GradientOperator grad_;
};

} // namespace hctv
