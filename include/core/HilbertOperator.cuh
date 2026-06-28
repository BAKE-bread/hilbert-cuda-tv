// HilbertOperator.cuh
//
// Abstract base interface for operators on (discretized) Hilbert spaces. 
// Kept literally as specified: apply() / applyAdjoint() virtual
// methods, host-side dispatch only (no device-side virtual calls --
// this avoids the classic CUDA "virtual function table doesn't exist on
// device" class of bugs entirely, since vtable dispatch happens on the host
// before launching the actual kernel).
//
// This interface is the extension point this abstraction was designed
// around (a uniform entry point for future extensions such as wavelet
// transforms or convolutions) -- e.g. a future WaveletOperator or
// ConvolutionOperator would derive from HilbertOperator<float> the same way
// GradientOperator/DivergenceOperator do.
#pragma once

#include <cuda_runtime.h>

namespace hctv {

template <typename T>
class HilbertOperator {
public:
    // Apply the forward operator: out = K(in). `in`/`out` are device
    // pointers. `stream` allows overlap with other work; pass 0 for the
    // default stream.
    virtual void apply(const T* in, T* out, cudaStream_t stream) = 0;

    // Apply the adjoint operator: out = K*(in). Same calling convention.
    virtual void applyAdjoint(const T* in, T* out, cudaStream_t stream) = 0;

    virtual ~HilbertOperator() = default;
};

} // namespace hctv
