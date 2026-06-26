// CudaCheck.cuh
// Error-checking macros for CUDA runtime API calls and kernel launches.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap any CUDA runtime call. On failure, prints file/line/error string and
// aborts. Use for cudaMalloc, cudaMemcpy, cudaMemset, cudaFree, cudaMemcpy*,
// stream/event creation, etc.
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t _hctv_err = (call);                                       \
        if (_hctv_err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(_hctv_err));                           \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

// Call immediately after a kernel launch to catch launch-configuration
// errors (invalid grid/block dims, too many resources requested, etc).
// This does NOT catch asynchronous execution errors inside the kernel; for
// those, follow with CHECK_CUDA(cudaDeviceSynchronize()) in debug builds.
#define CHECK_KERNEL_LAUNCH()                                                 \
    do {                                                                      \
        cudaError_t _hctv_launch_err = cudaGetLastError();                    \
        if (_hctv_launch_err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA Kernel Launch Error @ %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(_hctv_launch_err)); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// In debug builds (HCTV_DEBUG defined), synchronize and check for async
// errors after every kernel launch. This is comparatively expensive (forces
// a full pipeline drain) so it is compiled out in Release builds.
#ifdef HCTV_DEBUG
    #define CHECK_KERNEL_SYNC()                                               \
        do {                                                                  \
            CHECK_KERNEL_LAUNCH();                                            \
            CHECK_CUDA(cudaDeviceSynchronize());                              \
        } while (0)
#else
    #define CHECK_KERNEL_SYNC() CHECK_KERNEL_LAUNCH()
#endif
