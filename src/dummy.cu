// dummy.cu
//
// This file is intentionally near-empty. Its only purpose is to give the
// HilbertCUDA-TV executable target at least one genuine CUDA (.cu)
// translation unit of its own.
//
// Without it, the target had zero .cu sources but still requested
// CUDA_SEPARABLE_COMPILATION (needed because it links against hctv_core,
// which DOES use relocatable device code / __global__ symbols). On
// Windows with the Visual Studio 2022 CMake generator + CUDA 12.4, this
// combination caused the generated project's device-link step to be
// misconfigured. Confirmed fix (by actually building on real hardware,
// see devdocs/DEV_LOG.md section 11): add this file to the executable's
// source list, and also set the global CMAKE_CUDA_SEPARABLE_COMPILATION
// variable in CMakeLists.txt in addition to the per-target property.
//
// If you ever delete this file, re-add an equivalent placeholder rather
// than just removing it from CMakeLists.txt -- the executable target
// needs *some* .cu file to stay buildable on Windows/VS2022 per the above.

__global__ void hctv_dummy_kernel() {
    // Intentionally empty. Never launched by any code path; exists only
    // so this translation unit counts as a real CUDA source file.
}
