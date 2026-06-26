cmake_minimum_required(VERSION 3.18)

# CMake >= 3.18 is required for first-class CUDA_ARCHITECTURES support.
# Project targets: CUDA 12.4 toolkit, RTX 4080 Super (Ada Lovelace, compute
# capability 8.9), Windows 11 + Visual Studio 2022 generator.

project(HilbertCUDA-TV LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Global separable-compilation switch, in addition to the per-target
# property set below. Required on Windows + VS2022 + CUDA 12.4: a target
# with zero .cu sources of its own but CUDA_SEPARABLE_COMPILATION requested
# (because it links a static lib built with relocatable device code) was
# observed to misconfigure the generated project's device-link step.
# Confirmed fix via an actual build on real hardware -- see
# devdocs/DEV_LOG.md section 11 for the full story. Keeping both the
# global and per-target settings is redundant but harmless, and matches
# exactly what was verified to build cleanly.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# sm_89 = Ada Lovelace (RTX 40-series). If you build on a different GPU,
# override with -DCMAKE_CUDA_ARCHITECTURES=<your_cc> at configure time
# (e.g. 86 for RTX 30-series/Ampere, 75 for RTX 20-series/Turing).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

option(HCTV_USE_OPENCV "Use OpenCV for image I/O instead of stb_image" OFF)
option(HCTV_BUILD_TESTS "Build the tests/ executables" ON)
option(HCTV_DEBUG_CHECKS "Enable extra device-side sync + error checks (slower)" OFF)

if(HCTV_DEBUG_CHECKS)
    add_compile_definitions(HCTV_DEBUG)
endif()

# ---------------------------------------------------------------------------
# Core library: operators + solver (no main(), shared by the CLI tool and
# both test executables)
# ---------------------------------------------------------------------------
add_library(hctv_core STATIC
    src/core/GradientOp.cu
    src/solvers/ROFSolver.cu
    src/core/ColorGradientOp.cu
    src/solvers/ColorROFSolver.cu
    src/core/VolumeGradientOp.cu
    src/solvers/VolumeROFSolver.cu
)

target_include_directories(hctv_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(hctv_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# -lineinfo helps Nsight Compute / Nsight Systems map kernel hotspots back
# to source lines; cheap to always include, negligible runtime cost.
target_compile_options(hctv_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# --use_fast_math is intentionally NOT enabled by default. It does not just
# speed up sqrtf() in the TV projection -- it also makes EVERY division in
# every kernel approximate (--prec-div=false), including the primal
# update's `/(1.0f+tau)` and the projection's `qx/scale`, which are on the
# numerically-sensitive convergence path. Since this project's CUDA code
# could not be compiled or profiled in the authoring sandbox (no GPU/nvcc
# available there), enabling a global precision-reducing flag without
# being able to verify its effect on the (carefully fixed, see
# devdocs/DEV_LOG.md section 2) convergence behavior would be an
# unverified gamble. If you want the speedup and are willing to re-run
# tests/test_adjoint and tests/test_denoise to confirm nothing regresses,
# enable it explicitly:
#   cmake .. -DHCTV_FAST_MATH=ON
option(HCTV_FAST_MATH "Enable --use_fast_math (changes ALL division/sqrt precision -- re-run tests after enabling)" OFF)
if(HCTV_FAST_MATH)
    target_compile_options(hctv_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    message(WARNING "HCTV_FAST_MATH=ON: division/sqrt precision reduced project-wide. Re-run test_adjoint and test_denoise to confirm tolerances still pass.")
endif()

if(HCTV_USE_OPENCV)
    find_package(OpenCV REQUIRED)
    target_link_libraries(hctv_core PUBLIC ${OpenCV_LIBS})
    target_include_directories(hctv_core PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(hctv_core PUBLIC HCTV_USE_OPENCV)
else()
    # stb_image / stb_image_write are header-only and already vendored in
    # third_party/ -- just need the include path. See third_party/README.md
    # if you ever need to re-fetch or update them yourself.
    target_include_directories(hctv_core PUBLIC ${CMAKE_SOURCE_DIR}/third_party)
endif()

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
# src/dummy.cu: see devdocs/DEV_LOG.md section 11 for why this is here --
# short version: this target needs at least one real .cu file of its own
# for CUDA_SEPARABLE_COMPILATION to configure correctly with the VS2022
# generator, even though all the actual kernel code lives in hctv_core.
add_executable(HilbertCUDA-TV src/main.cpp src/dummy.cu)
target_link_libraries(HilbertCUDA-TV PRIVATE hctv_core)
set_target_properties(HilbertCUDA-TV PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(HCTV_BUILD_TESTS)
    add_executable(test_adjoint tests/test_adjoint.cu)
    target_link_libraries(test_adjoint PRIVATE hctv_core)

    add_executable(test_denoise tests/test_denoise.cu)
    target_link_libraries(test_denoise PRIVATE hctv_core)
    # NOTE: no need to repeat target_compile_definitions(... HCTV_USE_OPENCV)
    # here -- it's declared PUBLIC on hctv_core above (line ~76), so it
    # already propagates to anything that links against hctv_core,
    # including this target and the main CLI executable.

    add_executable(test_color_adjoint tests/test_color_adjoint.cu)
    target_link_libraries(test_color_adjoint PRIVATE hctv_core)

    add_executable(test_color_denoise tests/test_color_denoise.cu)
    target_link_libraries(test_color_denoise PRIVATE hctv_core)

    add_executable(test_volume_adjoint tests/test_volume_adjoint.cu)
    target_link_libraries(test_volume_adjoint PRIVATE hctv_core)

    add_executable(test_volume_denoise tests/test_volume_denoise.cu)
    target_link_libraries(test_volume_denoise PRIVATE hctv_core)

    # test_range_check.cpp is plain host C++ (RangeCheck.h has zero CUDA
    # dependency) -- compiled as CXX, not CUDA, so it builds even in a
    # CUDA-less environment (e.g. CI without a GPU, or this project's own
    # authoring sandbox). Mirrors devdocs/cpu_reference/cpu_reference.cpp's
    # "g++-only" validation pattern, but committed as a real test target
    # instead of a throwaway manual check (see devdocs/DEV_LOG.md section 29).
    add_executable(test_range_check tests/test_range_check.cpp)
    target_include_directories(test_range_check PRIVATE ${CMAKE_SOURCE_DIR}/include)

    enable_testing()
    add_test(NAME AdjointTest COMMAND test_adjoint)
    add_test(NAME DenoiseTest COMMAND test_denoise)
    add_test(NAME ColorAdjointTest COMMAND test_color_adjoint)
    add_test(NAME ColorDenoiseTest COMMAND test_color_denoise)
    add_test(NAME VolumeAdjointTest COMMAND test_volume_adjoint)
    add_test(NAME VolumeDenoiseTest COMMAND test_volume_denoise)
    add_test(NAME RangeCheckTest COMMAND test_range_check)
endif()

# ---------------------------------------------------------------------------
# Helpful configure-time summary
# ---------------------------------------------------------------------------
message(STATUS "HilbertCUDA-TV configuration:")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Use OpenCV         : ${HCTV_USE_OPENCV}")
message(STATUS "  Build tests        : ${HCTV_BUILD_TESTS} (scalar, color, volume)")
message(STATUS "  Debug checks       : ${HCTV_DEBUG_CHECKS}")
if(NOT HCTV_USE_OPENCV)
    message(STATUS "  NOTE: stb_image.h / stb_image_write.h are vendored in")
    message(STATUS "        third_party/ -- see third_party/README.md if you")
    message(STATUS "        ever need to re-fetch or update them")
endif()
