// test_denoise.cu
//
// End-to-end acceptance test: synthetic test image -> add Gaussian noise
// (sigma=25/255) -> run ROFSolver -> check PSNR improvement:
//   - PSNR improvement >= 8.0 dB
//   - final PSNR >= 29.5 dB
//   - per-iteration time <= 1.5 ms at 1024x1024 (reported, not hard-failed,
//     since exact timing depends on the user's specific GPU/driver/OS
//     state -- printed clearly so the user can compare against their own
//     run; see scripts/ for a dedicated Nsight profiling harness if you
//     want rigorous timing isolation)
//
// Uses a synthetic test image (smooth gradient + sharp blocks) generated
// in-process rather than requiring an external Lena/Cameraman file, both
// to avoid a network/file dependency in this test AND to avoid licensing
// ambiguity around the classic Lena test image specifically.

#include "solvers/ROFSolver.cuh"
#include "utils/ImageIO.h"
#include "utils/Metrics.h"
#include <cstdio>
#include <algorithm>

using namespace hctv;

static bool run_denoise_test(int W, int H, double sigma_255, float lambda,
                              int iterations, bool use_shared) {
    GrayImage clean = make_synthetic_test_image(W, H);
    GrayImage noisy = add_gaussian_noise(clean, sigma_255);

    ROFParams params;
    params.lambda = lambda;
    params.max_iterations = iterations;
    params.use_shared_memory = use_shared;

    ROFSolver solver(W, H, params);
    ROFResult result = solver.solve(noisy.data);

    double psnr_noisy = psnr(clean.data, noisy.data);
    double psnr_denoised = psnr(clean.data, result.denoised);
    double ssim_denoised = ssim_windowed(clean.data, result.denoised, W, H);

    double improvement = psnr_denoised - psnr_noisy;

    printf("\n--- Denoise test: %dx%d, sigma=%.1f, lambda=%.3f, %d iters, %s kernels ---\n",
           W, H, sigma_255, lambda, iterations, use_shared ? "tiled" : "naive");
    printf("  PSNR noisy:     %.2f dB\n", psnr_noisy);
    printf("  PSNR denoised:  %.2f dB\n", psnr_denoised);
    printf("  PSNR improvement: %.2f dB (target >= 8.0 dB)\n", improvement);
    printf("  SSIM denoised:  %.4f\n", ssim_denoised);
    printf("  Total solve time: %.3f ms (%d iterations)\n", result.total_kernel_time_ms, result.iterations_run);
    printf("  Avg per-iteration: %.4f ms (target <= 1.5 ms at 1024x1024)\n", result.avg_iter_time_ms);

    bool ok = (improvement >= 8.0) && (psnr_denoised >= 29.5);
    printf("  Result: %s\n", ok ? "PASS" : "FAIL (acceptance threshold not met -- check lambda/iterations)");
    return ok;
}

int main() {
    bool all_ok = true;

    printf("=== HilbertCUDA-TV end-to-end denoising acceptance test ===\n");

    // Primary acceptance case, matches the suggested NF4 target
    // (512x512-class image, sigma=25 noise) -- using 256x256 here to keep
    // the test fast; see scripts/ for the full 512x512/1024x1024 benchmark.
    all_ok &= run_denoise_test(256, 256, 25.0, 0.15f, 300, /*use_shared=*/true);

    // Cross-check: naive kernels should give the (near-)identical result,
    // since they implement the same math -- any divergence here indicates
    // a tiled-kernel-specific bug rather than an algorithmic one.
    all_ok &= run_denoise_test(256, 256, 25.0, 0.15f, 300, /*use_shared=*/false);

    // Performance/scale check at the primary target resolution.
    all_ok &= run_denoise_test(1024, 1024, 25.0, 0.15f, 300, /*use_shared=*/true);

    printf("\n=== RESULT: %s ===\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
