// test_color_denoise.cu
//
// End-to-end color denoising acceptance test: synthetic color test image
// (distinct-colored regions, specifically so color fringing would be
// visually/numerically obvious if channel coupling were broken) -> add 
// per-channel Gaussian noise -> ColorROFSolver -> check PSNR improvement.

#include "solvers/ColorROFSolver.cuh"
#include "utils/ImageIO.h"
#include "utils/Metrics.h"
#include <cstdio>
#include <algorithm>

using namespace hctv;

static bool run_color_denoise_test(int W, int H, double sigma_255, float lambda, int iterations) {
    ColorImage clean = make_synthetic_color_test_image(W, H);
    ColorImage noisy = add_gaussian_noise_color(clean, sigma_255);

    ColorROFParams params;
    params.lambda = lambda;
    params.max_iterations = iterations;
    params.channels = 3;

    ColorROFSolver solver(W, H, params);
    ColorROFResult result = solver.solve(noisy.data);

    double psnr_noisy = psnr(clean.data, noisy.data);
    double psnr_denoised = psnr(clean.data, result.denoised);
    double improvement = psnr_denoised - psnr_noisy;

    printf("\n--- Color denoise test: %dx%d, sigma=%.1f, lambda=%.3f, %d iters ---\n",
           W, H, sigma_255, lambda, iterations);
    printf("  PSNR noisy:     %.2f dB\n", psnr_noisy);
    printf("  PSNR denoised:  %.2f dB\n", psnr_denoised);
    printf("  PSNR improvement: %.2f dB (target >= 8.0 dB)\n", improvement);
    printf("  Total solve time: %.3f ms (%d iterations)\n", result.total_kernel_time_ms, result.iterations_run);
    printf("  Avg per-iteration: %.4f ms\n", result.avg_iter_time_ms);

    bool ok = (improvement >= 8.0);
    printf("  Result: %s\n", ok ? "PASS" : "FAIL (check lambda/iterations)");
    return ok;
}

int main() {
    bool all_ok = true;
    printf("=== HilbertCUDA-TV color denoising acceptance test ===\n");

    all_ok &= run_color_denoise_test(256, 256, 25.0, 0.15f, 300);
    all_ok &= run_color_denoise_test(512, 512, 25.0, 0.15f, 300);

    printf("\n=== RESULT: %s ===\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
