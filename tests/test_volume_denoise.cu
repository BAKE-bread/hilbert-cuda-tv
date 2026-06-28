// test_volume_denoise.cu
//
// End-to-end 3D volumetric denoising acceptance test: synthetic sphere-in-
// background test volume -> add Gaussian noise -> VolumeROFSolver ->
// check PSNR improvement. Uses the INDEPENDENTLY VERIFIED 3D step size 
// internally (tau=sigma=1/sqrt(12)), not the 2D constant -- if this
// test fails by converging to visible blocky/noisy artifacts despite the
// adjoint test passing, the step size is the first thing to double
// check (see VolumeROFSolver.cu comments).

#include "solvers/VolumeROFSolver.cuh"
#include "utils/VolumeIO.h"
#include "utils/Metrics.h"
#include <cstdio>
#include <algorithm>

using namespace hctv;

static bool run_volume_denoise_test(int W, int H, int D, double sigma_255, float lambda, int iterations) {
    Volume clean = make_synthetic_test_volume(W, H, D);
    Volume noisy = add_gaussian_noise_volume(clean, sigma_255);

    VolumeROFParams params;
    params.lambda = lambda;
    params.max_iterations = iterations;

    VolumeROFSolver solver(W, H, D, params);
    VolumeROFResult result = solver.solve(noisy.data);

    double psnr_noisy = psnr(clean.data, noisy.data);
    double psnr_denoised = psnr(clean.data, result.denoised);
    double improvement = psnr_denoised - psnr_noisy;

    printf("\n--- Volume denoise test: %dx%dx%d, sigma=%.1f, lambda=%.3f, %d iters ---\n",
           W, H, D, sigma_255, lambda, iterations);
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
    printf("=== HilbertCUDA-TV volumetric (3D) denoising acceptance test ===\n");

    all_ok &= run_volume_denoise_test(32, 32, 32, 25.0, 0.15f, 300);
    all_ok &= run_volume_denoise_test(64, 64, 64, 25.0, 0.15f, 300);

    printf("\n=== RESULT: %s ===\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
