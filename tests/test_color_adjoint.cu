// test_color_adjoint.cu
//
// Verifies <Ku,p>_Y == <u,K*p> for the multi-channel color gradient
// operator on the GPU, mirroring tests/test_adjoint.cu's methodology but
// generalized across channels. Since the gradient/divergence are
// per-channel-independent (only the PROJECTION couples channels, and the
// projection isn't part of the adjoint relation being tested here), this
// reduces to running the scalar adjoint check independently per channel --
// but doing it through the actual ColorGradientOperator/kernel_color_*
// code path (not the proven scalar kernels) so it specifically exercises
// the channel-loop + shared-memory-tile-reuse logic unique to the color
// kernels.

#include "core/ColorGradientOp.cuh"
#include "utils/CudaCheck.cuh"
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>

using namespace hctv;

static double inner_product(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t k = 0; k < a.size(); ++k) s += (double)a[k] * (double)b[k];
    return s;
}
static double l2norm(const std::vector<float>& a) {
    return std::sqrt(inner_product(a, a));
}

static bool run_color_adjoint_check(int W, int H, int C, std::mt19937& rng, double& out_rel_err) {
    size_t plane = (size_t)W * H;
    size_t n = (size_t)C * plane;
    std::vector<float> u(n), p_x(n), p_y(n);

    std::uniform_real_distribution<float> du(0.0f, 1.0f);
    std::uniform_real_distribution<float> dp(-1.0f, 1.0f);
    for (auto& v : u) v = du(rng);
    for (auto& v : p_x) v = dp(rng);
    for (auto& v : p_y) v = dp(rng);

    float *d_u, *d_px, *d_py, *d_kx, *d_ky, *d_kstar;
    CHECK_CUDA(cudaMalloc(&d_u, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_px, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_py, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ky, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kstar, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_u, u.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_px, p_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_py, p_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    ColorGradientOperator op(W, H, C);
    op.gradient(d_u, d_kx, d_ky, 0);
    op.divergence(d_px, d_py, d_kstar, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> kx(n), ky(n), kstar(n);
    CHECK_CUDA(cudaMemcpy(kx.data(), d_kx, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ky.data(), d_ky, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(kstar.data(), d_kstar, n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_u); cudaFree(d_px); cudaFree(d_py);
    cudaFree(d_kx); cudaFree(d_ky); cudaFree(d_kstar);

    // Sum the inner product ACROSS all channels jointly (the adjoint
    // identity holds for the full multi-channel operator, summed over c).
    double lhs = inner_product(kx, p_x) + inner_product(ky, p_y);
    double rhs = inner_product(u, kstar);

    double scale = std::max(l2norm(u), std::sqrt(l2norm(p_x) * l2norm(p_x) + l2norm(p_y) * l2norm(p_y)));
    scale = std::max(scale, 1e-12);
    double err = std::fabs(lhs - rhs);
    double rel = err / scale;
    out_rel_err = rel;

    bool ok = rel <= 1e-4; // same float32 tolerance as the scalar test
    printf("[color adjoint] W=%4d H=%4d C=%d  <Ku,p>=%.6e  <u,K*p>=%.6e  rel_err=%.3e  %s\n",
           W, H, C, lhs, rhs, rel, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937 rng(321);
    bool all_ok = true;
    double max_rel = 0.0;

    printf("=== HilbertCUDA-TV color (multi-channel) GPU adjoint verification ===\n");
    printf("(exercises the channel-loop + shared-tile-reuse logic specific to\n");
    printf(" the color kernels)\n\n");

    struct { int W, H, C; } sizes[] = {
        {1, 1, 1}, {1, 1, 3}, {1, 5, 3}, {5, 1, 3}, {16, 16, 3}, {17, 16, 3},
        {31, 31, 3}, {64, 64, 3}, {127, 129, 3}, {256, 256, 3}, {64, 64, 1}, {32, 32, 4}
    };

    for (auto& sz : sizes) {
        double rel;
        all_ok &= run_color_adjoint_check(sz.W, sz.H, sz.C, rng, rel);
        max_rel = std::max(max_rel, rel);
    }

    printf("\nmax relative adjoint error across all color GPU trials: %.3e\n", max_rel);
    printf("=== RESULT: %s ===\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
