// test_volume_adjoint.cu
//
// Verifies <Ku,p>_Y == <u,K*p> for the 3D volumetric gradient operator on
// the GPU, mirroring tests/test_adjoint.cu's methodology with a third
// dimension. // The independent operator-norm bound verification 
// (tau=sigma=1/sqrt(12)) this solver relies on is not tested here directly
// (that's a step-size choice, not an adjoint-identity property), 
// but it is documented for reference.

#include "core/VolumeGradientOp.cuh"
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

static bool run_volume_adjoint_check(int W, int H, int D, std::mt19937& rng, double& out_rel_err) {
    size_t n = (size_t)W * H * D;
    std::vector<float> u(n), p_x(n), p_y(n), p_z(n);

    std::uniform_real_distribution<float> du(0.0f, 1.0f);
    std::uniform_real_distribution<float> dp(-1.0f, 1.0f);
    for (auto& v : u) v = du(rng);
    for (auto& v : p_x) v = dp(rng);
    for (auto& v : p_y) v = dp(rng);
    for (auto& v : p_z) v = dp(rng);

    float *d_u, *d_px, *d_py, *d_pz, *d_kx, *d_ky, *d_kz, *d_kstar;
    CHECK_CUDA(cudaMalloc(&d_u, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_px, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_py, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pz, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ky, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kz, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kstar, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_u, u.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_px, p_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_py, p_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pz, p_z.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    VolumeGradientOperator op(W, H, D);
    op.gradient(d_u, d_kx, d_ky, d_kz, 0);
    op.divergence(d_px, d_py, d_pz, d_kstar, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> kx(n), ky(n), kz(n), kstar(n);
    CHECK_CUDA(cudaMemcpy(kx.data(), d_kx, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ky.data(), d_ky, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(kz.data(), d_kz, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(kstar.data(), d_kstar, n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_u); cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_kx); cudaFree(d_ky); cudaFree(d_kz); cudaFree(d_kstar);

    double lhs = inner_product(kx, p_x) + inner_product(ky, p_y) + inner_product(kz, p_z);
    double rhs = inner_product(u, kstar);

    double pnorm = std::sqrt(l2norm(p_x) * l2norm(p_x) + l2norm(p_y) * l2norm(p_y) + l2norm(p_z) * l2norm(p_z));
    double scale = std::max(l2norm(u), pnorm);
    scale = std::max(scale, 1e-12);
    double err = std::fabs(lhs - rhs);
    double rel = err / scale;
    out_rel_err = rel;

    bool ok = rel <= 1e-4;
    printf("[volume adjoint] W=%3d H=%3d D=%3d  <Ku,p>=%.6e  <u,K*p>=%.6e  rel_err=%.3e  %s\n",
           W, H, D, lhs, rhs, rel, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937 rng(555);
    bool all_ok = true;
    double max_rel = 0.0;

    printf("=== HilbertCUDA-TV volumetric (3D) GPU adjoint verification ===\n\n");

    struct { int W, H, D; } sizes[] = {
        {1, 1, 1}, {1, 1, 5}, {5, 1, 1}, {8, 8, 8}, {9, 7, 5}, {16, 16, 16},
        {17, 9, 5}, {24, 24, 24}, {32, 32, 32}, {64, 64, 16}
    };

    for (auto& sz : sizes) {
        double rel;
        all_ok &= run_volume_adjoint_check(sz.W, sz.H, sz.D, rng, rel);
        max_rel = std::max(max_rel, rel);
    }

    printf("\nmax relative adjoint error across all volume GPU trials: %.3e\n", max_rel);
    printf("=== RESULT: %s ===\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
