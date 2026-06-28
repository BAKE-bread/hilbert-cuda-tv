// Metrics.h
// PSNR and SSIM computation for grayscale images stored as float arrays in
// [0,1]. Host-side, single-threaded (image sizes here are small enough that
// this is not a bottleneck relative to the GPU solve itself).
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace hctv {

// Peak signal-to-noise ratio between two images of identical size, assuming
// both are normalized to [0,1] (peak = 1.0).
inline double psnr(const std::vector<float>& a, const std::vector<float>& b, double peak = 1.0) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double mse = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
        double d = static_cast<double>(a[k]) - static_cast<double>(b[k]);
        mse += d * d;
    }
    mse /= static_cast<double>(a.size());
    if (mse <= 1e-20) return 100.0; // identical images
    return 10.0 * std::log10(peak * peak / mse);
}

// Structural similarity index (SSIM), single-scale, global (whole-image)
// formulation per Wang et al. 2004, using global mean/variance/covariance
// rather than the typical 11x11 sliding Gaussian window. This is a
// deliberate simplification: it is fast, dependency-free, and is what
// is needed as a scalar pass/fail metric. It will read slightly 
// differently from windowed-SSIM implementations (e.g. scikit-image's), 
// so don't directly compare numbers across tools without
// accounting for this. Documented here and in README so nobody is
// surprised by a different SSIM value from a different library.
inline double ssim_global(const std::vector<float>& a, const std::vector<float>& b, double dynamic_range = 1.0) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    size_t n = a.size();
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t k = 0; k < n; ++k) { mean_a += a[k]; mean_b += b[k]; }
    mean_a /= n; mean_b /= n;

    double var_a = 0.0, var_b = 0.0, cov = 0.0;
    for (size_t k = 0; k < n; ++k) {
        double da = a[k] - mean_a;
        double db = b[k] - mean_b;
        var_a += da * da;
        var_b += db * db;
        cov += da * db;
    }
    var_a /= (n - 1);
    var_b /= (n - 1);
    cov /= (n - 1);

    const double C1 = (0.01 * dynamic_range) * (0.01 * dynamic_range);
    const double C2 = (0.03 * dynamic_range) * (0.03 * dynamic_range);

    double numerator = (2.0 * mean_a * mean_b + C1) * (2.0 * cov + C2);
    double denominator = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);
    return numerator / denominator;
}

// Windowed SSIM, computed over non-overlapping tiles (default 8x8), then
// averaged. Closer in spirit to standard SSIM than the pure-global version
// above, still without needing a Gaussian-blur dependency. Use this one for
// the acceptance-test numbers in tests/test_denoise.cu; use ssim_global only
// as a quick sanity scalar.
inline double ssim_windowed(const std::vector<float>& a, const std::vector<float>& b,
                             int W, int H, int tile = 8, double dynamic_range = 1.0) {
    if ((int)a.size() != W * H || (int)b.size() != W * H) return 0.0;
    double sum = 0.0;
    int count = 0;
    for (int ty = 0; ty < H; ty += tile) {
        for (int tx = 0; tx < W; tx += tile) {
            int h = std::min(tile, H - ty);
            int w = std::min(tile, W - tx);
            int n = h * w;
            if (n < 2) continue;
            double mean_a = 0.0, mean_b = 0.0;
            for (int dy = 0; dy < h; ++dy)
                for (int dx = 0; dx < w; ++dx) {
                    size_t idx = (size_t)(ty + dy) * W + (tx + dx);
                    mean_a += a[idx];
                    mean_b += b[idx];
                }
            mean_a /= n; mean_b /= n;
            double var_a = 0.0, var_b = 0.0, cov = 0.0;
            for (int dy = 0; dy < h; ++dy)
                for (int dx = 0; dx < w; ++dx) {
                    size_t idx = (size_t)(ty + dy) * W + (tx + dx);
                    double da = a[idx] - mean_a;
                    double db = b[idx] - mean_b;
                    var_a += da * da;
                    var_b += db * db;
                    cov += da * db;
                }
            var_a /= (n - 1); var_b /= (n - 1); cov /= (n - 1);
            const double C1 = (0.01 * dynamic_range) * (0.01 * dynamic_range);
            const double C2 = (0.03 * dynamic_range) * (0.03 * dynamic_range);
            double num = (2.0 * mean_a * mean_b + C1) * (2.0 * cov + C2);
            double den = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);
            sum += num / den;
            count++;
        }
    }
    return count > 0 ? sum / count : 0.0;
}

} // namespace hctv
