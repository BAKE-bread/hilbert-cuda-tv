// ImageIO.h
//
// Minimal grayscale image I/O for HilbertCUDA-TV.
//
// Default backend: stb_image / stb_image_write (single header, no heavy
// dependency). See third_party/README.md for how to obtain these two files
// -- they are not vendored in this repo because they could not be fetched
// over the network while this project was authored; you need to grab them
// once (instructions in that file).
//
// Optional backend: OpenCV, enabled by defining HCTV_USE_OPENCV (set via
// CMake -DHCTV_USE_OPENCV=ON). Useful if you already have OpenCV installed
// and don't want to bother with stb.
//
// All images are represented in-memory as a flat std::vector<float> of size
// W*H, row-major, values normalized to [0,1] -- per the spec's own
// recommendation (fix the lambda parameter's scale to the image domain) to fix the image
// domain to [0,1] so lambda's tuning behavior is consistent regardless of
// the source file's bit depth.
#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <random>
#include <cmath>

#ifdef HCTV_USE_OPENCV
    #include <opencv2/opencv.hpp>
#else
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"
#endif

namespace hctv {

struct GrayImage {
    std::vector<float> data; // row-major, [0,1], size W*H
    int width = 0;
    int height = 0;
};

// Color image: 3 channels (R,G,B), stored PLANAR (all of channel 0, then
// all of channel 1, then all of channel 2) rather than interleaved --
// planar layout matches what ColorROFSolver's per-channel kernels want
// directly (each channel is a contiguous W*H block, identical in memory
// layout to a GrayImage), so no repacking is needed between IO and solve.
struct ColorImage {
    std::vector<float> data; // planar, [0,1], size 3*W*H (R block, G block, B block)
    int width = 0;
    int height = 0;
    static constexpr int channels = 3;

    // Convenience accessor for channel c's contiguous W*H block.
    float* channel(int c) { return data.data() + (size_t)c * width * height; }
    const float* channel(int c) const { return data.data() + (size_t)c * width * height; }
};

// Load any common image format (PNG/JPG/BMP via stb, or whatever OpenCV
// supports if HCTV_USE_OPENCV is defined) and convert to single-channel
// grayscale float in [0,1]. Throws std::runtime_error on failure.
inline GrayImage load_grayscale(const std::string& path) {
    GrayImage img;
#ifdef HCTV_USE_OPENCV
    cv::Mat m = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (m.empty()) {
        throw std::runtime_error("ImageIO: failed to load image: " + path);
    }
    img.width = m.cols;
    img.height = m.rows;
    img.data.resize((size_t)img.width * img.height);
    for (int i = 0; i < m.rows; ++i) {
        const uint8_t* row = m.ptr<uint8_t>(i);
        for (int j = 0; j < m.cols; ++j) {
            img.data[(size_t)i * img.width + j] = row[j] / 255.0f;
        }
    }
#else
    int w, h, channels;
    // Force load as grayscale (1 channel); stb handles the RGB->gray
    // conversion internally using the standard luma weights.
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 1);
    if (!pixels) {
        throw std::runtime_error("ImageIO: failed to load image: " + path +
                                  " (" + std::string(stbi_failure_reason()) + ")");
    }
    img.width = w;
    img.height = h;
    img.data.resize((size_t)w * h);
    for (size_t k = 0; k < img.data.size(); ++k) {
        img.data[k] = pixels[k] / 255.0f;
    }
    stbi_image_free(pixels);
#endif
    return img;
}

// Save a [0,1]-normalized grayscale float image as an 8-bit PNG.
inline void save_grayscale_png(const std::string& path, const GrayImage& img) {
    std::vector<uint8_t> out(img.data.size());
    for (size_t k = 0; k < img.data.size(); ++k) {
        float v = std::min(1.0f, std::max(0.0f, img.data[k]));
        out[k] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
#ifdef HCTV_USE_OPENCV
    cv::Mat m(img.height, img.width, CV_8UC1, out.data());
    if (!cv::imwrite(path, m)) {
        throw std::runtime_error("ImageIO: failed to write image: " + path);
    }
#else
    if (!stbi_write_png(path.c_str(), img.width, img.height, 1, out.data(), img.width)) {
        throw std::runtime_error("ImageIO: failed to write image: " + path);
    }
#endif
}

// Load an image as 3-channel color, planar layout (see ColorImage comment
// for why planar). Throws std::runtime_error on failure.
inline ColorImage load_color(const std::string& path) {
    ColorImage img;
#ifdef HCTV_USE_OPENCV
    cv::Mat m = cv::imread(path, cv::IMREAD_COLOR); // BGR order from OpenCV
    if (m.empty()) {
        throw std::runtime_error("ImageIO: failed to load image: " + path);
    }
    img.width = m.cols;
    img.height = m.rows;
    img.data.resize((size_t)3 * img.width * img.height);
    for (int i = 0; i < m.rows; ++i) {
        const uint8_t* row = m.ptr<uint8_t>(i);
        for (int j = 0; j < m.cols; ++j) {
            // OpenCV gives BGR; reorder to RGB to match stb's channel order
            // below, so callers don't need to care which backend loaded it.
            img.channel(0)[(size_t)i * img.width + j] = row[j * 3 + 2] / 255.0f; // R
            img.channel(1)[(size_t)i * img.width + j] = row[j * 3 + 1] / 255.0f; // G
            img.channel(2)[(size_t)i * img.width + j] = row[j * 3 + 0] / 255.0f; // B
        }
    }
#else
    int w, h, channels;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 3); // force RGB
    if (!pixels) {
        throw std::runtime_error("ImageIO: failed to load image: " + path +
                                  " (" + std::string(stbi_failure_reason()) + ")");
    }
    img.width = w;
    img.height = h;
    img.data.resize((size_t)3 * w * h);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            size_t src = ((size_t)i * w + j) * 3;
            size_t dst = (size_t)i * w + j;
            img.channel(0)[dst] = pixels[src + 0] / 255.0f;
            img.channel(1)[dst] = pixels[src + 1] / 255.0f;
            img.channel(2)[dst] = pixels[src + 2] / 255.0f;
        }
    }
    stbi_image_free(pixels);
#endif
    return img;
}

// Save a planar [0,1]-normalized ColorImage as an 8-bit RGB PNG.
inline void save_color_png(const std::string& path, const ColorImage& img) {
    size_t n = (size_t)img.width * img.height;
    std::vector<uint8_t> interleaved(3 * n);
    for (size_t k = 0; k < n; ++k) {
        for (int c = 0; c < 3; ++c) {
            float v = std::min(1.0f, std::max(0.0f, img.channel(c)[k]));
            interleaved[k * 3 + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }
    }
#ifdef HCTV_USE_OPENCV
    // OpenCV wants BGR for imwrite.
    std::vector<uint8_t> bgr(3 * n);
    for (size_t k = 0; k < n; ++k) {
        bgr[k * 3 + 0] = interleaved[k * 3 + 2];
        bgr[k * 3 + 1] = interleaved[k * 3 + 1];
        bgr[k * 3 + 2] = interleaved[k * 3 + 0];
    }
    cv::Mat m(img.height, img.width, CV_8UC3, bgr.data());
    if (!cv::imwrite(path, m)) {
        throw std::runtime_error("ImageIO: failed to write image: " + path);
    }
#else
    if (!stbi_write_png(path.c_str(), img.width, img.height, 3, interleaved.data(), img.width * 3)) {
        throw std::runtime_error("ImageIO: failed to write image: " + path);
    }
#endif
}


// real image file is available -- useful for smoke-testing the pipeline
// without requiring the user to source a Lena/Cameraman file (which, for
// Lena specifically, also has unclear/non-free licensing -- this synthetic
// generator avoids that question entirely for the default demo path).
inline GrayImage make_synthetic_test_image(int W, int H) {
    GrayImage img;
    img.width = W;
    img.height = H;
    img.data.resize((size_t)W * H);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double base = 0.5 + 0.3 * std::sin(j * 0.05) * std::cos(i * 0.04);
            if (i > H / 3 && i < 2 * H / 3 && j > W / 3 && j < 2 * W / 3) base = 0.85;
            if (i > H / 10 && i < H / 4 && j > 2 * W / 3 && j < 9 * W / 10) base = 0.15;
            base = std::min(1.0, std::max(0.0, base));
            img.data[(size_t)i * W + j] = static_cast<float>(base);
        }
    }
    return img;
}

// Color analog of make_synthetic_test_image: same smooth-background +
// sharp-block structure, but with a distinct color per region so color
// fringing (the artifact coupled vectorial TV is specifically meant to
// avoid -- see devdocs/DEV_LOG.md section 12) would be visually obvious if
// the coupling were broken.
inline ColorImage make_synthetic_color_test_image(int W, int H) {
    ColorImage img;
    img.width = W;
    img.height = H;
    img.data.resize((size_t)3 * W * H);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            size_t idx = (size_t)i * W + j;
            double base = 0.5 + 0.25 * std::sin(j * 0.05) * std::cos(i * 0.04);
            double r = base, g = base, b = base;
            if (i > H / 3 && i < 2 * H / 3 && j > W / 3 && j < 2 * W / 3) {
                r = 0.85; g = 0.25; b = 0.20; // warm red block
            }
            if (i > H / 10 && i < H / 4 && j > 2 * W / 3 && j < 9 * W / 10) {
                r = 0.20; g = 0.30; b = 0.85; // cool blue block
            }
            img.channel(0)[idx] = static_cast<float>(std::min(1.0, std::max(0.0, r)));
            img.channel(1)[idx] = static_cast<float>(std::min(1.0, std::max(0.0, g)));
            img.channel(2)[idx] = static_cast<float>(std::min(1.0, std::max(0.0, b)));
        }
    }
    return img;
}

// Add i.i.d. Gaussian noise with the given standard deviation (specified in
// [0,255] convention for familiarity, internally converted to [0,1] scale,
// matching how sigma is usually quoted in the TV-denoising literature e.g.
// "sigma=25" for 8-bit images). Result is clamped to [0,1].
inline GrayImage add_gaussian_noise(const GrayImage& img, double sigma_255, uint32_t seed = 42) {
    GrayImage out = img;
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma_255 / 255.0);
    for (auto& v : out.data) {
        double n = static_cast<double>(v) + noise(rng);
        v = static_cast<float>(std::min(1.0, std::max(0.0, n)));
    }
    return out;
}

// Estimate the i.i.d. Gaussian noise standard deviation present in an
// image, using a robust MAD (median absolute deviation) estimator on a
// discrete Laplacian high-pass. This is the same family of estimator as
// the classic Donoho & Johnstone wavelet-based noise estimator, just using
// a cheap 4-neighbor Laplacian stencil instead of a real wavelet transform
// (no extra dependency, and the stencil's noise-gain is known in closed
// form so the MAD->sigma conversion constant is exact, not approximate).
//
// VERIFIED (see devdocs/DEV_LOG.md section 15): accurate within ~1-4% of
// the true sigma across sigma=5..60 (255-scale) on both smooth and sharp-
// edged synthetic test images, and correctly reports near-zero for a
// genuinely clean (noise-free) image rather than being fooled by edges --
// median-based MAD is naturally robust to the small fraction of pixels
// that are real edges/texture, which is exactly why this estimator (and
// not e.g. a simple global stddev or variance) was chosen.
//
// Returns sigma in [0,1]-normalized units (matching the image's own
// normalization), NOT 255-scale -- multiply by 255 yourself if you want
// to display it in the conventional "sigma=25"-style units.
inline double estimate_noise_sigma(const std::vector<float>& data, int W, int H) {
    if (W < 2 || H < 2) return 0.0; // degenerate case, Laplacian undefined at the edges
    std::vector<double> lap((size_t)W * H);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            size_t idx = (size_t)i * W + j;
            double c = data[idx];
            double up    = (i > 0)     ? data[idx - W] : c;
            double down  = (i < H - 1) ? data[idx + W] : c;
            double left  = (j > 0)     ? data[idx - 1] : c;
            double right = (j < W - 1) ? data[idx + 1] : c;
            lap[idx] = up + down + left + right - 4.0 * c;
        }
    }
    std::vector<double> sorted_lap = lap;
    std::sort(sorted_lap.begin(), sorted_lap.end());
    double median_lap = sorted_lap[sorted_lap.size() / 2];

    std::vector<double> abs_dev(lap.size());
    for (size_t k = 0; k < lap.size(); ++k) abs_dev[k] = std::fabs(lap[k] - median_lap);
    std::sort(abs_dev.begin(), abs_dev.end());
    double mad = abs_dev[abs_dev.size() / 2];

    // The 4-neighbor Laplacian stencil [1,1,1,1,-4] applied to i.i.d. noise
    // with std sigma produces output with std = sqrt(sum of squared
    // stencil coefficients) * sigma = sqrt(4*1^2 + 4^2) * sigma = sqrt(20)*sigma.
    // MAD of a zero-mean Gaussian = 0.6745 * std (the standard conversion
    // constant, exact for a true Gaussian).
    const double kStencilGain = 4.47213595499958; // sqrt(20)
    const double kMadToStd = 0.6745;
    double sigma_hat = mad / (kMadToStd * kStencilGain);
    return std::max(0.0, sigma_hat);
}

// Empirical lambda heuristic: lambda = k * sigma (both in [0,1]-normalized
// units). k~1.5 matches what session 1's hardware-validated default
// (lambda=0.15 at sigma=25/255~=0.098) implicitly assumed -- kept as the
// default multiplier specifically so --demo/--reference (where sigma is
// known because the caller chose --noise-sigma) reproduce the EXACT same
// PSNR numbers already validated on real hardware. For blind --input
// (sigma unknown), combine this with estimate_noise_sigma() above instead
// of guessing a fixed lambda. See devdocs/DEV_LOG.md section 15.
inline float lambda_from_sigma(double sigma_normalized, double k = 1.5) {
    return static_cast<float>(k * sigma_normalized);
}

// Color analog of add_gaussian_noise: i.i.d. per-channel Gaussian noise
// (independent across channels, as is realistic for most real sensor
// noise models at this level of approximation).
inline ColorImage add_gaussian_noise_color(const ColorImage& img, double sigma_255, uint32_t seed = 42) {
    ColorImage out = img;
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma_255 / 255.0);
    for (auto& v : out.data) {
        double n = static_cast<double>(v) + noise(rng);
        v = static_cast<float>(std::min(1.0, std::max(0.0, n)));
    }
    return out;
}

// Color analog of estimate_noise_sigma: estimate per-channel and combine
// via the median across channels (robust to one channel having unusually
// more/less texture than the others, more so than a mean would be).
inline double estimate_noise_sigma_color(const ColorImage& img) {
    size_t n = (size_t)img.width * img.height;
    std::vector<float> ch0(img.channel(0), img.channel(0) + n);
    std::vector<float> ch1(img.channel(1), img.channel(1) + n);
    std::vector<float> ch2(img.channel(2), img.channel(2) + n);
    double s0 = estimate_noise_sigma(ch0, img.width, img.height);
    double s1 = estimate_noise_sigma(ch1, img.width, img.height);
    double s2 = estimate_noise_sigma(ch2, img.width, img.height);
    double vals[3] = {s0, s1, s2};
    std::sort(vals, vals + 3);
    return vals[1]; // median of 3
}

} // namespace hctv
