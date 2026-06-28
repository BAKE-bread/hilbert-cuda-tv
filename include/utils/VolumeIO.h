// VolumeIO.h
//
// Minimal binary voxel format for 3D volumetric TV denoising input/output.
// The reason why a custom minimal format was chosen over 
// DICOM/NIfTI (short version: those are heavy, clinically-oriented 
// formats with parsing dependencies and metadata semantics this
// denoiser has no use for -- it only needs a 3D array of intensities).
//
// File layout (".rawvol"):
//   uint32_t magic       = 0x564C4854 ('HLTV' as a magic-ish tag, byte order
//                            native to the writing machine -- this format
//                            is NOT designed for cross-architecture
//                            portability, just local round-tripping)
//   uint32_t width  (W)
//   uint32_t height (H)
//   uint32_t depth  (D)
//   float32 voxel data, W*H*D values, row-major with z slowest-varying:
//     idx = z*H*W + y*W + x
//   (no compression, no metadata, no orientation/spacing info)
//
// To create a .rawvol file from an existing volume (e.g. a numpy array, a
// stack of DICOM slices already loaded elsewhere, a synthetic test
// volume), see the Python snippet in README.md's "secondary developer"
// section -- it's a 4-line numpy one-liner.
#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <random>

namespace hctv {

constexpr uint32_t kRawVolMagic = 0x564C4854u;

struct Volume {
    std::vector<float> data; // size W*H*D, [0,1], idx = z*H*W + y*W + x
    int width = 0;
    int height = 0;
    int depth = 0;

    size_t voxel_count() const { return (size_t)width * height * depth; }
};

inline Volume load_rawvol(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("VolumeIO: failed to open file: " + path);
    }

    uint32_t header[4];
    if (std::fread(header, sizeof(uint32_t), 4, f) != 4) {
        std::fclose(f);
        throw std::runtime_error("VolumeIO: failed to read header: " + path);
    }
    if (header[0] != kRawVolMagic) {
        std::fclose(f);
        throw std::runtime_error("VolumeIO: bad magic number in: " + path +
                                  " (not a .rawvol file, or wrong byte order)");
    }

    Volume vol;
    vol.width = static_cast<int>(header[1]);
    vol.height = static_cast<int>(header[2]);
    vol.depth = static_cast<int>(header[3]);

    if (vol.width <= 0 || vol.height <= 0 || vol.depth <= 0) {
        std::fclose(f);
        throw std::runtime_error("VolumeIO: invalid dimensions in: " + path);
    }

    size_t n = vol.voxel_count();
    vol.data.resize(n);
    size_t read = std::fread(vol.data.data(), sizeof(float), n, f);
    std::fclose(f);
    if (read != n) {
        throw std::runtime_error("VolumeIO: truncated voxel data in: " + path +
                                  " (expected " + std::to_string(n) + " floats, got " +
                                  std::to_string(read) + ")");
    }
    return vol;
}

inline void save_rawvol(const std::string& path, const Volume& vol) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("VolumeIO: failed to open file for writing: " + path);
    }
    uint32_t header[4] = {
        kRawVolMagic,
        static_cast<uint32_t>(vol.width),
        static_cast<uint32_t>(vol.height),
        static_cast<uint32_t>(vol.depth)
    };
    std::fwrite(header, sizeof(uint32_t), 4, f);
    size_t n = vol.voxel_count();
    size_t written = std::fwrite(vol.data.data(), sizeof(float), n, f);
    std::fclose(f);
    if (written != n) {
        throw std::runtime_error("VolumeIO: failed to write all voxel data to: " + path);
    }
}

// Synthetic test volume: a bright sphere in a dark background, the classic
// medical-imaging-style test case -- avoids needing a real CT/MRI dataset
// just to demo or test the volumetric path.
inline Volume make_synthetic_test_volume(int W, int H, int D) {
    Volume vol;
    vol.width = W; vol.height = H; vol.depth = D;
    vol.data.resize((size_t)W * H * D);
    double cx = W / 2.0, cy = H / 2.0, cz = D / 2.0;
    double radius = std::min({W, H, D}) * 0.3;
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                double r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                double v = (r < radius) ? 0.85 : 0.2;
                vol.data[(size_t)z * H * W + (size_t)y * W + x] = static_cast<float>(v);
            }
        }
    }
    return vol;
}

inline Volume add_gaussian_noise_volume(const Volume& vol, double sigma_255, uint32_t seed = 42) {
    Volume out = vol;
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma_255 / 255.0);
    for (auto& v : out.data) {
        double n = static_cast<double>(v) + noise(rng);
        v = static_cast<float>(std::min(1.0, std::max(0.0, n)));
    }
    return out;
}

// 3D analog of estimate_noise_sigma (ImageIO.h), using a 6-neighbor
// Laplacian stencil (the natural 3D generalization of the 4-neighbor 2D
// stencil). Same MAD-based robust estimation approach.
inline double estimate_noise_sigma_volume(const std::vector<float>& data, int W, int H, int D) {
    if (W < 2 || H < 2 || D < 2) return 0.0;
    std::vector<double> lap((size_t)W * H * D);
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t idx = (size_t)z * H * W + (size_t)y * W + x;
                double c = data[idx];
                double xm = (x > 0)     ? data[idx - 1] : c;
                double xp = (x < W - 1) ? data[idx + 1] : c;
                double ym = (y > 0)     ? data[idx - W] : c;
                double yp = (y < H - 1) ? data[idx + W] : c;
                double zm = (z > 0)     ? data[idx - (size_t)H * W] : c;
                double zp = (z < D - 1) ? data[idx + (size_t)H * W] : c;
                lap[idx] = xm + xp + ym + yp + zm + zp - 6.0 * c;
            }
        }
    }
    std::vector<double> sorted_lap = lap;
    std::sort(sorted_lap.begin(), sorted_lap.end());
    double median_lap = sorted_lap[sorted_lap.size() / 2];

    std::vector<double> abs_dev(lap.size());
    for (size_t k = 0; k < lap.size(); ++k) abs_dev[k] = std::fabs(lap[k] - median_lap);
    std::sort(abs_dev.begin(), abs_dev.end());
    double mad = abs_dev[abs_dev.size() / 2];

    // 6-neighbor stencil [1,1,1,1,1,1,-6]: gain = sqrt(6*1^2 + 6^2) = sqrt(42)
    const double kStencilGain = 6.48074069840786; // sqrt(42)
    const double kMadToStd = 0.6745;
    return std::max(0.0, mad / (kMadToStd * kStencilGain));
}

} // namespace hctv
