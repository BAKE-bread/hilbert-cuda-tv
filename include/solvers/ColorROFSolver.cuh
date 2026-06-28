// ColorROFSolver.cuh
//
// Chambolle-Pock solver for coupled vectorial TV color denoising. 
// Structurally mirrors ROFSolver.cuh (see the file for the base scalar 
// algorithm -- both fixes are inherited here, since this solver
// reuses the same corrected primal-update sign and gated-divergence
// formula, just generalized across channels with a joint dual-norm
// projection).
//
// Max supported channel count is fixed at compile time (kMaxColorChannels)
// so the projection kernel can use small fixed-size local arrays instead
// of dynamic shared memory -- 3 (RGB) is the only value actually exercised
// today, but the constant is defined generously at 4 in case of future
// RGBA-style use, at negligible extra register cost.
#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace hctv {

constexpr int kMaxColorChannels = 4;

struct ColorROFParams {
    float lambda = 0.15f;
    int   max_iterations = 300;
    int   channels = 3; // must be <= kMaxColorChannels
};

struct ColorROFResult {
    std::vector<float> denoised; // planar, C*W*H floats, [0,1]
    int width = 0;
    int height = 0;
    int channels = 0;
    int iterations_run = 0;
    double total_kernel_time_ms = 0.0;
    double avg_iter_time_ms = 0.0;
};

class ColorROFSolver {
public:
    ColorROFSolver(int width, int height, const ColorROFParams& params);
    ~ColorROFSolver();

    ColorROFSolver(const ColorROFSolver&) = delete;
    ColorROFSolver& operator=(const ColorROFSolver&) = delete;

    // f: host array, planar layout, C*W*H floats, [0,1].
    ColorROFResult solve(const std::vector<float>& f);

    void upload(const std::vector<float>& f);
    float iterate_once(); // diagnostic/profiling entry point, see ROFSolver's equivalent
    std::vector<float> download() const;

    int width() const { return W_; }
    int height() const { return H_; }
    int channels() const { return C_; }

private:
    int W_, H_, C_;
    ColorROFParams params_;

    float *d_f_ = nullptr, *d_u_ = nullptr, *d_ubar_ = nullptr;
    float *d_px_ = nullptr, *d_py_ = nullptr; // each C*W*H floats, planar

    cudaEvent_t ev_start_ = nullptr, ev_stop_ = nullptr;

    void allocate();
    void free_device_memory();
};

} // namespace hctv
