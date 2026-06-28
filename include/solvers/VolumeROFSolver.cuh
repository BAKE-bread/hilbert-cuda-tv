// VolumeROFSolver.cuh
//
// Chambolle-Pock solver for 3D volumetric TV denoising
// Structurally mirrors ROFSolver.cuh with a third (z) dimension;
// inherits the same two bug fixes (gated divergence, corrected
// primal-update sign) and uses the INDEPENDENTLY VERIFIED 3D
// operator norm bound tau=sigma=1/sqrt(12), NOT 1/sqrt(8) -- this is not
// the same constant as the 2D/color solvers.
#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace hctv {

struct VolumeROFParams {
    float lambda = 0.15f;
    int   max_iterations = 300;
};

struct VolumeROFResult {
    std::vector<float> denoised; // W*H*D floats, [0,1]
    int width = 0;
    int height = 0;
    int depth = 0;
    int iterations_run = 0;
    double total_kernel_time_ms = 0.0;
    double avg_iter_time_ms = 0.0;
};

class VolumeROFSolver {
public:
    VolumeROFSolver(int width, int height, int depth, const VolumeROFParams& params);
    ~VolumeROFSolver();

    VolumeROFSolver(const VolumeROFSolver&) = delete;
    VolumeROFSolver& operator=(const VolumeROFSolver&) = delete;

    VolumeROFResult solve(const std::vector<float>& f);

    void upload(const std::vector<float>& f);
    float iterate_once();
    std::vector<float> download() const;

    int width() const { return W_; }
    int height() const { return H_; }
    int depth() const { return D_; }

private:
    int W_, H_, D_;
    VolumeROFParams params_;

    float *d_f_ = nullptr, *d_u_ = nullptr, *d_ubar_ = nullptr;
    float *d_px_ = nullptr, *d_py_ = nullptr, *d_pz_ = nullptr;

    cudaEvent_t ev_start_ = nullptr, ev_stop_ = nullptr;

    void allocate();
    void free_device_memory();
};

} // namespace hctv
