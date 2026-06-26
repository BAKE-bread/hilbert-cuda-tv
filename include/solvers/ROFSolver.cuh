// ROFSolver.cuh
//
// Chambolle-Pock primal-dual solver for the ROF (Rudin-Osher-Fatemi) TV
// denoising model, per spec section 3. All iteration state lives on the
// GPU; only the initial upload and final download touch the host.
//
// IMPORTANT: this solver uses the CORRECTED primal-update sign (minus, not
// plus, on the tau*K*p term) -- see devdocs/DEV_LOG.md section 2, bug #2,
// for the full derivation and numerical proof that the spec's literal
// formula (and its own demo code) has the wrong sign there.
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace hctv {

struct ROFParams {
    float lambda = 0.15f;       // TV regularization weight (image assumed in [0,1])
    int   max_iterations = 300; // outer CP iteration count
    bool  use_shared_memory = true;

    // Spec Appendix A asks for an optional debug-mode per-iteration adjoint
    // assertion. NOTE: this flag is currently a documented NO-OP -- setting
    // it to true does not change solver behavior. Wiring it up for real
    // would require a reduction kernel + host readback every iteration,
    // which would reintroduce the per-iteration host sync that
    // ROFSolver::solve()'s hot loop is specifically designed to avoid (see
    // devdocs/DEV_LOG.md section 8). Use tests/test_adjoint.cu instead --
    // it performs the equivalent check as a standalone, one-time
    // verification rather than paying that cost on every solve() call.
    bool  debug_check_adjoint_each_iter = false;
};

// Result bundle returned to the caller after solve().
struct ROFResult {
    std::vector<float> denoised; // W*H floats, [0,1]
    int width = 0;
    int height = 0;
    int iterations_run = 0;
    double total_kernel_time_ms = 0.0;   // sum of per-iteration device time (cudaEvent-measured)
    double avg_iter_time_ms = 0.0;
};

class ROFSolver {
public:
    ROFSolver(int width, int height, const ROFParams& params);
    ~ROFSolver();

    // Disable copy (owns device memory); moving is fine but not implemented
    // since this class is used as a single long-lived object in main.cpp /
    // the test files, not stored in containers.
    ROFSolver(const ROFSolver&) = delete;
    ROFSolver& operator=(const ROFSolver&) = delete;

    // f: host array, W*H floats, [0,1]. Runs params.max_iterations CP
    // iterations and returns the denoised result plus timing info.
    ROFResult solve(const std::vector<float>& f);

    // Lower-level access for tests/benchmarks: run exactly one CP iteration
    // on already-uploaded device state, returning elapsed device time in ms
    // via cudaEvent. Used by tests/test_denoise.cu and Nsight profiling
    // harnesses that want to time a single steady-state iteration without
    // host<->device transfer overhead.
    void upload(const std::vector<float>& f);
    float iterate_once();             // returns elapsed ms for this iteration
    std::vector<float> download() const;

    int width() const { return W_; }
    int height() const { return H_; }

private:
    int W_, H_;
    ROFParams params_;

    float *d_f_ = nullptr, *d_u_ = nullptr, *d_ubar_ = nullptr;
    float *d_px_ = nullptr, *d_py_ = nullptr;

    cudaEvent_t ev_start_ = nullptr, ev_stop_ = nullptr;

    void allocate();
    void free_device_memory();
};

} // namespace hctv
