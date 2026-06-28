// MultiGpuStub.cuh
//
// Interface-only placeholder for multi-GPU distributed TV denoising.
//
// NONE of the methods below are implemented (they all currently just
// throw/abort with a clear message) -- this header exists purely to fix
// the SHAPE of the future API so that:
//   (a) calling code (CLI, tests) can be written against a stable
//       interface today, and
//   (b) a future implementation has a concrete contract to fill in,
//       without needing to renegotiate the API surface later.
//
// INTENDED DESIGN (not implemented, documented here for whoever picks
// this up later:
//   - Partition the image/volume along ROWS (2D) or Z-SLABS (3D) across
//     N available GPUs, one contiguous partition per device.
//   - Each device runs its OWN ROFSolver/VolumeROFSolver instance on its
//     partition, with a 1-pixel/1-voxel-deep HALO region mirrored from
//     its neighbor(s) -- this is the SAME halo concept the existing
//     shared-memory tiled kernels already use (see GradientOp.cu,
//     VolumeGradientOp.cu), just exchanged over NVLink/PCIe between
//     devices each iteration instead of through on-chip shared memory.
//   - Halo exchange would use cudaMemcpyPeerAsync (if NVLink/P2P access
//     is available between the devices) or a staged host-memory copy as
//     a fallback, once per CP iteration, before the dual-ascent/projection
//     kernel runs (since that kernel needs ubar's neighbor row/slab).
//   - Final result gather: each device's partition is copied back to a
//     single host-side buffer in the correct order.
//   - Open design question NOT yet resolved: whether to expose this as
//     "one MultiGpuROFSolver class that looks like ROFSolver from the
//     caller's perspective" (simpler call site, more complexity hidden
//     inside) or "a thin coordinator that owns N real ROFSolver instances
//     and drives them" (more transparent, easier to debug per-device
//     issues, but a less drop-in-compatible call site). Leaning toward
//     the former for API consistency with the existing single-GPU
//     solvers, but not committed -- flagging this explicitly rather than
//     locking in a choice without being able to prototype/test it.
#pragma once

#include <stdexcept>
#include <vector>
#include <string>

namespace hctv {

struct MultiGpuConfig {
    std::vector<int> device_ids; // CUDA device ordinals to use, e.g. {0, 1}
    bool enable_peer_access = true; // attempt cudaDeviceEnablePeerAccess between listed devices
};

// Mirrors ROFParams (include/solvers/ROFSolver.cuh) -- kept as a separate
// struct rather than reusing ROFParams directly so the multi-GPU API can
// diverge later (e.g. adding per-device tuning knobs) without disturbing
// the single-GPU solver's already-stable, hardware-validated struct.
struct MultiGpuROFParams {
    float lambda = 0.15f;
    int   max_iterations = 300;
};

// NOT IMPLEMENTED. Every method throws std::logic_error with a message
// pointing back to this file. Exists only to fix the future API shape --
// see file header for the intended design.
class MultiGpuROFSolver {
public:
    MultiGpuROFSolver(int width, int height, const MultiGpuConfig& gpu_config,
                       const MultiGpuROFParams& params) {
        (void)width; (void)height; (void)gpu_config; (void)params;
        throw std::logic_error(
            "MultiGpuROFSolver is an interface stub, not yet implemented. "
            "See include/distributed/MultiGpuStub.cuh for the intended "
            "design (row/slab partitioning + halo exchange between "
            "devices, mirroring the existing single-GPU tiled kernels' "
            "halo concept). Use ROFSolver (single GPU) for now.");
    }

    std::vector<float> solve(const std::vector<float>& /*f*/) {
        throw std::logic_error("MultiGpuROFSolver::solve not implemented.");
    }

    // Returns how many GPUs are actually visible/usable on this machine,
    // via cudaGetDeviceCount -- this part COULD be implemented trivially,
    // but is left as a stub too for interface consistency until the rest
    // of the class has a real implementation to go with it.
    static int query_available_gpu_count() {
        throw std::logic_error(
            "MultiGpuROFSolver::query_available_gpu_count not implemented. "
            "A real implementation would just call cudaGetDeviceCount().");
    }
};

} // namespace hctv
