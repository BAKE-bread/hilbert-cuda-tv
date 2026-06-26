# run_nsight_profile.ps1
#
# Convenience wrapper around Nsight Systems (nsys) and Nsight Compute (ncu)
# for profiling the ROF solver kernels, per spec section 9 (性能分析与调优).
#
# Prerequisites: Nsight Systems and Nsight Compute ship with the CUDA
# Toolkit installer (or are installable separately from NVIDIA's site).
# Confirm they're on PATH:
#   nsys --version
#   ncu --version
#
# Usage (from project root, after building -- see build_windows.ps1):
#   .\scripts\run_nsight_profile.ps1 -Mode timeline
#   .\scripts\run_nsight_profile.ps1 -Mode kernel
#
# -Mode timeline : runs Nsight Systems, produces a .nsys-rep timeline you
#                  can open in the Nsight Systems GUI. Good for seeing the
#                  overall iteration loop structure, host<->device copies,
#                  and whether kernels are back-to-back (no gaps) as
#                  intended by the no-per-iteration-sync design (see
#                  devdocs/DEV_LOG.md section 7).
# -Mode kernel   : runs Nsight Compute on the two CP kernels specifically,
#                  producing detailed occupancy / memory-throughput /
#                  warp-divergence metrics for the exact kernels named in
#                  spec section 9 (kernel_dual_ascent_project_tiled,
#                  kernel_primal_update_tiled). This is what tells you
#                  whether the spec's >=75% global memory efficiency target
#                  is actually being hit.

param(
    [ValidateSet("timeline", "kernel")]
    [string]$Mode = "timeline",
    [string]$Config = "Release",
    [string]$Exe = "test_denoise.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ExePath = Join-Path $ProjectRoot "build\$Config\$Exe"

if (-not (Test-Path $ExePath)) {
    Write-Error "Executable not found at $ExePath -- build the project first (see build_windows.ps1)."
    exit 1
}

if ($Mode -eq "timeline") {
    Write-Host "Running Nsight Systems timeline profile on $Exe..."
    $outFile = "nsight_timeline_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    nsys profile --output $outFile --force-overwrite=true $ExePath
    Write-Host ""
    Write-Host "Done. Open $outFile.nsys-rep in the Nsight Systems GUI to inspect:"
    Write-Host "  - whether kernel_dual_ascent_project_* and kernel_primal_update_*"
    Write-Host "    launches are back-to-back with no host sync gaps between"
    Write-Host "    iterations (verifies the no-per-iteration-sync design, see"
    Write-Host "    devdocs/DEV_LOG.md section 7)"
    Write-Host "  - total host<->device transfer time vs total kernel time"
}
elseif ($Mode -eq "kernel") {
    Write-Host "Running Nsight Compute kernel-level profile on $Exe..."
    $outFile = "ncu_kernels_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    # --set full gives the comprehensive metric set (occupancy, memory
    # throughput, warp stall reasons, etc); slower to collect than a
    # narrower --set but gives everything spec section 9 asks about in one
    # pass. Limit to a handful of launches (-c 20) since steady-state
    # behavior is what matters, not every one of e.g. 300 iterations.
    ncu --set full -c 20 --target-processes all -o $outFile -f $ExePath
    Write-Host ""
    Write-Host "Done. Open $outFile.ncu-rep in the Nsight Compute GUI, or run:"
    Write-Host "  ncu --import $outFile.ncu-rep --print-summary per-kernel"
    Write-Host "Look at:"
    Write-Host "  - 'Memory Throughput' / 'L2 Cache Throughput' sections for the"
    Write-Host "    >=75% global memory efficiency target (spec NF2)"
    Write-Host "  - 'Achieved Occupancy' vs 'Theoretical Occupancy'"
    Write-Host "  - 'Warp State Statistics' for divergence/stall reasons"
}
