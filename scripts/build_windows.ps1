# build_windows.ps1
#
# Convenience build script for Windows 11, Visual Studio 2022, CUDA 12.4.
# Run from the project root in PowerShell:
#
#   .\scripts\build_windows.ps1
#
# Optional parameters:
#   -Config Release|Debug   (default: Release)
#   -FastMath               enable HCTV_FAST_MATH (see CMakeLists.txt comment)
#   -UseOpenCV              build against OpenCV instead of stb_image
#   -Clean                  wipe the build/ directory first
#
# Prerequisites:
#   - Visual Studio 2022 with the "Desktop development with C++" workload
#   - CUDA Toolkit 12.4 (matches: nvcc release 12.4, V12.4.99,
#     build cuda_12.4.r12.4/compiler.33961263_0)
#   - CMake >= 3.18 (3.28+ recommended; ships with recent VS2022 installs
#     under "C++ CMake tools for Windows", or install separately)
#   - third_party/stb_image.h and third_party/stb_image_write.h present
#     (see third_party/README.md -- NOT vendored in this repo, must be
#     downloaded once)

param(
    [string]$Config = "Release",
    [switch]$FastMath,
    [switch]$UseOpenCV,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Clean -and (Test-Path "build")) {
    Write-Host "Removing existing build/ directory..."
    Remove-Item -Recurse -Force "build"
}

if (-not (Test-Path "third_party/stb_image.h") -and -not $UseOpenCV) {
    Write-Warning "third_party/stb_image.h not found. See third_party/README.md for download instructions."
    Write-Warning "Continuing anyway -- the build will fail at the ImageIO.h #include step if you don't fix this first."
}

New-Item -ItemType Directory -Force -Path "build" | Out-Null
Set-Location "build"

$cmakeArgs = @(
    "..",
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_CUDA_ARCHITECTURES=89"
)

if ($FastMath) {
    $cmakeArgs += "-DHCTV_FAST_MATH=ON"
}
if ($UseOpenCV) {
    $cmakeArgs += "-DHCTV_USE_OPENCV=ON"
}

Write-Host "Configuring with: cmake $($cmakeArgs -join ' ')"
cmake @cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configure step failed. See output above."
    exit 1
}

Write-Host "Building ($Config configuration)..."
cmake --build . --config $Config -- /m

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed. See output above."
    exit 1
}

Write-Host ""
Write-Host "Build succeeded. Executables are in build\$Config\:"
Write-Host "  HilbertCUDA-TV.exe   -- main CLI tool"
Write-Host "  test_adjoint.exe     -- GPU adjoint correctness test (run this first!)"
Write-Host "  test_denoise.exe     -- end-to-end PSNR/SSIM acceptance test"
Write-Host ""
Write-Host "Suggested first run:"
Write-Host "  .\$Config\test_adjoint.exe"
Write-Host "  .\$Config\test_denoise.exe"
Write-Host "  .\$Config\HilbertCUDA-TV.exe --demo --width 512 --height 512"
