// main.cpp
//
// Command-line entry point for all three denoising modes: scalar
// grayscale (2D), color (2D, coupled vectorial TV), and volumetric (3D).
//
// Usage:
//   HilbertCUDA-TV --input noisy.png --output denoised.png [options]
//   HilbertCUDA-TV --demo [options]
//   HilbertCUDA-TV --reference clean.png --output denoised.png [options]
//
// Mode selection:
//   --mode gray    (default) scalar grayscale 2D TV denoising
//   --mode color   coupled vectorial color 2D TV denoising (--input/
//                  --output/--reference expect RGB images; --demo
//                  generates a synthetic color test image)
//   --mode volume  3D volumetric TV denoising (--input/--output/
//                  --reference expect .rawvol files; --demo generates a
//                  synthetic sphere-in-background test volume)
//
// Options:
//   --lambda <float>       TV regularization weight. If NOT specified, it
//                          is estimated automatically (see "Automatic
//                          lambda" below). Specify this explicitly to override.
//   --iterations <int>     CP iteration count (default 300)
//   --naive                use naive (non-shared-memory) kernels [gray mode only]
//   --no-auto-normalize    skip auto-normalizing out-of-[0,1]-range input
//                          (still warns; see "Value range checking" below)
//   --noise-sigma <float>  if --demo or --reference is given, add Gaussian
//                          noise with this std-dev (0-255 scale) before
//                          solving (default 25.0)
//   --reference <path>     load a reference file, add synthetic noise,
//                          denoise, report PSNR/SSIM against the original.
//                          NOTE: this is a controlled self-test (inject
//                          KNOWN noise, measure how well it's removed),
//                          not a general "compare two files" tool -- see
//                          tools/compare_volumes.py for that instead.
//   --width/--height <int> size for --demo in gray/color mode (default 512x512)
//   --depth <int>          additional size dimension for --demo in volume mode (default 64)
//
// Automatic lambda (see devdocs/DEV_LOG.md sections 15 and 23):
//   lambda is ALWAYS estimated directly from the array that will actually
//   be fed to the solver (a robust MAD-based noise estimator), in EVERY
//   mode -- --input, --demo, and --reference alike. This deliberately does
//   NOT special-case --demo/--reference to "just trust --noise-sigma",
//   because that assumes the loaded reference file is already noise-free
//   AND already normalized to [0,1] -- neither of which is guaranteed for
//   a user-supplied --reference file (e.g. real medical imaging data in
//   its native intensity range, or a real photo that already has its own
//   noise). Estimating directly from the data unifies the measurement
//   standard between --input and --reference and was found, on real CT
//   data, to be the fix for a case where --reference produced a PSNR of
//   about -50 dB by silently assuming a [0,1]-normalized clean baseline
//   that wasn't actually true.
//
// Value range checking (see devdocs/DEV_LOG.md section 24):
//   Every loaded file's value range is checked immediately after loading.
//   Data far outside [0,1] (e.g. raw CT Hounsfield values) is reported
//   with a clear warning and auto-normalized via min-max rescaling unless
//   --no-auto-normalize is given. This is what prevents the silent-garbage
//   failure mode above from recurring with different data.

#include "solvers/ROFSolver.cuh"
#include "solvers/ColorROFSolver.cuh"
#include "solvers/VolumeROFSolver.cuh"
#include "utils/ImageIO.h"
#include "utils/VolumeIO.h"
#include "utils/Metrics.h"
#include "utils/RangeCheck.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>
#include <exception>

using namespace hctv;

enum class Mode { Gray, Color, Volume };

struct CliOptions {
    std::string input_path;
    std::string output_path;
    std::string reference_path;
    bool demo = false;
    Mode mode = Mode::Gray;

    bool lambda_set = false;
    float lambda = 0.15f;
    int iterations = 300;
    bool use_shared = true;
    bool auto_normalize = true;
    double noise_sigma = 25.0;
    int demo_width = 512;
    int demo_height = 512;
    int demo_depth = 64;
};

static void print_usage() {
    printf("HilbertCUDA-TV: GPU-accelerated Total Variation denoiser\n");
    printf("(scalar grayscale, color, and 3D volumetric modes)\n\n");
    printf("Usage:\n");
    printf("  HilbertCUDA-TV --input <file> --output <file> [options]\n");
    printf("  HilbertCUDA-TV --demo [options]\n");
    printf("  HilbertCUDA-TV --reference <file> --output <file> [options]\n\n");
    printf("Mode:\n");
    printf("  --mode gray|color|volume   (default: gray)\n\n");
    printf("Options:\n");
    printf("  --lambda <float>        TV weight (default: auto-estimated, see README)\n");
    printf("  --iterations <int>      CP iterations (default 300)\n");
    printf("  --naive                 use non-shared-memory kernels (gray mode only)\n");
    printf("  --no-auto-normalize     skip auto-normalizing out-of-[0,1]-range input\n");
    printf("                          (a warning is still printed; see README.md)\n");
    printf("  --noise-sigma <float>   noise std-dev, 0-255 scale (default 25.0)\n");
    printf("  --width <int>           demo width, gray/color mode (default 512)\n");
    printf("  --height <int>          demo height, gray/color mode (default 512)\n");
    printf("  --depth <int>           demo depth, volume mode (default 64)\n");
    printf("  --help                  show this message\n\n");
    printf("File formats: gray/color modes use PNG/JPG/BMP (via stb_image);\n");
    printf("volume mode uses the minimal .rawvol format (see README.md).\n");
}

static bool parse_args(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); exit(1); }
            return std::string(argv[++i]);
        };
        if (arg == "--input") opts.input_path = next();
        else if (arg == "--output") opts.output_path = next();
        else if (arg == "--reference") opts.reference_path = next();
        else if (arg == "--demo") opts.demo = true;
        else if (arg == "--mode") {
            std::string m = next();
            if (m == "gray") opts.mode = Mode::Gray;
            else if (m == "color") opts.mode = Mode::Color;
            else if (m == "volume") opts.mode = Mode::Volume;
            else { fprintf(stderr, "Unknown --mode: %s (expected gray|color|volume)\n", m.c_str()); return false; }
        }
        else if (arg == "--lambda") { opts.lambda = std::stof(next()); opts.lambda_set = true; }
        else if (arg == "--iterations") opts.iterations = std::stoi(next());
        else if (arg == "--naive") opts.use_shared = false;
        else if (arg == "--no-auto-normalize") opts.auto_normalize = false;
        else if (arg == "--noise-sigma") opts.noise_sigma = std::stod(next());
        else if (arg == "--width") opts.demo_width = std::stoi(next());
        else if (arg == "--height") opts.demo_height = std::stoi(next());
        else if (arg == "--depth") opts.demo_depth = std::stoi(next());
        else if (arg == "--help" || arg == "-h") { print_usage(); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", arg.c_str()); print_usage(); return false; }
    }
    if (opts.input_path.empty() && !opts.demo && opts.reference_path.empty()) {
        fprintf(stderr, "Must specify one of --input, --demo, or --reference.\n\n");
        print_usage();
        return false;
    }
    if (opts.output_path.empty()) {
        opts.output_path = (opts.mode == Mode::Volume) ? "denoised.rawvol" : "denoised.png";
    }
    return true;
}

// ---------------------------------------------------------------------------
// Gray mode
// ---------------------------------------------------------------------------
static int run_gray_mode(const CliOptions& opts) {
    GrayImage clean, solve_input;
    bool have_reference = false;

    try {
        if (opts.demo) {
            printf("Generating %dx%d synthetic demo image...\n", opts.demo_width, opts.demo_height);
            clean = make_synthetic_test_image(opts.demo_width, opts.demo_height);
            solve_input = add_gaussian_noise(clean, opts.noise_sigma);
            have_reference = true;
        } else if (!opts.reference_path.empty()) {
            printf("Loading reference image: %s\n", opts.reference_path.c_str());
            clean = load_grayscale(opts.reference_path);
            if (!validate_and_maybe_normalize(clean.data, "reference image", opts.auto_normalize)) return 1;
            solve_input = add_gaussian_noise(clean, opts.noise_sigma);
            have_reference = true;
        } else {
            printf("Loading input image: %s\n", opts.input_path.c_str());
            solve_input = load_grayscale(opts.input_path);
            if (!validate_and_maybe_normalize(solve_input.data, "input image", opts.auto_normalize)) return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error loading image: %s\n", e.what());
        return 1;
    }

    int W = solve_input.width, H = solve_input.height;

    // IMPORTANT: lambda is ALWAYS estimated from solve_input -- the actual
    // array the solver will see -- in EVERY mode, never assumed from
    // opts.noise_sigma directly. This unifies --input and --reference's
    // measurement standard (see devdocs/DEV_LOG.md section 23): previously
    // --reference/--demo blindly trusted opts.noise_sigma/255, which silently
    // assumed the loaded "reference" was already clean [0,1] data -- true for
    // --demo's own synthetic image, but NOT guaranteed for a user-supplied
    // --reference file (e.g. unnormalized medical data, or a real photo that
    // already has its own noise). Estimating from solve_input directly is
    // correct in all cases.
    //
    // Numerically re-verified (Python replica of this exact estimator and
    // the synthetic test image, see devdocs/DEV_LOG.md section 30): at the
    // historically-validated default noise-sigma=25/255, the resulting
    // lambda differs from the old (pre-fix) trust-noise-sigma lambda by
    // under ~1% on average across seeds on the built-in synthetic test
    // image, so old --demo PSNR/SSIM numbers at that setting remain a
    // reasonable sanity check. This is NOT a tight bound at other settings,
    // though: the gap grows with noise-sigma (up to ~5% by sigma=60/255 in
    // the same test) because heavier injected noise clips more pixels at
    // the [0,1] boundary, a real and expected nonlinearity (clipping
    // suppresses the *effective* noise variance the estimator measures) --
    // not an estimator bug. An earlier version of this comment claimed
    // "~0.1%" as a general bound; that number was specific to one
    // seed/sigma combination and did not hold up under broader
    // re-verification, so it has been corrected here.
    float lambda = opts.lambda;
    if (!opts.lambda_set) {
        double sigma_to_use = estimate_noise_sigma(solve_input.data, W, H);
        printf("No --lambda given; estimated noise sigma=%.4f (255-scale %.1f) from the %s.\n",
               sigma_to_use, sigma_to_use * 255.0, have_reference ? "noisy (post-injection) image" : "input image");
        lambda = lambda_from_sigma(sigma_to_use);
        printf("Using auto lambda=%.4f (= 1.5 * sigma). Override with --lambda if this over/under-smooths.\n", lambda);
    }

    printf("Image size: %dx%d  lambda=%.4f  iterations=%d  kernels=%s\n",
           W, H, lambda, opts.iterations, opts.use_shared ? "tiled" : "naive");

    ROFParams params;
    params.lambda = lambda;
    params.max_iterations = opts.iterations;
    params.use_shared_memory = opts.use_shared;

    ROFSolver solver(W, H, params);
    ROFResult result = solver.solve(solve_input.data);

    printf("\nSolve complete: %d iterations, %.3f ms total, %.4f ms/iteration\n",
           result.iterations_run, result.total_kernel_time_ms, result.avg_iter_time_ms);

    if (have_reference) {
        double psnr_before = psnr(clean.data, solve_input.data);
        double psnr_after = psnr(clean.data, result.denoised);
        double ssim_after = ssim_windowed(clean.data, result.denoised, W, H);
        printf("\nPSNR (noisy vs clean):     %.2f dB\n", psnr_before);
        printf("PSNR (denoised vs clean): %.2f dB  (improvement: %+.2f dB)\n",
               psnr_after, psnr_after - psnr_before);
        printf("SSIM (denoised vs clean): %.4f\n", ssim_after);
    }

    GrayImage out_img;
    out_img.width = W;
    out_img.height = H;
    out_img.data = result.denoised;

    try {
        save_grayscale_png(opts.output_path, out_img);
        printf("\nSaved denoised image to: %s\n", opts.output_path.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error saving image: %s\n", e.what());
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Color mode
// ---------------------------------------------------------------------------
static int run_color_mode(const CliOptions& opts) {
    ColorImage clean, solve_input;
    bool have_reference = false;

    try {
        if (opts.demo) {
            printf("Generating %dx%d synthetic color demo image...\n", opts.demo_width, opts.demo_height);
            clean = make_synthetic_color_test_image(opts.demo_width, opts.demo_height);
            solve_input = add_gaussian_noise_color(clean, opts.noise_sigma);
            have_reference = true;
        } else if (!opts.reference_path.empty()) {
            printf("Loading reference image: %s\n", opts.reference_path.c_str());
            clean = load_color(opts.reference_path);
            if (!validate_and_maybe_normalize(clean.data, "reference image", opts.auto_normalize)) return 1;
            solve_input = add_gaussian_noise_color(clean, opts.noise_sigma);
            have_reference = true;
        } else {
            printf("Loading input image: %s\n", opts.input_path.c_str());
            solve_input = load_color(opts.input_path);
            if (!validate_and_maybe_normalize(solve_input.data, "input image", opts.auto_normalize)) return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error loading image: %s\n", e.what());
        return 1;
    }

    int W = solve_input.width, H = solve_input.height;

    // See run_gray_mode's comment: lambda is ALWAYS estimated from
    // solve_input directly, in every mode, never assumed from
    // opts.noise_sigma. Unifies --input/--reference measurement standard.
    float lambda = opts.lambda;
    if (!opts.lambda_set) {
        double sigma_to_use = estimate_noise_sigma_color(solve_input);
        printf("No --lambda given; estimated noise sigma=%.4f (255-scale %.1f) from the %s.\n",
               sigma_to_use, sigma_to_use * 255.0, have_reference ? "noisy (post-injection) image" : "input image");
        lambda = lambda_from_sigma(sigma_to_use);
        printf("Using auto lambda=%.4f (= 1.5 * sigma). Override with --lambda if this over/under-smooths.\n", lambda);
    }

    printf("Image size: %dx%d (3 channels)  lambda=%.4f  iterations=%d\n",
           W, H, lambda, opts.iterations);

    ColorROFParams params;
    params.lambda = lambda;
    params.max_iterations = opts.iterations;
    params.channels = 3;

    ColorROFSolver solver(W, H, params);
    ColorROFResult result = solver.solve(solve_input.data);

    printf("\nSolve complete: %d iterations, %.3f ms total, %.4f ms/iteration\n",
           result.iterations_run, result.total_kernel_time_ms, result.avg_iter_time_ms);

    if (have_reference) {
        double psnr_before = psnr(clean.data, solve_input.data);
        double psnr_after = psnr(clean.data, result.denoised);
        printf("\nPSNR (noisy vs clean):     %.2f dB\n", psnr_before);
        printf("PSNR (denoised vs clean): %.2f dB  (improvement: %+.2f dB)\n",
               psnr_after, psnr_after - psnr_before);
    }

    ColorImage out_img;
    out_img.width = W;
    out_img.height = H;
    out_img.data = result.denoised;

    try {
        save_color_png(opts.output_path, out_img);
        printf("\nSaved denoised image to: %s\n", opts.output_path.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error saving image: %s\n", e.what());
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Volume mode
// ---------------------------------------------------------------------------
static int run_volume_mode(const CliOptions& opts) {
    Volume clean, solve_input;
    bool have_reference = false;

    try {
        if (opts.demo) {
            printf("Generating %dx%dx%d synthetic demo volume...\n", opts.demo_width, opts.demo_height, opts.demo_depth);
            clean = make_synthetic_test_volume(opts.demo_width, opts.demo_height, opts.demo_depth);
            solve_input = add_gaussian_noise_volume(clean, opts.noise_sigma);
            have_reference = true;
        } else if (!opts.reference_path.empty()) {
            printf("Loading reference volume: %s\n", opts.reference_path.c_str());
            clean = load_rawvol(opts.reference_path);
            if (!validate_and_maybe_normalize(clean.data, "reference volume", opts.auto_normalize)) return 1;
            solve_input = add_gaussian_noise_volume(clean, opts.noise_sigma);
            have_reference = true;
        } else {
            printf("Loading input volume: %s\n", opts.input_path.c_str());
            solve_input = load_rawvol(opts.input_path);
            if (!validate_and_maybe_normalize(solve_input.data, "input volume", opts.auto_normalize)) return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error loading volume: %s\n", e.what());
        return 1;
    }

    int W = solve_input.width, H = solve_input.height, D = solve_input.depth;

    // See run_gray_mode's comment: lambda is ALWAYS estimated from
    // solve_input directly, in every mode, never assumed from
    // opts.noise_sigma. Unifies --input/--reference measurement standard
    // -- this is what fixes the catastrophic --reference-on-unnormalized-
    // CT-data failure (PSNR ~-50dB) reported on real MSD heart CT data,
    // see devdocs/DEV_LOG.md section 21/23 for the full diagnosis.
    float lambda = opts.lambda;
    if (!opts.lambda_set) {
        double sigma_to_use = estimate_noise_sigma_volume(solve_input.data, W, H, D);
        printf("No --lambda given; estimated noise sigma=%.4f (255-scale %.1f) from the %s.\n",
               sigma_to_use, sigma_to_use * 255.0, have_reference ? "noisy (post-injection) volume" : "input volume");
        lambda = lambda_from_sigma(sigma_to_use);
        printf("Using auto lambda=%.4f (= 1.5 * sigma). Override with --lambda if this over/under-smooths.\n", lambda);
    }

    printf("Volume size: %dx%dx%d  lambda=%.4f  iterations=%d\n", W, H, D, lambda, opts.iterations);

    VolumeROFParams params;
    params.lambda = lambda;
    params.max_iterations = opts.iterations;

    VolumeROFSolver solver(W, H, D, params);
    VolumeROFResult result = solver.solve(solve_input.data);

    printf("\nSolve complete: %d iterations, %.3f ms total, %.4f ms/iteration\n",
           result.iterations_run, result.total_kernel_time_ms, result.avg_iter_time_ms);

    if (have_reference) {
        double psnr_before = psnr(clean.data, solve_input.data);
        double psnr_after = psnr(clean.data, result.denoised);
        printf("\nPSNR (noisy vs clean):     %.2f dB\n", psnr_before);
        printf("PSNR (denoised vs clean): %.2f dB  (improvement: %+.2f dB)\n",
               psnr_after, psnr_after - psnr_before);
    }

    Volume out_vol;
    out_vol.width = W;
    out_vol.height = H;
    out_vol.depth = D;
    out_vol.data = result.denoised;

    try {
        save_rawvol(opts.output_path, out_vol);
        printf("\nSaved denoised volume to: %s\n", opts.output_path.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error saving volume: %s\n", e.what());
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    CliOptions opts;
    if (!parse_args(argc, argv, opts)) return 1;

    switch (opts.mode) {
        case Mode::Gray:   return run_gray_mode(opts);
        case Mode::Color:  return run_color_mode(opts);
        case Mode::Volume: return run_volume_mode(opts);
    }
    return 1; // unreachable
}
