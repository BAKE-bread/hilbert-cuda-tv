// RangeCheck.h
//
// Shared input-data sanity check, used by every mode (gray/color/volume)
// right after loading data, before doing anything else with it.
//
// Loading raw medical-imaging data (e.g. CT Hounsfield-range values in
// [0,1999] or [-1000,500]) without normalizing it first silently produced
// nonsensical results (PSNR around -50 dB) because the rest of the
// pipeline assumes [0,1]-normalized input. This check catches that BEFORE
// the solver runs, rather than letting the user discover it only via a
// bizarre metric after the fact.
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <limits>
#include <string>

namespace hctv {

enum class RangeStatus { Ok, WarnUnnormalized, Error };

struct RangeCheckResult {
    RangeStatus status = RangeStatus::Ok;
    double min_val = 0.0;
    double max_val = 0.0;
    std::string message;
};

// tol: how much slack to allow around [0,1] before warning (default 0.05,
// i.e. values in [-0.05, 1.05] are still considered "normalized" -- this
// avoids spurious warnings from minor float roundoff at the boundary,
// while still catching genuinely unnormalized data by a wide margin).
inline RangeCheckResult check_value_range(const std::vector<float>& data, double tol = 0.05) {
    RangeCheckResult result;
    if (data.empty()) {
        result.status = RangeStatus::Error;
        result.message = "input data is empty";
        return result;
    }

    double vmin = std::numeric_limits<double>::infinity();
    double vmax = -std::numeric_limits<double>::infinity();
    bool has_nan_or_inf = false;

    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) {
            has_nan_or_inf = true;
            break;
        }
        double dv = static_cast<double>(v);
        vmin = std::min(vmin, dv);
        vmax = std::max(vmax, dv);
    }

    if (has_nan_or_inf) {
        result.status = RangeStatus::Error;
        result.message = "input data contains NaN or Inf values";
        return result;
    }

    result.min_val = vmin;
    result.max_val = vmax;

    if (vmax - vmin < 1e-9) {
        result.status = RangeStatus::Error;
        result.message = "input data is (near-)constant (min == max == " +
                          std::to_string(vmin) + "); nothing to denoise";
        return result;
    }

    if (vmin < -tol || vmax > 1.0 + tol) {
        result.status = RangeStatus::WarnUnnormalized;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                       "detected value range [%.6f, %.6f] is outside [0,1] -- "
                       "this data does not look normalized.",
                       vmin, vmax);
        result.message = buf;
        return result;
    }

    result.status = RangeStatus::Ok;
    return result;
}

// Rescales data in-place to [0,1] via min-max normalization using the
// already-detected min_val/max_val (avoids a second full pass over the
// data). Throws if max_val <= min_val (caller should have already
// checked via check_value_range before calling this).
inline void normalize_to_unit_range(std::vector<float>& data, double min_val, double max_val) {
    if (max_val <= min_val) {
        throw std::runtime_error("normalize_to_unit_range: max_val <= min_val, cannot normalize");
    }
    double scale = 1.0 / (max_val - min_val);
    for (auto& v : data) {
        v = static_cast<float>((static_cast<double>(v) - min_val) * scale);
    }
}

// Convenience wrapper combining the check and (optional) auto-normalize,
// with consistent logging, for use at the top of each mode's data-loading
// path. Returns true if the data is now safe to proceed with (either it
// was already fine, or it was successfully auto-normalized); false if it
// should be treated as a hard error by the caller.
inline bool validate_and_maybe_normalize(std::vector<float>& data, const std::string& what,
                                          bool allow_auto_normalize = true) {
    RangeCheckResult r = check_value_range(data);
    switch (r.status) {
        case RangeStatus::Ok:
            return true;
        case RangeStatus::Error:
            fprintf(stderr, "Error: %s -- %s\n", what.c_str(), r.message.c_str());
            return false;
        case RangeStatus::WarnUnnormalized:
            if (allow_auto_normalize) {
                printf("WARNING: %s %s\n", what.c_str(), r.message.c_str());
                printf("  Auto-normalizing to [0,1] using observed min/max "
                       "(pass --no-auto-normalize to disable this and proceed "
                       "as-is; see README.md for why this matters).\n");
                normalize_to_unit_range(data, r.min_val, r.max_val);
                return true;
            } else {
                printf("WARNING: %s %s\n", what.c_str(), r.message.c_str());
                printf("  Proceeding WITHOUT normalizing (--no-auto-normalize was given). "
                       "Lambda will be auto-estimated relative to this data's actual scale, "
                       "but results may still be harder to interpret/compare across datasets "
                       "than if you normalize consistently yourself -- see README.md.\n");
                return true;
            }
    }
    return false; // unreachable, silences -Wreturn-type on some compilers
}

} // namespace hctv
