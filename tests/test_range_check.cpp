// test_range_check.cpp
//
// Standalone unit test for include/utils/RangeCheck.h. Previously this
// header was only manually verified via a throwaway /tmp test during
// development -- this gives it a real, repeatable, assertion-based test 
// committed to the repo.
//
// RangeCheck.h is pure host C++ with no CUDA dependency at all, so this
// test (like devdocs/cpu_reference/cpu_reference.cpp) builds and runs with
// a plain C++17 compiler -- no nvcc/GPU needed:
//
//   g++ -O2 -std=c++17 -I include tests/test_range_check.cpp -o test_range_check
//   ./test_range_check
//
// Exit code 0 means all assertions passed; any failure prints which case
// failed and returns non-zero.

#include "utils/RangeCheck.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <limits>

using hctv::RangeStatus;
using hctv::check_value_range;
using hctv::normalize_to_unit_range;
using hctv::validate_and_maybe_normalize;

static int g_failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
            g_failures++; \
        } \
    } while (0)

static void test_already_normalized_is_ok() {
    std::vector<float> data = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    auto r = check_value_range(data);
    CHECK(r.status == RangeStatus::Ok, "in-[0,1] data should be Ok");
}

static void test_slightly_outside_tolerance_is_still_ok() {
    // Default tol=0.05 -> [-0.05, 1.05] should still read as Ok (avoids
    // spurious warnings from float roundoff right at the boundary).
    std::vector<float> data = {-0.03f, 0.5f, 1.03f};
    auto r = check_value_range(data);
    CHECK(r.status == RangeStatus::Ok, "data within tol of [0,1] should be Ok");
}

static void test_clearly_unnormalized_warns() {
    // e.g. CT Hounsfield-range-like values, the exact bug class this
    // header exists to catch.
    std::vector<float> data = {-1000.0f, -500.0f, 0.0f, 500.0f, 1999.0f};
    auto r = check_value_range(data);
    CHECK(r.status == RangeStatus::WarnUnnormalized, "CT-like range should warn, not silently pass");
    CHECK(!r.message.empty(), "warning should include a message");
}

static void test_nan_is_error() {
    std::vector<float> data = {0.1f, std::numeric_limits<float>::quiet_NaN(), 0.5f};
    auto r = check_value_range(data);
    CHECK(r.status == RangeStatus::Error, "NaN input should be Error");
}

static void test_inf_is_error() {
    std::vector<float> data = {0.1f, std::numeric_limits<float>::infinity(), 0.5f};
    auto r = check_value_range(data);
    CHECK(r.status == RangeStatus::Error, "Inf input should be Error");
}

static void test_constant_data_is_error() {
    std::vector<float> data = {0.5f, 0.5f, 0.5f, 0.5f};
    auto r = check_value_range(data);
    CHECK(r.status == RangeStatus::Error, "constant (min==max) data should be Error -- nothing to denoise");
}

static void test_empty_data_is_error() {
    std::vector<float> data;
    auto r = check_value_range(data);
    CHECK(r.status == RangeStatus::Error, "empty input should be Error");
}

static void test_normalize_to_unit_range_basic() {
    std::vector<float> data = {-1000.0f, 0.0f, 1999.0f};
    normalize_to_unit_range(data, -1000.0, 1999.0);
    CHECK(std::fabs(data[0] - 0.0f) < 1e-5f, "normalize: min should map to 0");
    CHECK(std::fabs(data[2] - 1.0f) < 1e-5f, "normalize: max should map to 1");
    float expected_mid = 1000.0f / 2999.0f;
    CHECK(std::fabs(data[1] - expected_mid) < 1e-5f, "normalize: midpoint should map proportionally");
}

static void test_normalize_throws_on_degenerate_range() {
    std::vector<float> data = {0.5f, 0.5f};
    bool threw = false;
    try {
        normalize_to_unit_range(data, 0.5, 0.5);  // max_val <= min_val
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "normalize_to_unit_range should throw when max_val <= min_val");
}

static void test_validate_and_maybe_normalize_ok_passthrough() {
    std::vector<float> data = {0.0f, 0.5f, 1.0f};
    std::vector<float> original = data;
    bool ok = validate_and_maybe_normalize(data, "test data", /*allow_auto_normalize=*/true);
    CHECK(ok, "already-normalized data should pass");
    CHECK(data == original, "already-normalized data should be untouched");
}

static void test_validate_and_maybe_normalize_auto_normalizes() {
    std::vector<float> data = {-1000.0f, -500.0f, 0.0f, 500.0f, 1999.0f};
    bool ok = validate_and_maybe_normalize(data, "CT-like data", /*allow_auto_normalize=*/true);
    CHECK(ok, "unnormalized data with auto-normalize=true should succeed");
    for (float v : data) {
        CHECK(v >= -1e-5f && v <= 1.0f + 1e-5f, "after auto-normalize, every value should be in [0,1]");
    }
}

static void test_validate_and_maybe_normalize_respects_no_auto_normalize() {
    std::vector<float> data = {-1000.0f, -500.0f, 0.0f, 500.0f, 1999.0f};
    std::vector<float> original = data;
    bool ok = validate_and_maybe_normalize(data, "CT-like data", /*allow_auto_normalize=*/false);
    CHECK(ok, "--no-auto-normalize should still return true (warn, don't block)");
    CHECK(data == original, "--no-auto-normalize should leave data untouched");
}

static void test_validate_and_maybe_normalize_rejects_error_cases() {
    std::vector<float> data = {0.5f, 0.5f, 0.5f};  // constant -> Error
    bool ok = validate_and_maybe_normalize(data, "constant data", /*allow_auto_normalize=*/true);
    CHECK(!ok, "Error-status data should return false regardless of auto_normalize");
}

int main() {
    test_already_normalized_is_ok();
    test_slightly_outside_tolerance_is_still_ok();
    test_clearly_unnormalized_warns();
    test_nan_is_error();
    test_inf_is_error();
    test_constant_data_is_error();
    test_empty_data_is_error();
    test_normalize_to_unit_range_basic();
    test_normalize_throws_on_degenerate_range();
    test_validate_and_maybe_normalize_ok_passthrough();
    test_validate_and_maybe_normalize_auto_normalizes();
    test_validate_and_maybe_normalize_respects_no_auto_normalize();
    test_validate_and_maybe_normalize_rejects_error_cases();

    if (g_failures == 0) {
        std::printf("All RangeCheck.h tests PASSED.\n");
        return 0;
    } else {
        std::printf("%d test(s) FAILED.\n", g_failures);
        return 1;
    }
}
