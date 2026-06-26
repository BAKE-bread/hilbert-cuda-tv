#!/usr/bin/env python3
"""
verify_rangecheck_math.py -- algebraic verification of RangeCheck.h's
normalize_to_unit_range():
  scale = 1/(max-min); v_new = (v_old - min) * scale
This must satisfy: f(min) = 0, f(max) = 1, and be exactly affine. Also
verifies the detection tolerance (tol=0.05) is self-consistent: a value
exactly at the boundary of [-tol, 1+tol] should read Ok, and
infinitesimally outside should read WarnUnnormalized (no gap or overlap
in the logic). Self-contained --re-derives the formulas independently 
rather than importing include/utils/RangeCheck.h (which is C++, not Python), 
so this is a from-scratch algebraic check, not a transcription test.

Run from anywhere: python3 devdocs/verification/verify_rangecheck_math.py
"""
import sys

import numpy as np


def normalize(data, vmin, vmax):
    scale = 1.0 / (vmax - vmin)
    return (data - vmin) * scale


def status(vmin, vmax, tol=0.05):
    if vmax - vmin < 1e-9:
        return "Error(constant)"
    if vmin < -tol or vmax > 1.0 + tol:
        return "WarnUnnormalized"
    return "Ok"


def main():
    ok = True

    # Property 1: endpoints map exactly to 0 and 1
    for vmin, vmax in [(-1000, 1999), (0, 255), (-1, 1), (0.1, 0.9)]:
        out_min = normalize(np.array([vmin]), vmin, vmax)[0]
        out_max = normalize(np.array([vmax]), vmin, vmax)[0]
        if abs(out_min - 0.0) >= 1e-12 or abs(out_max - 1.0) >= 1e-12:
            print(f"FAIL: [{vmin},{vmax}] -> min={out_min}, max={out_max} (expected 0, 1)")
            ok = False
    if ok:
        print("Property 1 PASS: min->0, max->1 exactly, across multiple ranges")

    # Property 2: affine ratio preservation -- normalize is an affine
    # bijection [vmin,vmax] -> [0,1], so it must preserve ratios of
    # differences (the defining property of affine maps, distinct from
    # linear maps since there's a translation involved).
    vmin, vmax = -100.0, 400.0
    x1, x2, x3 = 50.0, 150.0, 350.0
    y1, y2, y3 = normalize(np.array([x1, x2, x3]), vmin, vmax)
    ratio_input = (x2 - x1) / (x3 - x1)
    ratio_output = (y2 - y1) / (y3 - y1)
    if abs(ratio_input - ratio_output) >= 1e-12:
        print(f"FAIL: affine ratio not preserved ({ratio_input} != {ratio_output})")
        ok = False
    else:
        print(f"Property 2 PASS: affine ratio preserved ({ratio_input:.6f} == {ratio_output:.6f})")

    # Property 3: tolerance boundary self-consistency in check_value_range's
    # logic: status flips to WarnUnnormalized iff vmin < -tol OR vmax > 1+tol.
    print()
    print("Boundary behavior (tol=0.05):")
    test_cases = [
        (-0.05, 1.0, "exactly at lower boundary", "Ok"),
        (-0.0500001, 1.0, "epsilon past lower boundary", "WarnUnnormalized"),
        (0.0, 1.05, "exactly at upper boundary", "Ok"),
        (0.0, 1.0500001, "epsilon past upper boundary", "WarnUnnormalized"),
    ]
    for vmin, vmax, desc, expected in test_cases:
        s = status(vmin, vmax)
        match = "OK" if s == expected else "MISMATCH"
        if s != expected:
            ok = False
        print(f"  [{vmin:.7f}, {vmax:.7f}] ({desc}): {s} (expected {expected}) [{match}]")

    print()
    if ok:
        print("Property 3 PASS: boundary is closed (inclusive) on the Ok side, exactly")
        print("matching the C++ source's strict < / > comparisons (vmin < -tol, vmax > 1+tol).")
        print()
        print("ALL PROPERTIES PASS")
    else:
        print("SOME PROPERTIES FAILED -- see above")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
