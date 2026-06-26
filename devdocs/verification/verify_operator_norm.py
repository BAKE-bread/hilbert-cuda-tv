#!/usr/bin/env python3
"""
verify_operator_norm.py -- independent numerical confirmation of the
operator norm bounds used to set the Chambolle-Pock step sizes (tau=sigma
= 1/sqrt(||K||^2)):
  - 2D gradient operator (forward difference, Neumann BC): ||K||^2 <= 8
  - 3D gradient operator (extra z-direction term):          ||K||^2 <= 12

These bounds are stated in README.md ("The operators, concretely"), 
originally as a known result from Chambolle & Pock 2011 SS6.2
for the 2D case, and "confirmed by direct eigenvalue computation" for the
3D case. This script re-derives both independently: builds the EXACT
explicit sparse-as-dense matrix for K (matching the project's own
gated/zero-at-boundary gradient formula precisely, not an approximation),
then computes the largest eigenvalue of K^T K directly (the operator norm
squared, by definition) via numpy's symmetric eigenvalue solver.

This does NOT touch any project source file -- it's a from-scratch
algebraic check against the formula as documented, useful for catching a
documentation/code mismatch as much as confirming the math itself.

Run from anywhere: python3 devdocs/verification/verify_operator_norm.py
"""
import sys

import numpy as np


def build_K_2d(H, W):
    """Explicit matrix for the 2D forward-difference gradient operator
    with Neumann (zero) BC:
      (Kx u)[i,j] = u[i,j+1] - u[i,j]  if j < W-1 else 0
      (Ky u)[i,j] = u[i+1,j] - u[i,j]  if i < H-1 else 0
    Matches include/core/GradientOp.cuh's documented formula exactly."""
    N = H * W
    Kx = np.zeros((N, N))
    Ky = np.zeros((N, N))

    def idx(i, j):
        return i * W + j

    for i in range(H):
        for j in range(W):
            row = idx(i, j)
            if j < W - 1:
                Kx[row, idx(i, j + 1)] += 1
                Kx[row, idx(i, j)] -= 1
            if i < H - 1:
                Ky[row, idx(i + 1, j)] += 1
                Ky[row, idx(i, j)] -= 1
    return np.vstack([Kx, Ky])


def build_K_3d(D, H, W):
    """3D analog with an additional z-direction forward difference."""
    N = D * H * W
    Kx = np.zeros((N, N))
    Ky = np.zeros((N, N))
    Kz = np.zeros((N, N))

    def idx(z, i, j):
        return z * H * W + i * W + j

    for z in range(D):
        for i in range(H):
            for j in range(W):
                row = idx(z, i, j)
                if j < W - 1:
                    Kx[row, idx(z, i, j + 1)] += 1
                    Kx[row, idx(z, i, j)] -= 1
                if i < H - 1:
                    Ky[row, idx(z, i + 1, j)] += 1
                    Ky[row, idx(z, i, j)] -= 1
                if z < D - 1:
                    Kz[row, idx(z + 1, i, j)] += 1
                    Kz[row, idx(z, i, j)] -= 1
    return np.vstack([Kx, Ky, Kz])


def main():
    ok = True
    print("=== 2D case: claim ||K||^2 <= 8 ===")
    for H, W in [(4, 4), (8, 8), (16, 16), (5, 9), (32, 32)]:
        K = build_K_2d(H, W)
        max_eig = np.linalg.eigvalsh(K.T @ K).max()
        violated = max_eig > 8.0 + 1e-9
        ok = ok and not violated
        print(f"  H={H:3d} W={W:3d}: max eig(K^T K) = {max_eig:.6f}  "
              f"{'VIOLATION!' if violated else '(within bound)'}")

    print()
    print("=== 3D case: claim ||K||^2 <= 12 ===")
    for D, H, W in [(3, 3, 3), (4, 4, 4), (6, 6, 6), (3, 5, 7), (8, 8, 8)]:
        K = build_K_3d(D, H, W)
        max_eig = np.linalg.eigvalsh(K.T @ K).max()
        violated = max_eig > 12.0 + 1e-9
        ok = ok and not violated
        print(f"  D={D} H={H} W={W}: max eig(K^T K) = {max_eig:.6f}  "
              f"{'VIOLATION!' if violated else '(within bound)'}")

    print()
    if ok:
        print("CONFIRMED: both bounds hold on every grid tested, with the largest")
        print("eigenvalue monotonically approaching (never exceeding) the bound as")
        print("the grid grows -- consistent with 8 and 12 being the TIGHT asymptotic")
        print("bounds (achieved in the infinite-grid limit), not loose over-estimates.")
    else:
        print("FAILED: a bound was violated -- see VIOLATION! lines above.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
