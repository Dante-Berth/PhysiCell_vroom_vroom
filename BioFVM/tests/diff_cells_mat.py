#!/usr/bin/env python3
"""Numerically diff the CELL state .mat files of two PhysiCell output directories.

Companion to diff_microenvironment_mat.py, for verifying mechanics changes: an
optimized mechanics path must leave cell positions/velocities/state bit-identical
(within tolerance) to the baseline. Compares files named like
    output########_cells.mat   (also initial_cells.mat / final_cells.mat)
matching by path relative to each root, and reports the max abs/rel difference
across all matching frames. Exits non-zero if it exceeds --tol so it can gate a
`make verify-mech` check.

PhysiCell writes cells.mat as one matrix: rows = per-cell data fields
(ID, position x/y/z, ... ), columns = cells, ordered by cell ID. We compare the
full matrix. If the two runs produced a different cell COUNT (columns) or field
count (rows), that is itself a divergence and reported as a shape mismatch.

Usage:
    diff_cells_mat.py <dir_a> <dir_b> [--tol 1e-9]
"""
import sys
import glob
import os
import argparse
import numpy as np
from scipy.io import loadmat


def load_cells(path):
    """Return the dense cell data matrix from a PhysiCell *_cells.mat file."""
    m = loadmat(path)
    arrays = [v for k, v in m.items() if not k.startswith("__")
              and isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 0]
    if not arrays:
        raise ValueError(f"no data matrix in {path}")
    # the cell dump is the largest array
    return max(arrays, key=lambda a: a.size).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    pat = "**/*_cells.mat"
    files_a = sorted(glob.glob(os.path.join(args.dir_a, pat), recursive=True))
    files_a += sorted(glob.glob(os.path.join(args.dir_a, "*_cells.mat")))
    files_a = sorted(set(files_a))
    if not files_a:
        print(f"FAIL: no *_cells.mat files under {args.dir_a}")
        return 2

    max_abs = 0.0
    max_rel = 0.0
    n_compared = 0
    worst = None
    for fa in files_a:
        rel = os.path.relpath(fa, args.dir_a)
        fb = os.path.join(args.dir_b, rel)
        if not os.path.exists(fb):
            print(f"FAIL: {rel} missing in {args.dir_b}")
            return 2
        a, b = load_cells(fa), load_cells(fb)
        if a.shape != b.shape:
            print(f"FAIL: shape mismatch {rel}: {a.shape} vs {b.shape} "
                  f"(cell count or field count diverged)")
            return 2
        d = np.abs(a - b)
        r = d / (np.abs(b) + 1e-300)
        fa_abs = float(d.max())
        if fa_abs > max_abs:
            worst = rel
        max_abs = max(max_abs, fa_abs)
        max_rel = max(max_rel, float(r.max()))
        n_compared += 1

    print(f"frames compared    : {n_compared}")
    print(f"max abs difference : {max_abs:.3e}" + (f"  (worst: {worst})" if worst else ""))
    print(f"max rel difference : {max_rel:.3e}")
    if max_abs > args.tol:
        print(f"RESULT: FAIL (exceeds tol {args.tol:.1e})")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
