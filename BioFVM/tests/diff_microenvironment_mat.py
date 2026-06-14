#!/usr/bin/env python3
"""Numerically diff two PhysiCell/BioFVM output directories.

Compares the microenvironment MultiCellDS .mat files written by two runs (e.g.
project_orig vs project_opt) and reports the maximum absolute / relative
difference across all matching output frames. Exits non-zero if the difference
exceeds the tolerance, so it can gate a `make verify` correctness check.

Usage:
    diff_microenvironment_mat.py <dir_a> <dir_b> [--tol 1e-9]

It matches files named like  output########_microenvironment0.mat  in both dirs.
"""
import sys
import glob
import os
import argparse
import numpy as np
from scipy.io import loadmat


def load_field(path):
    """Return the dense data array from a BioFVM microenvironment .mat file.

    BioFVM writes a single matrix; we take the largest 2-D array in the file as
    the concatenated [position; densities] block, which is what we compare.
    """
    m = loadmat(path)
    arrays = [v for k, v in m.items() if not k.startswith("__")
              and isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 0]
    if not arrays:
        raise ValueError(f"no data matrix in {path}")
    # the microenvironment dump is the largest array
    return max(arrays, key=lambda a: a.size).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    # Recurse: match *_microenvironment0.mat files by their path relative to each
    # root, so nested layouts (e.g. episode########/ subfolders) work.
    pat = "**/*microenvironment0.mat"
    files_a = sorted(glob.glob(os.path.join(args.dir_a, pat), recursive=True))
    # also catch flat layout
    files_a += sorted(glob.glob(os.path.join(args.dir_a, "*microenvironment0.mat")))
    files_a = sorted(set(files_a))
    if not files_a:
        print(f"FAIL: no *microenvironment0.mat files under {args.dir_a}")
        return 2

    max_abs = 0.0
    max_rel = 0.0
    n_compared = 0
    for fa in files_a:
        rel = os.path.relpath(fa, args.dir_a)
        fb = os.path.join(args.dir_b, rel)
        if not os.path.exists(fb):
            print(f"FAIL: {rel} missing in {args.dir_b}")
            return 2
        a, b = load_field(fa), load_field(fb)
        if a.shape != b.shape:
            print(f"FAIL: shape mismatch {os.path.basename(fa)}: {a.shape} vs {b.shape}")
            return 2
        d = np.abs(a - b)
        r = d / (np.abs(b) + 1e-300)
        max_abs = max(max_abs, float(d.max()))
        max_rel = max(max_rel, float(r.max()))
        n_compared += 1

    print(f"frames compared    : {n_compared}")
    print(f"max abs difference : {max_abs:.3e}")
    print(f"max rel difference : {max_rel:.3e}")
    if max_abs > args.tol:
        print(f"RESULT: FAIL (exceeds tol {args.tol:.1e})")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
