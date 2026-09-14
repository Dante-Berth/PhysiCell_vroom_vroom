"""Blob initial conditions at a MATCHED cell count, whatever the threshold.

The stock path loses cells as the threshold rises, and the loss is large: realised
tumour counts in the vera_blob sweep were 120 / 109 / 97 / 68 at thresholds
0.55 / 0.75 / 0.85 / 0.95 against a requested 128. So "blob" and "smaller tumour"
varied together and the monotone efficacy trend could not be attributed to
compactness alone.

The mechanism, in init_conds.py: weighted_pick samples voxels WITH replacement
(np.random.choice(..., replace=True)), then generate_initial_condition rounds to the
integer grid and drops duplicate (x, y). A high threshold leaves a small mask, so more
of the n draws collide and more are dropped. It is a birthday-problem loss, not a
capacity limit -- at threshold 0.95 the mask still holds far more than 128 voxels.

This module draws voxels WITHOUT replacement instead, so every requested cell gets its
own grid spot and the realised count equals the requested count exactly, as long as the
mask has enough voxels. It keeps the same probability weights, so the spatial
distribution within the mask is unchanged -- only the collisions are removed.

⚠️ If a mask has fewer voxels than cells requested, no amount of resampling helps: the
function raises rather than silently returning a short population, because a silent
shortfall is the exact failure this module exists to remove.
"""
import numpy as np
import pandas as pd


def weighted_pick_nodup(arr, threshold, n, rng):
    """Same weights as init_conds.weighted_pick, but without replacement."""
    mask = arr > threshold
    coords = np.argwhere(mask)
    if len(coords) < n:
        raise ValueError(
            f"mask has {len(coords)} voxels above threshold {threshold} but {n} cells "
            f"were requested; lower the threshold or reduce number_cells")
    probs = arr[mask].astype(float)
    probs /= probs.sum()
    idx = rng.choice(len(coords), size=n, replace=False, p=probs)
    return coords[idx]


def generate_blob_ic(csv_path, correlation_length, threshold, counts, seed,
                     domain=(63, 63), bounds_lo=(0, 0)):
    """One IC with exactly `counts[type]` cells of each type, written to csv_path.

    Cell types are placed one after another and each type's voxels are removed from the
    pool for later types, so no two cells of ANY type share a grid spot -- matching the
    stock path's global drop_duplicates, which is also across types.
    """
    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "custom_modules/physigym/physigym/envs"))
    from init_conds import generate_balanced_fields, set_seed

    # set_seed drives the field generation exactly as the stock path does, so the
    # FIELDS are identical to the unmatched sweep for the same (cl, threshold, seed).
    set_seed(seed)
    params = {k: {"correlation_length": correlation_length, "threshold": threshold,
                  "number_cells": v} for k, v in counts.items()}
    fields = generate_balanced_fields(domain, params)

    rng = np.random.default_rng(seed)
    taken = set()
    xs, ys, types = [], [], []
    for ct, n in counts.items():
        field = fields[ct].copy()
        # Blank out voxels already used so the next type cannot collide with them.
        for (a, b) in taken:
            field[a, b] = -np.inf
        pts = weighted_pick_nodup(field, threshold, n, rng)
        for a, b in pts:
            taken.add((int(a), int(b)))
            xs.append(int(a) + bounds_lo[0])
            ys.append(int(b) + bounds_lo[1])
            types.append(ct)

    df = pd.DataFrame({"x": xs, "y": ys, "z": 0, "type": types})
    assert not df.duplicated(subset=["x", "y"]).any(), "collision survived"
    for ct, n in counts.items():
        got = int((df.type == ct).sum())
        assert got == n, f"{ct}: wanted {n}, got {got}"
    df.to_csv(csv_path, index=False)
    return df
