"""
Smoke-test a set of observation modes through the same single-env factory
that vectorized.py uses. For each mode we build the env, reset, take a few
steps, and confirm the observation matches the declared observation_space.
"""
import ctypes
import ctypes.util
import os

gomp_path = ctypes.util.find_library("gomp")
if gomp_path:
    ctypes.CDLL(gomp_path, mode=ctypes.RTLD_GLOBAL)
os.environ.setdefault("PHYSIGYM_QUIET", "1")

import copy
import numpy as np

import vectorized as V

MODES_TO_TEST = [
    "spatial_scalars_cells_m1m2",
    "spatial_scalars_cells_substrates_m1m2",
    "spatial_scalars_cells_substrates",
    "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2",
]

# Modes already known to work — sanity anchors.
KNOWN_GOOD = [
    "spatial_scalars_cells",
    "spatial_scalars_cells_spatial_no_scalars_substrates",
]


def base_cfg():
    params = {
        "tumor": {"correlation_length": 45, "threshold": 0.55, "number_cells": 64},
        "t_cell": {"correlation_length": 45, "threshold": 0.55, "number_cells": 8},
        "macrophage": {"correlation_length": 45, "threshold": 0.55, "number_cells": 32},
    }
    return {
        "simulation": {"max_time": 1440.0, "seed": 3},
        "vectorization": {"num_envs": 1, "rl_threads": 1},
        "model": {
            "id": "physigym/ModelPhysiCellEnv-v0",
            "settingxml": "config/PhysiCell_settings.xml",
            "settingcells": "config/cells.csv",
            "action_mode": "targeted",
            "output_dir": "./test_obs_modes_output",
            "figsize": (6, 6),
            "observation_mode": "scalars_macrophages",
            "render_mode": None,
            "verbose": False,
            "img_rgb_grid_size_x": 64,
            "img_rgb_grid_size_y": 64,
            "img_mc_grid_size_x": 64,
            "img_mc_grid_size_y": 64,
            "normalization_factor": 512,
        },
        "wrapper": {
            "list_variable_name": [
                "drug_1_dose",
                "drug_1_x",
                "drug_1_y",
                "drug_1_radius",
            ],
            "w_cell": 0.3,
            "w_smooth": 0.02,
            "action_delta_max": [1.0, 0.15, 0.15, 0.05],
        },
        "generation": {
            "x_min": 0,
            "x_max": 64,
            "y_min": 0,
            "y_max": 64,
            "params": params,
            "seed": 3,
            "mode_train": "network_field",
            "mode_test": ["rectangle", "circular", "network_field"],
        },
        "rl": {"total_timesteps": 100},
    }


def test_mode(mode, n_steps=3):
    cfg = copy.deepcopy(base_cfg())
    cfg["model"]["observation_mode"] = mode

    env = V.test_make_physigym_env(cfg)
    obs, info = env.reset()

    space = env.observation_space
    obs = np.asarray(obs)

    if obs.shape != space.shape:
        raise AssertionError(
            f"reset obs shape {obs.shape} != space shape {space.shape}"
        )
    if not space.contains(obs):
        # report where it falls out of bounds
        lo = np.asarray(space.low)
        hi = np.asarray(space.high)
        below = int(np.sum(obs < lo))
        above = int(np.sum(obs > hi))
        raise AssertionError(
            f"reset obs not in space (dtype obs={obs.dtype} space={space.dtype}; "
            f"{below} below low, {above} above high; "
            f"obs range [{obs.min()}, {obs.max()}] vs space [{lo.min()}, {hi.max()}])"
        )

    for i in range(n_steps):
        a = np.random.rand(4).astype(np.float32)
        obs, reward, term, trunc, info = env.step(a)
        obs = np.asarray(obs)
        if obs.shape != space.shape:
            raise AssertionError(
                f"step {i} obs shape {obs.shape} != space shape {space.shape}"
            )
        if term or trunc:
            break

    try:
        env.close()
    except Exception:
        pass

    return space.shape, obs.dtype


if __name__ == "__main__":
    import sys

    # The compiled PhysiCell module is a per-process singleton (only ONE env
    # can be loaded per runtime), so each mode must run in its own process.
    # Invoked with a mode arg -> test just that mode. No arg -> orchestrate.
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        try:
            shape, dtype = test_mode(mode)
            print(f"RESULT PASS {mode} shape={shape} dtype={dtype}", flush=True)
            sys.exit(0)
        except Exception as e:
            print(f"RESULT FAIL {mode} {e!r}", flush=True)
            sys.exit(1)

    import subprocess

    results = {}
    for mode in KNOWN_GOOD + MODES_TO_TEST:
        tag = "ANCHOR" if mode in KNOWN_GOOD else "TEST  "
        proc = subprocess.run(
            [sys.executable, __file__, mode],
            capture_output=True, text=True,
        )
        out = proc.stdout + proc.stderr
        line = next(
            (l for l in out.splitlines() if l.startswith("RESULT ")), None
        )
        if line is None:
            results[mode] = ("FAIL", f"no result (exit {proc.returncode})")
            print(f"[{tag}] FAIL  {mode}  no RESULT line (exit {proc.returncode})")
            print("  --- tail ---")
            for l in out.splitlines()[-15:]:
                print("  " + l)
        else:
            status = line.split()[1]
            results[mode] = (status, line)
            print(f"[{tag}] {line[len('RESULT '):]}", flush=True)

    print("\n================ SUMMARY ================")
    for mode in KNOWN_GOOD + MODES_TO_TEST:
        status, _ = results[mode]
        print(f"  {status:4s}  {mode}")
