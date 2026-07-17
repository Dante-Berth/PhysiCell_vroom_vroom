"""Single-process random-action driver to reproduce the PhysiCell segfault
under gdb. No SubprocVecEnv — the sim runs in THIS process so gdb sees the
crash frame. Single-threaded (threads_per_env=1) to remove OpenMP noise.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "custom_modules/physigym/physigym/envs")
sys.path.insert(0, "custom_modules")
sys.path.insert(0, ".")

from vectorized import test_make_physigym_env  # noqa: E402

cfg = {
    "simulation": {"max_time": 100000.0, "seed": 1},
    # force threads_per_env = (total_cores - rl_threads)//num_envs -> 1
    "vectorization": {"num_envs": 1, "rl_threads": os.cpu_count() - 1},
    "model": {
        "id": "physigym/ModelPhysiCellEnv-v0",
        "settingxml": "config/PhysiCell_settings.xml",
        "settingcells": "config/cells.csv",
        "action_mode": "targeted",
        "output_dir": "./segfault_hunt_output",
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
    },
    "generation": {
        "x_min": 0, "x_max": 64, "y_min": 0, "y_max": 64,
        "params": {
            "tumor": {"correlation_length": 45, "threshold": 0.55, "number_cells": 64},
            "t_cell": {"correlation_length": 45, "threshold": 0.55, "number_cells": 8},
            "macrophage": {"correlation_length": 45, "threshold": 0.55, "number_cells": 32},
        },
        "seed": 1,
        "mode_train": "network_field",
        "mode_test": ["rectangle", "circular", "network_field"],
    },
    "rl": {"total_timesteps": 100000},
}

env = test_make_physigym_env(cfg)
env.reset()

rng = np.random.default_rng(1)
ACTION_REPEAT = 6
step = 0
MAX = 40000
print("[driver] starting random loop", flush=True)
while step < MAX:
    raw = rng.random(4).astype(np.float32)
    for _ in range(ACTION_REPEAT):
        obs, r, term, trunc, info = env.step(raw)
        step += 1
        if step % 50 == 0:
            print(f"[driver] step={step}", flush=True)
        if term or trunc:
            env.reset()
            break
print("[driver] finished without crash", flush=True)
