"""Shared pieces: XML preparation, env construction, encoder registry, recording.

Design notes that matter:

* We drive the RAW env, not PhysiCellModelWrapper. The wrapper deletes files in
  its output directory on reset, rewrites the settings XML on disk, and anchors
  its action-delta clipping at the midpoint of the action space (dose 0.5). None
  of that is wanted for an untreated study.
* action_mode="full" gives a single-key action space {"drug_1_dose"}, so the
  action dict is complete. drug_1_x/y/radius keep their XML values, which we pin
  to 0. add_local_substrate returns early when dose<=0 OR radius<=0, so zero drug
  is guaranteed twice over.
* With dose=0 the observation mode does not affect the simulation, so we run ONE
  simulation per parameter point and call every feature builder directly on the
  same instant. That is what makes "many state spaces" cheap.
"""

import json
import os
import shutil

import numpy as np
from lxml import etree

import bootstrap  # noqa: F401  (side effects: sys.path, cwd, gomp)

PROJECT_ROOT = bootstrap.PROJECT_ROOT
BASE_XML = os.path.join(PROJECT_ROOT, "config/PhysiCell_settings.xml")

DT_GYM = 15.0  # min per gym step, //user_parameters/dt_gym

BASE_ENV_KWARGS = dict(
    id="physigym/ModelPhysiCellEnv-v0",
    render_mode=None,  # MANDATORY: get_img() queries substrates that no longer exist
    verbose=False,
    observation_mode="scalars_cells",  # cheapest; we compute encodings ourselves
    action_mode="full",  # -> action space is {"drug_1_dose"} only
    k=1,  # run.py never passes k, so the paper value is the default 1
    grid_n=8,
    img_mc_grid_size_x=64,
    img_mc_grid_size_y=64,
    normalization_factor=128,  # run.py:1574 uses args.tumor (=128), not 512
    disable_env_checker=True,
)

NOOP_ACTION = {"drug_1_dose": np.array([0.0], dtype=np.float32)}

# Paper parameterisation (run.py:1639-1655, run.py:1447-1449)
PAPER_COUNTS = {"tumor": 128, "macrophage": 32, "t_cell": 32}
CORR_LENGTH = 45
THRESHOLD = 0.55


def make_params(tumor, macrophage, t_cell):
    return {
        "tumor": {
            "correlation_length": CORR_LENGTH,
            "threshold": THRESHOLD,
            "number_cells": tumor,
        },
        "macrophage": {
            "correlation_length": CORR_LENGTH,
            "threshold": THRESHOLD,
            "number_cells": macrophage,
        },
        "t_cell": {
            "correlation_length": CORR_LENGTH,
            "threshold": THRESHOLD,
            "number_cells": t_cell,
        },
    }


def prepare_xml(dst, save_folder, max_time, omp=1):
    """Copy the tracked settings XML and pin everything we control.

    Never mutates config/PhysiCell_settings.xml itself.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    shutil.copy(BASE_XML, dst)
    tree = etree.parse(dst)
    root = tree.getroot()

    def put(xpath, value):
        root.xpath(xpath)[0].text = str(value)

    put("//overall/max_time", float(max_time))
    put("//parallel/omp_num_threads", omp)
    put("//save/folder", save_folder)
    put("//save/SVG/enable", "false")
    put("//save/full_data/enable", "false")
    # setup_tissue()'s random scatter loop must stay off: every cell comes from
    # the generated initial-condition CSV.
    put("//user_parameters/number_of_cells", 0)
    for p in ("drug_1", "drug_1_x", "drug_1_y", "drug_1_radius", "drug_1_dose",
              "drug_1_amount_used"):
        put(f"//user_parameters/{p}", 0)
    tree.write(dst, pretty_print=True)
    return tree, root


def point_xml_at_ic(tree, root, dst, ic_csv, seed):
    """Between episodes only the IC path and the seed change."""
    root.xpath("//initial_conditions/cell_positions/folder")[0].text = str(
        os.path.dirname(ic_csv)
    )
    root.xpath("//initial_conditions/cell_positions/filename")[0].text = str(
        os.path.basename(ic_csv)
    )
    root.xpath("//random_seed")[0].text = str(int(seed))
    tree.write(dst, pretty_print=True)


# ── encoder registry ────────────────────────────────────────────────────────
# Every published observation mode in physicell_model.get_observation() is a
# concatenation of these primitives, so recording all of them lets any state
# space be reassembled offline without re-running the simulation.

SCALAR_ENCODERS = [
    ("cells", "get_cells_scalars"),
    ("substrates", "get_substrates_scalars"),
    ("macro_pol", "get_macrophage_polarization_scalars"),
    ("spatial", "get_spatial_features"),
    ("spatial_m1m2", "get_spatial_features_m1m2"),
    ("spatial_subs", "get_spatial_substrate_features"),
    ("relational", "get_relational_features"),
    ("cross_nn", "get_cross_nn_features"),
    ("occupancy", "get_occupancy_grid"),
]

IMAGE_ENCODERS = [
    ("img_cells", "get_matrix_cells"),
    ("img_subs", "get_matrix_substrates"),
    ("img_m1m2", "get_matrix_macrophage_polarization"),
]

# How the paper's composite modes decompose (recorded in the manifest so the
# offline reconstruction is documented rather than folklore).
COMPOSITES = {
    "scalars_cells": ["cells"],
    "scalars_substrates": ["substrates"],
    "scalars_cells_substrates": ["cells", "substrates"],
    "scalars_macrophages": ["macro_pol"],
    "spatial_scalars_cells": ["cells", "spatial"],
    "spatial_scalars_cells_substrates": ["cells", "substrates", "spatial"],
    "spatial_scalars_cells_m1m2": ["macro_pol", "spatial_m1m2"],
    "spatial_scalars_cells_substrates_m1m2": [
        "macro_pol", "substrates", "spatial_m1m2",
    ],
    "spatial_scalars_cells_spatial_substrates": [
        "cells", "substrates", "spatial", "spatial_subs",
    ],
    "kmeans_spatial_scalars_cells_substrates": ["spatial", "spatial_subs"],
    "relational": ["relational"],
    "cross_nn_relational": ["relational", "cross_nn"],
    "occupancy_grid": ["occupancy"],
    "img_mc_cells": ["img_cells"],
    "img_mc_substrates": ["img_subs"],
    "img_mc_cells_substrates": ["img_cells", "img_subs"],
    "img_mc_cells_m1m2": ["img_cells", "img_m1m2"],
    "img_mc_cells_substrates_m1m2": ["img_cells", "img_subs", "img_m1m2"],
}


def population_counts(df_alive):
    t = df_alive["type"]
    return {
        "n_tumor": int((t == "tumor").sum()),
        "n_t_cell": int((t == "t_cell").sum()),
        "n_macrophage": int((t == "macrophage").sum()),
    }


def m1m2_split(env):
    """M1/M2 counts, replicating get_macrophage_polarization_scalars exactly.

    get_microenv(name) returns (N,4) rows of x,y,z,concentration. Nearest voxel
    by cKDTree over the xy columns; pro > anti => M2.
    """
    from scipy.spatial import cKDTree
    from extending import physicell

    u = env.unwrapped
    mac = u.df_alive[u.df_alive["type"] == "macrophage"]
    if len(mac) == 0:
        return {"n_M1": 0, "n_M2": 0, "m2_fraction": float("nan")}
    pro = np.asarray(physicell.get_microenv("pro_tumoral_factor"))
    anti = np.asarray(physicell.get_microenv("anti_tumoral_factor"))
    tree = cKDTree(pro[:, :2])
    _, nearest = tree.query(mac[["x", "y"]].to_numpy())
    n_m2 = int((pro[:, -1][nearest] > anti[:, -1][nearest]).sum())
    n_m1 = int(len(mac) - n_m2)
    return {"n_M1": n_m1, "n_M2": n_m2, "m2_fraction": float(n_m2 / len(mac))}


def substrate_max(name):
    """Max concentration over all voxels, matching get_substrates_scalars."""
    from extending import physicell

    return float(np.asarray(physicell.get_microenv(name))[:, -1].max())


def geometry(df_alive):
    """Interpretable spatial ground truth the encodings get judged against."""
    out = {}
    sub = {k: df_alive[df_alive["type"] == k] for k in
           ("tumor", "t_cell", "macrophage")}
    for name, d in sub.items():
        if len(d):
            out[f"{name}_cx"] = float(d["x"].mean())
            out[f"{name}_cy"] = float(d["y"].mean())
        else:
            out[f"{name}_cx"] = float("nan")
            out[f"{name}_cy"] = float("nan")
    tum, tc = sub["tumor"], sub["t_cell"]
    if len(tum):
        out["tumor_rg"] = float(
            np.sqrt(((tum["x"] - tum["x"].mean()) ** 2
                     + (tum["y"] - tum["y"].mean()) ** 2).mean())
        )
    else:
        out["tumor_rg"] = float("nan")
    if len(tum) and len(tc):
        from scipy.spatial import cKDTree

        d, _ = cKDTree(tc[["x", "y"]].to_numpy()).query(tum[["x", "y"]].to_numpy())
        out["d_tumor_tcell_centroid"] = float(
            np.hypot(out["tumor_cx"] - out["t_cell_cx"],
                     out["tumor_cy"] - out["t_cell_cy"])
        )
        out["nn_tumor_to_tcell_mean"] = float(d.mean())
        out["nn_tumor_to_tcell_p10"] = float(np.percentile(d, 10))
        out["frac_tumor_within_3um_of_tcell"] = float((d <= 3.0).mean())
    else:
        for kk in ("d_tumor_tcell_centroid", "nn_tumor_to_tcell_mean",
                   "nn_tumor_to_tcell_p10", "frac_tumor_within_3um_of_tcell"):
            out[kk] = float("nan")
    return out


def classify_end(terminated, truncated, c_t):
    if terminated and c_t <= 3:
        return "eradication"
    if terminated and c_t > 256:
        return "escape"
    if truncated:
        return "truncation"
    return "running"


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=json_default)
