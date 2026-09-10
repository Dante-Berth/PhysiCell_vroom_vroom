"""One standalone PhysiCell run with a Dirichlet boundary influx. No PhysiGym.

Vera's scenario: the drug does not arrive as an injected disc, it enters through the
domain boundary "as if it was blood flowing in from the circulation", either from one
side or from all four.

That actuator has no (x, y, radius) at all, so it cannot be aimed even in principle.
It is a DIFFERENT actuator from the disc injection the thesis studies, not a variant of
it, and this module exists to measure what it does.

Why this runs standalone and the disc arms do not
-------------------------------------------------
A boundary influx is a Dirichlet condition: PhysiCell parses it out of the settings XML
(PhysiCell_settings.cpp:764-853) and BioFVM applies it (BioFVM_microenvironment.cpp:1466+),
so nothing python needs to be in the loop. `./project <xml>` is the whole invocation.

The disc injection cannot: `add_local_substrate` (custom.cpp:224) has NO C++ caller. It
is reached only from physicellmodule.cpp:283, under the python bridge. So the disc arms
need PhysiGym and the boundary arms do not, and the report says so rather than hiding it.

Traps this module exists to absorb (each cost a run to find)
-----------------------------------------------------------
* main.cpp:49 hard-codes a FOUR-episode loop and overrides //save/folder with
  output/episode0000000N. Output does not go where the XML asks. We run with cwd set to
  a per-task scratch directory so those four land somewhere private, and read episode 0.
* The tracked XML's initial-condition path names ic_000055.csv, which does not exist.
  The binary exits with "not found during cell loading. Quitting." Every run must be
  pointed at a generated IC first.
* //save/full_data/enable is pinned false in the tracked XML, so a run writes nothing
  and still exits 0. It must be turned back on or there is no data to read.
* Setting a face's `enabled` attribute without also setting its VALUE silently clamps
  that face to 0, which is indistinguishable from "the drug never arrived".
* The binary prints "which boundaries?" followed by six flags
  (BioFVM_microenvironment.cpp:1497). That line is the ground truth for which faces are
  live; we parse it back and record it, rather than trusting the XML we wrote.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
BASE_XML = os.path.join(PROJECT_ROOT, "config/PhysiCell_settings.xml")
BINARY = os.path.join(PROJECT_ROOT, "project")

DT_GYM = 15.0          # min, to match the gym-driven arms step for step
DRUG_DECAY = 0.05      # //drug_1/decay_rate, sets the diffusion length sqrt(D/lambda)
FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")

# Substrate row order inside final_microenvironment0.mat: x, y, z, volume, then one row
# per substrate in XML declaration order. drug_1 is the third substrate, hence row 6.
SUB_ROWS = {"anti_tumoral_factor": 4, "pro_tumoral_factor": 5, "drug_1": 6,
            "tumor_molecule": 7, "cytokine": 8}


def build_xml(dst, save_folder, ic_csv, seed, max_time, faces, value, drug_D,
              interval=DT_GYM):
    """Copy the tracked settings XML and set exactly what this scenario controls.

    Never mutates config/PhysiCell_settings.xml, mirroring common.prepare_xml.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    shutil.copy(BASE_XML, dst)
    tree = etree.parse(dst)
    root = tree.getroot()

    def put(xpath, v):
        root.xpath(xpath)[0].text = str(v)

    put("//overall/max_time", float(max_time))
    put("//parallel/omp_num_threads", 1)
    put("//save/folder", save_folder)
    # ⚠️ Both are pinned false in the tracked XML. Without this the run writes nothing
    # and still exits 0, which looks exactly like a successful empty simulation.
    put("//save/full_data/enable", "true")
    put("//save/full_data/interval", float(interval))
    put("//save/SVG/enable", "false")
    # setup_tissue()'s random scatter loop must stay off: every cell comes from the CSV.
    put("//user_parameters/number_of_cells", 0)
    # No disc injection in any standalone arm. add_local_substrate is unreachable from
    # here anyway, but pin the slots so the intent is on the record.
    for p in ("drug_1", "drug_1_x", "drug_1_y", "drug_1_radius", "drug_1_dose",
              "drug_1_amount_used"):
        put(f"//user_parameters/{p}", 0)

    put("//initial_conditions/cell_positions/folder", os.path.dirname(ic_csv))
    put("//initial_conditions/cell_positions/filename", os.path.basename(ic_csv))
    put("//random_seed", int(seed))

    v = root.xpath("//microenvironment_setup/variable[@name='drug_1']")[0]
    v.xpath("./physical_parameter_set/diffusion_coefficient")[0].text = repr(float(drug_D))

    # The top-level condition seeds all six faces (PhysiCell_settings.cpp:748-760); the
    # per-face block below then overrides. Enable it iff any face is wanted.
    bc = v.xpath("./Dirichlet_boundary_condition")[0]
    bc.set("enabled", "True" if faces else "False")
    bc.text = repr(float(value)) if faces else "0"
    for f in FACES:
        b = v.xpath(f"./Dirichlet_options/boundary_value[@ID='{f}']")[0]
        on = f in faces
        b.set("enabled", "True" if on else "False")
        # ⚠️ Always write the value, not just the flag.
        b.text = repr(float(value)) if on else "0"

    tree.write(dst, pretty_print=True)
    return dst


def parse_live_faces(stdout):
    """Read back the six flags the binary prints, rather than trusting our own XML."""
    m = re.search(r"which boundaries\?\s*\n\s*([01](?:\s+[01]){5})", stdout)
    if not m:
        return None
    flags = [bool(int(t)) for t in m.group(1).split()]
    return [f for f, on in zip(FACES, flags) if on]


def read_microenv(mat_path):
    from scipy.io import loadmat
    m = loadmat(mat_path)
    key = [k for k in m if not k.startswith("__")][0]
    return m[key]


def field_stats(A, x_hi=63.0):
    """Summarise the drug_1 field: what a cell at each place would experience."""
    x, y, drug = A[0], A[1], A[SUB_ROWS["drug_1"]]
    edge = (x <= 0.5) | (x >= x_hi - 0.5) | (y <= 0.5) | (y >= x_hi - 0.5)
    mid = x_hi / 2.0
    centre = (np.abs(x - mid) < 3) & (np.abs(y - mid) < 3)
    return {
        "drug_max": float(drug.max()),
        "drug_min": float(drug.min()),
        "drug_mean": float(drug.mean()),
        "drug_edge_mean": float(drug[edge].mean()),
        "drug_interior_mean": float(drug[~edge].mean()),
        "drug_centre_mean": float(centre.any() and drug[centre].mean() or 0.0),
        # The drug only repolarises macrophages above the Hill half-max of 0.5
        # (cell_rules.csv:2-3, half_max 0.5, coefficient 4), so the therapeutic
        # question is what FRACTION of the domain clears that, not the mean.
        "frac_above_half_max": float((drug >= 0.5).mean()),
        "frac_above_quarter": float((drug >= 0.25).mean()),
    }


# Row indices inside *_cells.mat, read off the `labels` block of the matching
# output XML and verified against a known initial condition (113 tumour / 29 T /
# 32 macrophage went in, and rows 5 and 26 reproduce exactly that).
CELL_TYPE_ROW, CELL_DEAD_ROW = 5, 26
CELL_TYPES = {0: "tumor", 1: "t_cell", 2: "macrophage"}


def count_cells(mat_path):
    """Live cell counts by type from one MultiCellDS frame.

    This is the y axis Vera asked for. The standalone binary has no equivalent of
    physicell_model's self.c_t, and the legacy simulation_report.txt path is gated
    off, so the counts come from the cell matrix directly.
    """
    A = read_microenv(mat_path)
    ctype, dead = A[CELL_TYPE_ROW], A[CELL_DEAD_ROW]
    alive = dead < 0.5
    out = {f"n_{n}": int(((ctype == k) & alive).sum()) for k, n in CELL_TYPES.items()}
    out["n_alive"] = int(alive.sum())
    out["n_dead"] = int((~alive).sum())
    return out


def run(task):
    """One standalone episode. Returns the meta dict."""
    tag = task["config_id"]
    work = os.path.join(task["run_dir"], "work", tag)
    os.makedirs(work, exist_ok=True)
    save = os.path.join(work, "save")
    xml = os.path.join(work, "settings.xml")
    build_xml(xml, save, task["ic_csv"], task["seed"], task["max_time"],
              tuple(task["faces"]), task["boundary_value"], task["drug_diffusion"])

    # ⚠️ main.cpp writes to ./output/episodeNNNNNNNN relative to CWD, ignoring
    # //save/folder. Run inside `work` so those land here and not in the shared tree.
    proc = subprocess.run([BINARY, xml], cwd=work, capture_output=True, text=True,
                          timeout=task.get("timeout", 1800))
    out = proc.stdout
    live = parse_live_faces(out)

    ep0 = os.path.join(work, "output", "episode00000000")
    final = os.path.join(ep0, "final_microenvironment0.mat")
    ok = os.path.exists(final)

    meta = dict(
        config_id=tag, arm=task["arm"], faces=list(task["faces"]),
        faces_live=live, boundary_value=task["boundary_value"],
        drug_diffusion=task["drug_diffusion"],
        diffusion_length=float(np.sqrt(task["drug_diffusion"] / DRUG_DECAY))
        if task["drug_diffusion"] > 0 else 0.0,
        ic_family=task["ic_family"], seed=task["seed"],
        max_time=task["max_time"], realised=task.get("realised"),
        returncode=proc.returncode, wrote_output=ok,
    )
    # ⚠️ D=300 segfaulted at teardown yet wrote valid output, so a non-zero return code
    # is recorded but is not by itself treated as failure. Presence of data decides.
    if not ok:
        meta["error"] = (out[-2000:] + proc.stderr[-2000:]) or "no output"
        return meta, None

    meta.update(field_stats(read_microenv(final)))
    if live is not None and sorted(live) != sorted(task["faces"]):
        meta["face_mismatch"] = True

    frames = sorted(glob.glob(os.path.join(ep0, "output*_microenvironment0.mat")))
    rows = []
    for i, f in enumerate(frames):
        st = field_stats(read_microenv(f))
        cells = f.replace("_microenvironment0.mat", "_cells.mat")
        if os.path.exists(cells):
            st.update(count_cells(cells))
        st.update(frame=i, sim_time_min=i * task.get("interval", DT_GYM))
        rows.append(st)
    df = pd.DataFrame(rows) if rows else None

    fc = os.path.join(ep0, "final_cells.mat")
    if os.path.exists(fc):
        meta.update({f"final_{k}": v for k, v in count_cells(fc).items()})
    if df is not None and "n_tumor" in df.columns and len(df):
        meta["n_tumor_0"] = int(df["n_tumor"].iloc[0])
        meta["n_tumor_T"] = int(df["n_tumor"].iloc[-1])
    return meta, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    a = ap.parse_args()
    task = json.load(open(a.task_json))
    meta, df = run(task)
    ep = os.path.join(task["run_dir"], "episodes", task["config_id"])
    os.makedirs(ep, exist_ok=True)
    if df is not None:
        df.to_csv(os.path.join(ep, "field.csv.gz"), index=False,
                  compression="gzip")
    json.dump(meta, open(os.path.join(ep, "meta.json"), "w"), indent=2)
    print(json.dumps({k: meta[k] for k in
                      ("config_id", "wrote_output", "faces_live", "drug_centre_mean")
                      if k in meta}))


if __name__ == "__main__":
    main()
