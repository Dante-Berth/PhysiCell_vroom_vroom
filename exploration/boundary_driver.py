"""Fan the boundary-influx scenario grid out over worker processes.

    python boundary_driver.py --run-tag vera_report --workers 8

Answers the question Vera asked on 2026-09-10: what does a drug that enters through the
domain boundary do, as against one injected as a disc? The boundary arms run on the
STANDALONE PhysiCell binary with no PhysiGym in the loop, which is the constraint she set.

The grid is a cross of three things, because the answer depends on all three and reporting
one value of any of them would be reporting an artefact:

  faces           all four (her "flow in from all sides, to avoid asymmetries")
                  vs xmin only (her "as if it was blood flowing in from the circulation")
  diffusion D     0.3 / 3 / 30 / 300. The drug only reaches the middle if the diffusion
                  length sqrt(D/0.05) is comparable to the 63 um domain, and 0.3 is the
                  value proposed in the discussion, so it has to be in the grid.
  boundary value  0.25 / 0.5 / 1.0, straddling the Hill half-max of 0.5 at which the drug
                  actually repolarises a macrophage (cell_rules.csv:2-3). A single value
                  risks reporting "nothing happened" when the truth is "not at that
                  concentration".

Plus an untreated control per initial condition, which is the reference every arm is read
against and which must show drug identically zero.

Each task is one OS process running `./project`, so there is no shared state to corrupt
and no per-process singleton to work around: the standalone binary has neither of the
constraints the gym-driven harness does.
"""
import argparse
import concurrent.futures as cf
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PYTHON = sys.executable

PAPER = {"tumor": 128, "macrophage": 32, "t_cell": 32}
FAMILIES = ["network_field", "rectangle"]
DIFFUSIONS = [0.3, 3.0, 30.0, 300.0]
VALUES = [0.25, 0.5, 1.0]
ARMS = {
    "boundary_all": ["xmin", "xmax", "ymin", "ymax"],
    "boundary_one": ["xmin"],
}


def make_ics(run_dir, families, seeds):
    """Generate the initial conditions once, offline, and share them across every arm.

    Done here rather than inside each worker so that every arm, standalone and
    gym-driven alike, starts from byte-identical cell positions. The IC generator
    is pure python and writes a CSV, so nothing about it needs PhysiGym at run time.
    """
    sys.path.insert(0, os.path.join(
        PROJECT_ROOT, "custom_modules/physigym/physigym/envs"))
    from init_conds import generate_initial_condition

    ic_dir = os.path.join(run_dir, "ics")
    os.makedirs(ic_dir, exist_ok=True)
    out = {}
    for fam, seed in itertools.product(families, seeds):
        csv = os.path.join(ic_dir, f"ic_{fam}_s{seed}.csv")
        # The wrapper shrinks the domain to 90%; match it, as dose_run.py does.
        df, _ = generate_initial_condition(
            csv_path=csv, mode=fam,
            x_min=0.0 * 0.9, x_max=63.0 * 0.9,
            y_min=0.0 * 0.9, y_max=63.0 * 0.9,
            params={k: {"correlation_length": 45, "threshold": 0.55,
                        "number_cells": v} for k, v in PAPER.items()},
            seed=seed)
        counts = df["type"].value_counts().to_dict()
        # Requested counts are an upper bound: init_conds rounds to the integer grid
        # and drops duplicates, so the realised counts are what the analysis keys on.
        out[(fam, seed)] = (csv, {k: int(counts.get(k, 0))
                                 for k in ("tumor", "macrophage", "t_cell")})
    return out


def build_tasks(run_dir, ics, families, seeds, max_time, interval):
    tasks = []

    def add(arm, faces, value, D, fam, seed, cid):
        csv, realised = ics[(fam, seed)]
        tasks.append(dict(config_id=cid, arm=arm, faces=faces, boundary_value=value,
                          drug_diffusion=D, ic_family=fam, seed=seed,
                          ic_csv=csv, realised=realised, run_dir=run_dir,
                          max_time=max_time, interval=interval))

    for fam, seed in itertools.product(families, seeds):
        # The untreated reference. No faces enabled, so drug_1 must stay identically
        # zero; that is the control that makes every other arm readable.
        add("untreated", [], 0.0, 0.3, fam, seed, f"untreated__{fam}__s{seed}")

    for arm, (fam, seed, D, v) in itertools.product(
            ARMS, itertools.product(families, seeds, DIFFUSIONS, VALUES)):
        add(arm, ARMS[arm], v, D,
            fam, seed, f"{arm}__D{D:g}__v{v:g}__{fam}__s{seed}")
    return tasks


def run_task(task):
    tmp = os.path.join(task["run_dir"], "tasks")
    os.makedirs(tmp, exist_ok=True)
    tj = os.path.join(tmp, task["config_id"] + ".json")
    json.dump(task, open(tj, "w"))
    t0 = time.time()
    p = subprocess.run([PYTHON, os.path.join(HERE, "boundary_run.py"),
                        "--task-json", tj],
                       capture_output=True, text=True)
    return task["config_id"], p.returncode, time.time() - t0, p.stdout.strip(), p.stderr[-800:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default=time.strftime("boundary_%Y%m%d_%H%M%S"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--hours", type=float, default=50.0,
                    help="simulated horizon; 50 h is what was promised to Vera")
    ap.add_argument("--interval", type=float, default=15.0,
                    help="output interval in minutes; 15 matches the gym step")
    ap.add_argument("--families", default=None)
    ap.add_argument("--only", default=None, help="substring filter, for smoke tests")
    a = ap.parse_args()

    families = a.families.split(",") if a.families else FAMILIES
    seeds = list(range(a.seeds))
    run_dir = os.path.join(HERE, "output", a.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    max_time = a.hours * 60.0

    ics = make_ics(run_dir, families, seeds)
    tasks = build_tasks(run_dir, ics, families, seeds, max_time, a.interval)
    if a.only:
        tasks = [t for t in tasks if a.only in t["config_id"]]

    json.dump({
        "run_tag": a.run_tag, "n_tasks": len(tasks), "seeds": seeds,
        "hours": a.hours, "max_time_min": max_time, "interval_min": a.interval,
        "families": families, "arms": ARMS, "diffusions": DIFFUSIONS,
        "boundary_values": VALUES, "paper_counts": PAPER,
        "binary": os.path.join(PROJECT_ROOT, "project"),
        "physigym_used": False,
        "realised": {f"{k[0]}__s{k[1]}": v[1] for k, v in ics.items()},
    }, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)

    print(f"{len(tasks)} tasks -> {run_dir}  ({a.workers} workers)")
    t0 = time.time()
    bad = 0
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (cid, rc, dt, out, err) in enumerate(
                ex.map(run_task, tasks), 1):
            flag = "" if rc == 0 else f"  RC={rc} {err[-200:]}"
            if rc != 0:
                bad += 1
            print(f"[{i}/{len(tasks)}] {dt:5.1f}s {cid} {out}{flag}", flush=True)
    print(f"done in {time.time()-t0:.0f}s, {bad} failed")


if __name__ == "__main__":
    main()
