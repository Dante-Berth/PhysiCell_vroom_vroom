"""Boundary influx on BLOB initial conditions: threshold swept, correlation_length fixed.

    python boundary_blob_driver.py --run-tag vera_blob --workers 16 --seeds 8

Sibling of boundary_corrlen_driver.py, which swept correlation_length with the
threshold PINNED at 0.55. That pin is why no panel of its output looks like a blob:
measured over a 7x6 grid, the threshold -- not the correlation length -- is what
compacts the population. At correlation_length 45 the tumour occupies ~37% of the
domain at threshold 0.55, ~10% at 0.85 and ~3% at 0.95, in a single connected
component in every case above 0.85.

So this sweep holds correlation_length at 45 (the value used everywhere else in the
thesis) and sweeps THRESHOLD over {0.55, 0.75, 0.85, 0.95}: from the diffuse field
the corrlen sweep already covered, to one compact blob per cell type. 0.55 is kept
as the overlap point, so this grid can be checked against the existing one.

Why blobs matter beyond the picture: the blob's POSITION varies by seed while its
compactness does not, so "where is the tumour relative to the dosed wall" becomes a
per-episode question. That is the geometry a one-sided influx can act on and an
aggregate cell count cannot report.

⚠️ At threshold 0.95 with a SHORT correlation length the mask can collapse to a few
voxels and all 128 tumour cells stack on ~1% of the domain. correlation_length is
held at 45 here precisely to stay out of that corner; the realised counts are
recorded per episode regardless.

rectangle is NOT crossed with this grid, for the same reason as the two sibling
drivers: rectangle_mode ignores correlation_length and threshold entirely, so
crossing it would produce duplicate runs and a fabricated s.d. It is run once per
seed, tagged ic_cell="base".
"""
import argparse
import concurrent.futures as cf
import glob
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
DIFFUSIONS = [0.3, 3.0, 30.0, 300.0]
VALUES = [0.25, 0.5, 1.0]
ARMS = {"boundary_all": ["xmin", "xmax", "ymin", "ymax"],
        "boundary_one": ["xmin"]}

# The single-factor grid: THRESHOLD varies, correlation_length is pinned at 45, the
# value used everywhere else in the thesis. 0.55 overlaps the corrlen sweep so the two
# grids can be checked against each other at that point.
THRESHOLDS = [0.55, 0.75, 0.85, 0.95]
FIXED_CORR = 45
BASE_CORR, BASE_THRESH = 45, 0.55   # for the rectangle "base" IC, unchanged


def make_ics(run_dir, seeds):
    sys.path.insert(0, os.path.join(
        PROJECT_ROOT, "custom_modules/physigym/physigym/envs"))
    from init_conds import generate_initial_condition

    ic_dir = os.path.join(run_dir, "ics")
    os.makedirs(ic_dir, exist_ok=True)
    out = {}

    def gen(cell, fam, seed, cl, th):
        csv = os.path.join(ic_dir, f"ic_{fam}_{cell}_s{seed}.csv")
        df, _ = generate_initial_condition(
            csv_path=csv, mode=fam,
            x_min=0.0 * 0.9, x_max=63.0 * 0.9,
            y_min=0.0 * 0.9, y_max=63.0 * 0.9,
            params={k: {"correlation_length": cl, "threshold": th,
                        "number_cells": v} for k, v in PAPER.items()},
            seed=seed)
        counts = df["type"].value_counts().to_dict()
        out[(cell, fam, seed)] = (csv, {k: int(counts.get(k, 0))
                                       for k in ("tumor", "macrophage", "t_cell")})

    for seed in seeds:
        for th in THRESHOLDS:
            gen(f"th{th:g}", "network_field", seed, FIXED_CORR, th)
        gen("base", "rectangle", seed, BASE_CORR, BASE_THRESH)
    return out


def build_tasks(run_dir, ics, max_time, interval):
    tasks = []

    # Keep raw frames for none of these: this sweep is purely for the timecourse/grid
    # analysis, which reads field.csv.gz, not the .mat frames. Slide 5's field figure
    # already has its own dedicated runs from boundary_ic_driver.py.
    def add(arm, faces, value, D, cell, fam, seed, cid):
        csv, realised = ics[(cell, fam, seed)]
        tasks.append(dict(config_id=cid, arm=arm, faces=faces, boundary_value=value,
                          drug_diffusion=D, ic_family=fam, seed=seed,
                          ic_cell=cell, ic_csv=csv, realised=realised,
                          keep_frames=False,
                          run_dir=run_dir, max_time=max_time, interval=interval))

    for (cell, fam, seed) in ics:
        add("untreated", [], 0.0, 0.3, cell, fam, seed,
            f"untreated__{fam}__{cell}__s{seed}")

    for (cell, fam, seed) in ics:
        for arm, D, v in itertools.product(ARMS, DIFFUSIONS, VALUES):
            add(arm, ARMS[arm], v, D, cell, fam, seed,
                f"{arm}__D{D:g}__v{v:g}__{fam}__{cell}__s{seed}")
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
    ap.add_argument("--run-tag", default=time.strftime("boundary_blob_%Y%m%d_%H%M%S"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--hours", type=float, default=50.0)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--only", default=None, help="substring filter, for smoke tests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    seeds = list(range(a.seeds))
    run_dir = os.path.join(HERE, "output", a.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    max_time = a.hours * 60.0

    ics = make_ics(run_dir, seeds)
    tasks = build_tasks(run_dir, ics, max_time, a.interval)
    if a.only:
        tasks = [t for t in tasks if a.only in t["config_id"]]

    controls = {(t["ic_cell"], t["ic_family"], t["seed"])
                for t in tasks if t["arm"] == "untreated"}
    orphans = {(t["ic_cell"], t["ic_family"], t["seed"]) for t in tasks
               if t["arm"] != "untreated"} - controls
    if orphans and not a.only:
        sys.exit(f"ABORT: {len(orphans)} treated groups have no untreated control: "
                 f"{sorted(orphans)[:5]}")

    if a.resume:
        done = set()
        for mp in glob.glob(os.path.join(run_dir, "episodes", "*", "meta.json")):
            try:
                m = json.load(open(mp))
            except (ValueError, OSError):
                continue
            if m.get("wrote_output") and not m.get("error") and m.get("n_rules_loaded"):
                done.add(m.get("config_id"))
        before = len(tasks)
        tasks = [t for t in tasks if t["config_id"] not in done]
        print(f"resume: {len(done)} already complete, "
              f"{before - len(tasks)} skipped, {len(tasks)} to run")

    json.dump({
        "run_tag": a.run_tag, "n_tasks": len(tasks), "seeds": seeds,
        "hours": a.hours, "max_time_min": max_time, "interval_min": a.interval,
        "arms": ARMS, "diffusions": DIFFUSIONS, "boundary_values": VALUES,
        "thresholds": THRESHOLDS, "fixed_correlation_length": FIXED_CORR,
        "paper_counts": PAPER, "physigym_used": False,
        "rectangle_crossed_with_grid": False,
        "binary": os.path.join(PROJECT_ROOT, "project"),
        "realised": {f"{c}__{f}__s{s}": v[1] for (c, f, s), v in ics.items()},
    }, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)

    print(f"{len(tasks)} tasks -> {run_dir}  ({a.workers} workers)")
    print(f"  {len(controls)} untreated controls, 0 orphaned treated groups")
    if a.dry_run:
        return

    t0 = time.time()
    bad = 0
    log = open(os.path.join(run_dir, "driver.log"), "w")
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (cid, rc, dt, out, err) in enumerate(ex.map(run_task, tasks), 1):
            flag = "" if rc == 0 else f"  RC={rc} {err[-200:]}"
            if rc != 0:
                bad += 1
            el = time.time() - t0
            eta = el / i * (len(tasks) - i)
            line = (f"[{i}/{len(tasks)}] {dt:5.1f}s eta={eta/60:5.1f}m "
                    f"{cid} {out}{flag}")
            print(line, flush=True)
            log.write(line + "\n"); log.flush()
    log.close()
    print(f"done in {time.time()-t0:.0f}s, {bad} failed")


if __name__ == "__main__":
    main()
