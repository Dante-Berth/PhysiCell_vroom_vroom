"""Boundary-influx grid crossed with the initial-condition geometry knobs.

    python boundary_ic_driver.py --run-tag vera_ic --workers 18 --seeds 8

Extends boundary_driver.py, which answered "what does a boundary influx do?" on a single
IC setting and six seeds. Two things were missing and this adds both:

  1. The IC geometry was fixed at correlation_length=45, threshold=0.55. Those govern how
     clumped the network-field tumour is, so the finding "a boundary influx only works
     once the field has stopped being localised" was measured against exactly one
     geometry. Here they are crossed, 3 x 3.
  2. Six seeds supported a median but not a spread. Eight seeds, and the analysis reports
     mean and s.d. alongside the median.

WHY RECTANGLE IS NOT CROSSED WITH THE IC GRID
---------------------------------------------
rectangle_mode (init_conds.py:189-234) reads ONLY number_cells out of params. Its band
width is the literal 0.1 on line 204 and its band positions are random.uniform draws from
hard-coded intervals. correlation_length and threshold are silently ignored.

So crossing rectangle with the 3x3 grid would run nine BYTE-IDENTICAL copies of the same
runs, and any s.d. computed over them would be zero-variance duplicates presented as
replicates. That is a fabricated error bar, so rectangle varies through its seed only and
carries the ic_cell label "base".

Everything else (the arms, the D grid, the boundary values, the 50 h horizon, the pairing
against an untreated run of the same seed AND the same ic cell) is unchanged from
boundary_driver.py, so the two runs are comparable where they overlap.
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

# The IC geometry grid. 45/0.55 is the setting the first sweep used, so it sits in the
# middle of both ranges and the earlier result is recoverable as one cell of this one.
CORR_LENGTHS = [25, 35, 45]
THRESHOLDS = [0.45, 0.55, 0.65]
BASE_CORR, BASE_THRESH = 45, 0.55


def make_ics(run_dir, seeds):
    """Generate every IC once, offline, shared byte-identically across all arms.

    Returns {(ic_cell, family, seed): (csv, realised_counts)}. network_field gets one
    entry per (corr_len, threshold) cell; rectangle gets a single "base" entry per seed,
    for the reason in the module docstring.
    """
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
        for cl, th in itertools.product(CORR_LENGTHS, THRESHOLDS):
            gen(f"cl{cl}_th{th:g}", "network_field", seed, cl, th)
        gen("base", "rectangle", seed, BASE_CORR, BASE_THRESH)
    return out


def build_tasks(run_dir, ics, max_time, interval):
    tasks = []

    # Runs whose raw frames are kept rather than deleted after the time course is
    # written. The field maps on slide 5 are drawn from the .mat frames directly, so a
    # few runs must retain them or that figure stops being reproducible. Seed 0 of the
    # centre IC cell, which is the setting the original sweep used.
    KEEP = {"untreated__network_field__cl45_th0.55__s0",
            "boundary_all__D0.3__v1__network_field__cl45_th0.55__s0",
            "boundary_all__D300__v1__network_field__cl45_th0.55__s0",
            "boundary_one__D300__v1__network_field__cl45_th0.55__s0"}

    def add(arm, faces, value, D, cell, fam, seed, cid):
        csv, realised = ics[(cell, fam, seed)]
        tasks.append(dict(config_id=cid, arm=arm, faces=faces, boundary_value=value,
                          drug_diffusion=D, ic_family=fam, seed=seed,
                          ic_cell=cell, ic_csv=csv, realised=realised,
                          keep_frames=cid in KEEP,
                          run_dir=run_dir, max_time=max_time, interval=interval))

    # One untreated control per (ic_cell, family, seed). Every treated run is read as a
    # difference against the control sharing all three, so a missing control silently
    # turns its whole group into NaN in the analysis.
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
    ap.add_argument("--run-tag", default=time.strftime("boundary_ic_%Y%m%d_%H%M%S"))
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--hours", type=float, default=50.0)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--only", default=None, help="substring filter, for smoke tests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip tasks whose episode meta.json already records a "
                         "completed run, so an interrupted sweep continues rather "
                         "than repeating work")
    a = ap.parse_args()

    seeds = list(range(a.seeds))
    run_dir = os.path.join(HERE, "output", a.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    max_time = a.hours * 60.0

    ics = make_ics(run_dir, seeds)
    tasks = build_tasks(run_dir, ics, max_time, a.interval)
    if a.only:
        tasks = [t for t in tasks if a.only in t["config_id"]]

    # Assert the pairing is complete BEFORE burning CPU: every treated task must have an
    # untreated control on its own (ic_cell, family, seed). Checked against the FULL
    # design, before --resume removes finished work, because a control that already ran
    # still pairs its group.
    controls = {(t["ic_cell"], t["ic_family"], t["seed"])
                for t in tasks if t["arm"] == "untreated"}
    orphans = {(t["ic_cell"], t["ic_family"], t["seed"]) for t in tasks
               if t["arm"] != "untreated"} - controls
    if orphans and not a.only:
        sys.exit(f"ABORT: {len(orphans)} treated groups have no untreated control: "
                 f"{sorted(orphans)[:5]}")

    if a.resume:
        # A task counts as done only if its meta.json says the run actually wrote
        # output and loaded its rules. A half-written episode directory is NOT done:
        # cell rules load silently empty on a bad cwd, and such a run would otherwise
        # be skipped forever while carrying no treatment at all.
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
        "corr_lengths": CORR_LENGTHS, "thresholds": THRESHOLDS,
        "paper_counts": PAPER, "physigym_used": False,
        "rectangle_crossed_with_ic_grid": False,
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
