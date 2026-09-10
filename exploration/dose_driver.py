"""Dose ladder: fan (arm x dose x initial-condition family) out over workers.

    python exploration/dose_driver.py --run-tag dose_ladder_v1 --workers 12

The untreated study (driver.py) answers "what does the model do on its own".
This answers the two questions that one cannot reach:

  1. Does the model respond to the drug at all, and how sharply?
  2. What does a step of treatment COST under the reward the sweeps optimised?

(2) is the reason this is worth the compute. wrapper.py:758 pays
`w_cell * r_cancer_cells - w_dose * dose_spent`, and dose_spent is
`drug_1_amount_used / total_volume` -- the mass PhysiCell consumed over the
whole domain volume. add_local_substrate accumulates `r_dose * voxel_volume`
over the voxels inside the disc, so

    dose_spent = commanded dose x (fraction of the domain the disc covers)

The uniform disc is the half-diagonal and covers the domain entirely; the
largest targeted disc covers 6.3% of it. So the SAME commanded dose is charged
about sixteen times more in the uniform action space than in the targeted one,
and that has nothing to do with what the drug achieves. Measuring both arms on
one ladder prices that difference without training anything.

Neither w_cell nor w_dose is baked in here. r_tumor and dose_spent are recorded
per step, so any composite reward can be recomputed offline for any weighting.

Arms:
  uniform  -- domain centre, radius = max_radius. Reproduces action_mode="full".
  targeted -- re-aimed at the macrophage centre of mass, radius = 0.20*max_radius,
              the largest disc the targeted action space permits.

Dose modes:
  const  -- the ladder, one fixed dose for the whole episode
  random -- dose ~ U(0,1) redrawn every step, the random-policy baseline
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import bootstrap

import common as C  # noqa: E402

PYTHON = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

PAPER = dict(tumor=128, macrophage=32, t_cell=32)
FAMILIES = ["network_field", "rectangle"]  # the trained and the held-out family
ARMS = ["uniform", "targeted", "targeted_dense",
        "grid_macro", "grid_m2", "grid_m2_tumor"]
DOSES = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0


WARMUP = [0]
NOTERM = [False]
DIFF = [None]


def build_tasks(seeds, max_steps, run_dir):
    tasks = []

    def add(arm, dose_mode, dose, family, cid):
        tasks.append(dict(config_id=cid, arm=arm, dose_mode=dose_mode, dose=dose,
                          ic_family=family, counts=dict(PAPER), seeds=list(seeds),
                          max_steps=max_steps, run_dir=run_dir,
                          warmup=WARMUP[0], no_terminate=NOTERM[0],
                          drug_diffusion=DIFF[0]))

    for fam, arm, dose in itertools.product(FAMILIES, ARMS, DOSES):
        # dose 0 is the same simulation in both arms (add_local_substrate returns
        # early), but it is run in both so every arm carries its own control and
        # the ladder does not borrow a baseline from the other one.
        add(arm, "const", dose, fam, f"{arm}__d{dose:0.1f}__{fam}")
    for fam, arm in itertools.product(FAMILIES, ARMS):
        add(arm, "random", None, fam, f"{arm}__random__{fam}")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default=time.strftime("dose_%Y%m%d_%H%M%S"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=480)
    ap.add_argument("--drug-diffusion", type=float, default=None,
                    help="override drug_1 diffusion_coefficient (shipped: 0)")
    ap.add_argument("--no-terminate", action="store_true",
                    help="disable the c_t>256 / c_t<=3 terminations so every run "
                         "reaches the horizon and no burden is censored")
    ap.add_argument("--families", default=None,
                    help="comma-separated IC families (default: the train and "
                         "held-out pair the thesis uses)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of ARMS to run")
    ap.add_argument("--warmup", type=int, default=0,
                    help="untreated steps before treatment starts (fig:tme:aim "
                         "uses 60)")
    ap.add_argument("--split-seeds", action="store_true",
                    help="one process per seed; works around the segfault on the "
                         "second env.reset() in a worker after a treated episode")
    ap.add_argument("--only", default=None,
                    help="substring filter on config_id, for smoke tests")
    args = ap.parse_args()

    run_dir = os.path.join(HERE, "output", args.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    seeds = list(range(args.seeds))
    WARMUP[0] = args.warmup
    NOTERM[0] = args.no_terminate
    DIFF[0] = args.drug_diffusion
    if args.families:
        FAMILIES[:] = args.families.split(",")
    if args.arms:
        keep = args.arms.split(",")
        ARMS[:] = [a for a in ARMS if a in keep]
    tasks = build_tasks(seeds, args.max_steps, run_dir)
    if args.split_seeds:
        tasks = [dict(t, seeds=[s]) for t in tasks for s in t["seeds"]]
    if args.only:
        tasks = [t for t in tasks if args.only in t["config_id"]]
    n_ep = sum(len(t["seeds"]) for t in tasks)

    so, pgym = bootstrap.assert_all()
    C.dump_json(os.path.join(run_dir, "manifest.json"), {
        "run_tag": args.run_tag, "n_tasks": len(tasks), "n_episodes": n_ep,
        "seeds": seeds, "max_steps": args.max_steps,
        "dt_gym": C.DT_GYM, "max_time_min": args.max_steps * C.DT_GYM,
        "env_kwargs": {k: v for k, v in C.BASE_ENV_KWARGS.items()},
        "paper_counts": PAPER, "families": FAMILIES, "arms": ARMS, "doses": DOSES,
        "radius_max_norm": 0.20, "warmup": args.warmup,
        "no_terminate": args.no_terminate,
        "drug_diffusion": args.drug_diffusion,
        "so": so, "so_mtime": os.path.getmtime(so), "physigym": pgym,
        "python": PYTHON,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=bootstrap.PROJECT_ROOT,
            capture_output=True, text=True).stdout.strip(),
    })
    print(f"{len(tasks)} tasks / {n_ep} episodes -> {run_dir}", flush=True)

    logs = os.path.join(run_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    queue, running, failed = list(enumerate(tasks)), [], []
    t0 = time.perf_counter()
    done = 0

    while queue or running:
        while queue and len(running) < args.workers:
            i, task = queue.pop(0)
            task = dict(task, worker_id=i % args.workers)
            fh = open(os.path.join(logs, f"{task['config_id']}.log"), "w")
            p = subprocess.Popen(
                [PYTHON, os.path.join(HERE, "dose_run.py"), json.dumps(task)],
                cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            running.append((p, task, fh))
        time.sleep(1.0)
        for entry in list(running):
            p, task, fh = entry
            if p.poll() is None:
                continue
            running.remove(entry)
            fh.close()
            done += 1
            if p.returncode != 0:
                failed.append(task["config_id"])
            el = time.perf_counter() - t0
            print(f"[{done}/{len(tasks)}] {task['config_id']} rc={p.returncode} "
                  f"({el/60:.1f} min elapsed)", flush=True)

    print(f"\ndone in {(time.perf_counter()-t0)/60:.1f} min; {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
