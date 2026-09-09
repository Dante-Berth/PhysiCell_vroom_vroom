"""Fan the parameter grid out over worker processes, then concatenate.

    python exploration/driver.py --run-tag dose0_v1 --workers 12

The grid is a parameter analysis of the untreated model: does the qualitative
regime (immunosuppression establishes itself, tumour escapes) survive changes to
the parameters that were chosen rather than measured?

Three axes, deliberately separated so tumour burden and immune pressure are not
confounded the way a single "scale everything" axis would confound them:

  burden : initial tumour count, immune fixed at the paper's 32/32
  immune : initial macrophage/T-cell counts, tumour fixed at the paper's 128
  geometry: which initial-condition family places the cells

plus an ablation arm that removes an immune population outright, which is the
cheapest way to attribute the outcome to a mechanism rather than to a number.
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
FAMILIES = ["network_field", "rectangle", "circular", "random"]
BURDENS = [32, 64, 128, 192]
IMMUNE = [(8, 8), (16, 16), (64, 64)]  # (32,32) is already the burden arm
ABLATIONS = {
    "no_tcell": dict(tumor=128, macrophage=32, t_cell=0),
    "no_macro": dict(tumor=128, macrophage=0, t_cell=32),
    "none": dict(tumor=128, macrophage=0, t_cell=0),
}


def build_tasks(seeds, max_steps, run_dir):
    tasks = []

    def add(arm, family, counts, cid):
        tasks.append(dict(config_id=cid, arm=arm, ic_family=family,
                          counts=counts, seeds=list(seeds),
                          max_steps=max_steps, run_dir=run_dir))

    # burden x geometry
    for fam, tum in itertools.product(FAMILIES, BURDENS):
        add("burden", fam, dict(tumor=tum, macrophage=32, t_cell=32),
            f"burden__{fam}__t{tum}")
    # immune pressure, on the two families the thesis actually trains/tests on
    for fam, (mac, tc) in itertools.product(["network_field", "rectangle"], IMMUNE):
        add("immune", fam, dict(tumor=128, macrophage=mac, t_cell=tc),
            f"immune__{fam}__m{mac}_c{tc}")
    # mechanism ablations, training geometry only
    for name, counts in ABLATIONS.items():
        add("ablation", "network_field", counts, f"ablation__{name}")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=480)
    args = ap.parse_args()

    run_dir = os.path.join(HERE, "output", args.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    seeds = list(range(args.seeds))
    tasks = build_tasks(seeds, args.max_steps, run_dir)
    n_ep = sum(len(t["seeds"]) for t in tasks)

    so, pgym = bootstrap.assert_all()
    C.dump_json(os.path.join(run_dir, "manifest.json"), {
        "run_tag": args.run_tag, "n_tasks": len(tasks), "n_episodes": n_ep,
        "seeds": seeds, "max_steps": args.max_steps,
        "dt_gym": C.DT_GYM, "max_time_min": args.max_steps * C.DT_GYM,
        "env_kwargs": {k: v for k, v in C.BASE_ENV_KWARGS.items()},
        "paper_counts": PAPER, "families": FAMILIES, "burdens": BURDENS,
        "immune": IMMUNE, "ablations": ABLATIONS,
        "composites": C.COMPOSITES,
        "so": so, "so_mtime": os.path.getmtime(so), "physigym": pgym,
        "python": PYTHON,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=bootstrap.PROJECT_ROOT,
            capture_output=True, text=True).stdout.strip(),
    })
    print(f"{len(tasks)} tasks / {n_ep} episodes -> {run_dir}")

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
                [PYTHON, os.path.join(HERE, "run_one.py"), json.dumps(task)],
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
            print(f"[{done}/{len(tasks)}] {task['config_id']} "
                  f"rc={p.returncode}  {el / 60:.1f} min elapsed", flush=True)

    if failed:
        print(f"\nFAILED tasks: {failed}")

    # concatenate
    import pandas as pd

    eps = sorted(
        os.path.join(run_dir, "episodes", d, "steps.csv.gz")
        for d in os.listdir(os.path.join(run_dir, "episodes"))
        if os.path.exists(os.path.join(run_dir, "episodes", d, "steps.csv.gz"))
    )
    df = pd.concat([pd.read_csv(f) for f in eps], ignore_index=True)
    df.to_csv(os.path.join(run_dir, "all_steps.csv.gz"), index=False,
              compression="gzip")
    summ = df[df["step"] == 0].copy()
    last = df.sort_values("step").groupby(["config_id", "seed"]).tail(1)
    summ = summ.merge(
        last[["config_id", "seed", "n_tumor", "m2_fraction", "step"]],
        on=["config_id", "seed"], suffixes=("_t0", "_end"))
    summ.to_csv(os.path.join(run_dir, "episodes_summary.csv"), index=False)
    print(f"\n{len(df)} rows / {len(eps)} episodes -> all_steps.csv.gz")
    print(f"total {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
