"""Morris screening of the untreated model, repeated across initial conditions.

    python exploration/sa_driver.py --run-tag sa_v3 \
        --trajectories 12 --workers 12 --horizon 200 \
        --ic-families network_field,rectangle --ic-seeds 0,1,2

Morris elementary effects, not Sobol: the thesis scopes this as a screening of
the parameters that were chosen rather than measured, explicitly not a full
global sensitivity analysis. Morris answers "which of these matter at all",
which is the question asked, at a fraction of the cost.

The SAME sample matrix is evaluated under every initial condition, so a
difference in the resulting indices is attributable to the initial condition
rather than to resampling. That turns "is the parameter ranking stable across
geometries and seeds?" into a question the run can answer.

Sampling and index computation are SALib's, the same library UQ-PhysiCell wraps,
so the method is the community-standard one even though the execution layer is
ours (UQ-PhysiCell drives a standalone binary and parses output from the
configured save folder; this model's custom.cpp writes to output/episodeNNNNNNNN
instead, and our pipeline runs the compiled module in process).
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import bootstrap

import common as C  # noqa: E402
import sa_params as P  # noqa: E402

PYTHON = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="sa_" + time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--trajectories", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--ic-families", default="network_field,rectangle")
    ap.add_argument("--ic-seeds", default="0,1,2")
    args = ap.parse_args()

    from SALib.sample import morris as morris_sample

    families = [f.strip() for f in args.ic_families.split(",") if f.strip()]
    seeds = [int(s) for s in args.ic_seeds.split(",") if s.strip()]
    ic_configs = [(f, s) for f in families for s in seeds]

    run_dir = os.path.join(HERE, "output", args.run_tag)
    os.makedirs(run_dir, exist_ok=True)

    problem = P.salib_problem()
    X = morris_sample.sample(problem, N=args.trajectories, num_levels=4)
    n_total = len(X) * len(ic_configs)
    print(f"{len(X)} samples x {len(ic_configs)} initial conditions "
          f"= {n_total} runs ({problem['num_vars']} parameters)")
    for f, s in ic_configs:
        print(f"    {f} seed {s}")

    so, pgym = bootstrap.assert_all()
    C.dump_json(os.path.join(run_dir, "manifest.json"), {
        "run_tag": args.run_tag, "method": "morris", "num_levels": 4,
        "trajectories": args.trajectories, "n_samples_per_ic": len(X),
        "n_runs_total": n_total,
        "ic_families": families, "ic_seeds": seeds,
        "problem": problem, "defaults": P.defaults(),
        "horizon_steps": args.horizon,
        "horizon_min": args.horizon * C.DT_GYM,
        "counts": C.PAPER_COUNTS, "so": so, "physigym": pgym,
        "note": "the same X matrix is evaluated under every initial condition",
    })

    logs = os.path.join(run_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    # one PROCESS per run: the physigym singleton is never reset, and cell rules
    # bind at cell-type creation so a reused process keeps stale rules
    queue = []
    for fam, seed in ic_configs:
        tag = f"{fam}__s{seed}"
        for i in range(len(X)):
            queue.append(dict(
                indices=[i], X=[X[i].tolist()], run_dir=run_dir,
                horizon=args.horizon, ic_family=fam, ic_seed=seed,
                ic_tag=tag, counts=C.PAPER_COUNTS,
                worker_id=len(queue) % args.workers))

    running, done, t0 = [], 0, time.perf_counter()
    while queue or running:
        while queue and len(running) < args.workers:
            task = queue.pop(0)
            fh = open(os.path.join(
                logs, f"{task['ic_tag']}_s{task['indices'][0]:05d}.log"), "w")
            p = subprocess.Popen(
                [PYTHON, os.path.join(HERE, "sa_run.py"), json.dumps(task)],
                cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            running.append((p, task, fh))
        time.sleep(0.5)
        for e in list(running):
            p, task, fh = e
            if p.poll() is None:
                continue
            running.remove(e)
            fh.close()
            done += 1
            if done % 25 == 0 or done == n_total:
                el = (time.perf_counter() - t0) / 60
                rate = done / max(el, 1e-9)
                print(f"[{done}/{n_total}] {el:.1f} min elapsed, "
                      f"~{(n_total - done) / max(rate, 1e-9):.1f} min left",
                      flush=True)

    analyse(run_dir, problem)


def analyse(run_dir, problem=None):
    import numpy as np
    import pandas as pd
    from SALib.analyze import morris as morris_analyze

    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    if problem is None:
        problem = man["problem"]
    names = problem["names"]

    frames = []
    for d in sorted(os.listdir(os.path.join(run_dir, "results"))):
        parts = sorted(glob.glob(os.path.join(run_dir, "results", d, "sample_*.csv")))
        if parts:
            frames.append(pd.concat([pd.read_csv(f) for f in parts],
                                    ignore_index=True))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(run_dir, "samples.csv"), index=False)

    print(f"\n{len(df)} runs across {df['ic_tag'].nunique()} initial conditions")
    if "dose_ok" in df:
        print(f"dose-zero held everywhere: "
              f"{bool(df['dose_ok'].fillna(False).all())}")

    QOIS = ("log_growth", "steps_to_escape", "c_T")
    rank_tables = {}

    for qname in QOIS:
        if qname not in df:
            continue
        per_ic = {}
        for tag, g in df.groupby("ic_tag"):
            g = g.sort_values("sample")
            X = g[[f"p__{n}" for n in names]].to_numpy(dtype=float)
            Y = g[qname].to_numpy(dtype=float)
            if not np.isfinite(Y).all() or np.std(Y) == 0:
                continue
            res = morris_analyze.analyze(problem, X, Y, num_levels=4)
            per_ic[tag] = dict(zip(res["names"], res["mu_star"]))
        if not per_ic:
            print(f"\n=== {qname}: constant everywhere, skipped ===")
            continue

        tab = pd.DataFrame(per_ic)
        tab["mean"] = tab.mean(axis=1)
        tab["cv"] = tab[list(per_ic)].std(axis=1) / tab["mean"].replace(0, np.nan)
        tab = tab.sort_values("mean", ascending=False)
        rank_tables[qname] = tab
        tab.to_csv(os.path.join(run_dir, f"morris_{qname}.csv"))

        print(f"\n=== {qname}: mu* per initial condition "
              f"(range {df[qname].min():.4g} .. {df[qname].max():.4g}) ===")
        print(tab.to_string(float_format=lambda v: f"{v:9.4g}"))

        # rank stability: Spearman between each IC's ordering and the mean
        order = tab.index.tolist()
        print("  rank correlation with the pooled ordering:")
        for tag in per_ic:
            s = pd.Series(per_ic[tag]).reindex(order)
            rho = s.rank().corr(pd.Series(range(len(order), 0, -1),
                                          index=order).rank(), method="spearman")
            print(f"    {tag:28s} rho = {rho: .3f}")

    return rank_tables


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--analyse-only":
        analyse(sys.argv[2])
    else:
        main()
