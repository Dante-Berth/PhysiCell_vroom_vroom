"""One worker = one batch of Morris samples, run untreated.

Each sample writes its own settings XML and cell_rules.csv, then runs a fixed
horizon with dose 0 and reports the quantities of interest.

    python exploration/sa_run.py '<task json>'
"""

import json
import os
import sys
import time
import traceback

import bootstrap

bootstrap.assert_all()

import common as C  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import physigym  # noqa: F401,E402
import sa_params as P  # noqa: E402
from extending import physicell  # noqa: E402
from init_conds import generate_initial_condition  # noqa: E402
from lxml import etree  # noqa: E402

RULES_SRC = os.path.join(bootstrap.PROJECT_ROOT, "config/cell_rules.csv")


def prepare_sample(work, values, save_folder, max_time):
    """Write a settings XML + rules CSV for one sampled parameter vector."""
    os.makedirs(work, exist_ok=True)
    xml = os.path.join(work, "settings.xml")
    rules = os.path.join(work, "cell_rules.csv")

    P.write_rules(RULES_SRC, rules, values)
    tree, root = C.prepare_xml(xml, save_folder, max_time=max_time)
    P.apply_xml(root, values)
    # point the ruleset at our modified copy
    root.xpath("//cell_rules/rulesets/ruleset/folder")[0].text = work
    root.xpath("//cell_rules/rulesets/ruleset/filename")[0].text = "cell_rules.csv"
    tree.write(xml, pretty_print=True)
    return xml, tree, root


def qoi(rows, horizon):
    """Quantities of interest for the untreated regime."""
    df = pd.DataFrame(rows)
    c0 = float(df["n_tumor"].iloc[0])
    cT = float(df["n_tumor"].iloc[-1])
    reached = df[df["n_tumor"] > 256]
    return {
        "c_0": c0,
        "c_T": cT,
        "log_growth": float(np.log(max(cT, 1) / max(c0, 1))),
        "net_growth_per_step": float(
            np.log(max(cT, 1) / max(c0, 1)) / max(len(df) - 1, 1)
        ),
        "steps_to_escape": int(reached["step"].iloc[0]) if len(reached) else horizon,
        "escaped": bool(len(reached) > 0),
        "tumor_killed": float(df["n_dead"].iloc[-1]),
        "m2_fraction_mean": float(df["m2_fraction"].mean()),
        "n_t_cell_end": float(df["n_t_cell"].iloc[-1]),
        "n_steps": int(len(df) - 1),
    }


def run_sample(task, idx, values):
    work = os.path.join(task["run_dir"], "samples", task["ic_tag"], f"s{idx:05d}")
    scratch = os.path.join(task["run_dir"], "pc_scratch", f"w{task['worker_id']}")
    horizon = task["horizon"]
    max_time = horizon * C.DT_GYM + C.DT_GYM

    xml, tree, root = prepare_sample(work, values, scratch, max_time)

    # A fresh env per sample, and therefore a fresh PROCESS per sample. Two
    # independent reasons force this:
    #   1. physicell.flag_envphysigym is set at the end of __init__ and never
    #      reset, not even by close(), so a second gym.make in one process
    #      raises RuntimeWarning.
    #   2. Cell rules bind to cell definitions at cell-type creation, which the
    #      reload path does not redo, so a reused process would silently keep
    #      the FIRST sample's rules.
    env = gym.make(**{**C.BASE_ENV_KWARGS, "settingxml": xml})
    u = env.unwrapped
    try:
        ic = os.path.join(work, "ic.csv")
        generate_initial_condition(
            csv_path=ic, mode=task["ic_family"],
            x_min=u.x_min * 0.9, x_max=u.x_max * 0.9,
            y_min=u.y_min * 0.9, y_max=u.y_max * 0.9,
            params=C.make_params(**task["counts"]), seed=task["ic_seed"],
        )
        C.point_xml_at_ic(tree, root, xml, ic, seed=task["ic_seed"])
        env.reset(seed=-1)
        u.time_simulation = -1

        rows = []

        def snap(step):
            r = {"step": step}
            r.update(C.population_counts(u.df_alive))
            r.update(C.m1m2_split(env))
            r["n_dead"] = int(len(u.df_dead))
            return r

        rows.append(snap(0))
        for t in range(1, horizon + 1):
            _, _, term, trunc, _ = env.step(C.NOOP_ACTION)
            rows.append(snap(t))
            if term or trunc:
                break

        out = qoi(rows, horizon)
        out["dose_ok"] = bool(
            physicell.get_parameter("drug_1_amount_used") == 0.0
            and C.substrate_max("drug_1") == 0.0
        )
    finally:
        env.close()

    out["sample"] = idx
    out["ic_tag"] = task["ic_tag"]
    out["ic_family"] = task["ic_family"]
    out["ic_seed"] = task["ic_seed"]
    out.update({f"p__{k}": v for k, v in values.items()})
    return out


def main():
    task = json.loads(sys.argv[1])
    X = np.array(task["X"])
    results = []
    for j, idx in enumerate(task["indices"]):
        values = dict(zip(P.NAMES, X[j]))
        t0 = time.perf_counter()
        try:
            r = run_sample(task, idx, values)
            r["wall_s"] = time.perf_counter() - t0
            results.append(r)
            print(f"sample {idx}: log_growth={r['log_growth']:.3f} "
                  f"escape@{r['steps_to_escape']} ({r['wall_s']:.1f}s)", flush=True)
        except Exception:
            traceback.print_exc()
            results.append({"sample": idx, "failed": True,
                            **{f"p__{k}": v for k, v in values.items()}})
    out = os.path.join(task["run_dir"], "results", task["ic_tag"],
                       f"sample_{task['indices'][0]:05d}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)


if __name__ == "__main__":
    main()
