"""One worker process = one task = one parameter point, several seeds.

The compiled physicell module is a per-process singleton, so a task gets its own
interpreter. Several episodes per process is safe because physicell_start's
reload path re-reads the settings XML (and therefore //random_seed and the
initial-condition path) on every reset.

    python exploration/run_one.py '<task json>'
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
from extending import physicell  # noqa: E402
from init_conds import generate_initial_condition  # noqa: E402

SUBSTRATES = [
    "anti_tumoral_factor",
    "pro_tumoral_factor",
    "drug_1",
    "tumor_molecule",
    "cytokine",
]


def encode_all(u, prev):
    """Call every feature builder on the current instant.

    Returns (summary columns, full vectors, timings). `prev` holds the previous
    step's vectors so we can report the step-to-step jump, which is the property
    that matters for whether a state space is well conditioned for value
    learning.
    """
    cols, vecs = {}, {}
    for name, meth in C.SCALAR_ENCODERS + C.IMAGE_ENCODERS:
        t0 = time.perf_counter()
        v = np.asarray(getattr(u, meth)())
        dt = time.perf_counter() - t0
        flat = v.astype(np.float32).ravel()
        vecs[name] = v
        n = float(np.linalg.norm(flat))
        cols[f"{name}__dim"] = int(flat.size)
        cols[f"{name}__l2"] = n
        cols[f"{name}__mean"] = float(flat.mean())
        cols[f"{name}__std"] = float(flat.std())
        cols[f"{name}__min"] = float(flat.min())
        cols[f"{name}__max"] = float(flat.max())
        cols[f"{name}__nnz_frac"] = float((flat != 0).mean())
        cols[f"{name}__wall_s"] = dt
        if prev is not None and name in prev:
            d = float(np.linalg.norm(flat - prev[name].astype(np.float32).ravel()))
            cols[f"{name}__l2_delta"] = d
            cols[f"{name}__l2_delta_rel"] = d / n if n > 0 else float("nan")
        else:
            cols[f"{name}__l2_delta"] = float("nan")
            cols[f"{name}__l2_delta_rel"] = float("nan")
    return cols, vecs


def record(env, task, seed, step, prev_vecs, realised):
    u = env.unwrapped
    row = {
        "config_id": task["config_id"],
        "arm": task["arm"],
        "ic_family": task["ic_family"],
        "req_tumor": task["counts"]["tumor"],
        "req_macrophage": task["counts"]["macrophage"],
        "req_t_cell": task["counts"]["t_cell"],
        "seed": seed,
        "step": step,
        "sim_time_min": step * C.DT_GYM,
    }
    row.update({f"ic_realised_{k}": v for k, v in realised.items()})
    row.update(C.population_counts(u.df_alive))
    row.update(C.m1m2_split(env))
    row.update(C.geometry(u.df_alive))
    row["n_alive"] = int(len(u.df_alive))
    row["n_dead"] = int(len(u.df_dead))
    row["c_t"] = int(u.c_t)
    row["c_prev"] = int(u.c_prev) if u.c_prev is not None else int(u.c_t)
    row["c_0"] = int(u.c_0)
    row["tumor_log_ratio"] = (
        float(np.log(u.c_t / u.c_0)) if u.c_t > 0 and u.c_0 > 0 else float("nan")
    )
    for s in SUBSTRATES:
        row[f"submax__{s}"] = C.substrate_max(s)
    # dose-zero guards: what we sent, what PhysiCell consumed, what reached the field
    row["drug_1_amount_used"] = float(physicell.get_parameter("drug_1_amount_used"))
    row["drug_1_dose_param"] = float(physicell.get_parameter("drug_1_dose"))
    row["dose_guard_ok"] = bool(
        row["drug_1_amount_used"] == 0.0
        and row["drug_1_dose_param"] == 0.0
        and row["submax__drug_1"] == 0.0
    )
    cols, vecs = encode_all(u, prev_vecs)
    row.update(cols)
    return row, vecs


def run_episode(env, tree, root, xml, task, seed, ep_dir):
    u = env.unwrapped
    os.makedirs(ep_dir, exist_ok=True)
    ic_csv = os.path.join(ep_dir, "ic.csv")

    df_ic, mode = generate_initial_condition(
        csv_path=ic_csv,
        mode=task["ic_family"],
        # the wrapper shrinks the domain to 90%; match it so the geometry is the
        # same one the training runs saw
        x_min=u.x_min * 0.9,
        x_max=u.x_max * 0.9,
        y_min=u.y_min * 0.9,
        y_max=u.y_max * 0.9,
        params=C.make_params(**task["counts"]),
        seed=seed,
    )
    counts = df_ic["type"].value_counts().to_dict()
    realised = {k: int(counts.get(k, 0)) for k in ("tumor", "macrophage", "t_cell")}

    C.point_xml_at_ic(tree, root, xml, ic_csv, seed)
    env.reset(seed=-1)  # seed<0 => take //random_seed from the XML we just wrote
    # physicell_core sets time_simulation in __init__ only and mutates it in
    # get_truncated, but never resets it. Without this a short episode makes the
    # NEXT one truncate spuriously at step 1.
    u.time_simulation = -1

    rows, buf = [], {}
    r0, prev = record(env, task, seed, 0, None, realised)
    rows.append(r0)
    for name, v in prev.items():
        buf.setdefault(name, []).append(np.asarray(v, dtype=np.float32).ravel())

    end_reason, last = "truncation", 0
    t_start = time.perf_counter()
    for t in range(1, task["max_steps"] + 1):
        _, _, term, trunc, _ = env.step(C.NOOP_ACTION)
        row, prev = record(env, task, seed, t, prev, realised)
        rows.append(row)
        for name, v in prev.items():
            buf.setdefault(name, []).append(np.asarray(v, dtype=np.float32).ravel())
        last = t
        if term or trunc:
            end_reason = C.classify_end(term, trunc, u.c_t)
            break
    wall = time.perf_counter() - t_start

    df = pd.DataFrame(rows)
    df["end_reason"] = end_reason
    df["end_step"] = last
    df["n_steps_total"] = last
    df["escaped"] = end_reason == "escape"
    df["eradicated"] = end_reason == "eradication"
    df["controlled"] = end_reason == "truncation"
    df.to_csv(os.path.join(ep_dir, "steps.csv.gz"), index=False,
              compression="gzip")

    np.savez_compressed(
        os.path.join(ep_dir, "encodings.npz"),
        step=np.arange(last + 1),
        **{k: np.stack(v) for k, v in buf.items()},
    )
    C.dump_json(
        os.path.join(ep_dir, "meta.json"),
        {
            "config_id": task["config_id"], "arm": task["arm"],
            "ic_family": task["ic_family"], "counts": task["counts"],
            "seed": seed, "realised": realised, "end_reason": end_reason,
            "end_step": last, "wall_s": wall,
            "dose_guard_all_ok": bool(df["dose_guard_ok"].all()),
        },
    )
    return end_reason, last, wall


def main():
    task = json.loads(sys.argv[1])
    run_dir = task["run_dir"]
    xml = os.path.join(run_dir, "cfg", f"settings_w{task['worker_id']}.xml")
    scratch = os.path.join(run_dir, "pc_scratch", f"w{task['worker_id']}")
    max_time = task["max_steps"] * C.DT_GYM + C.DT_GYM  # one spare step

    tree, root = C.prepare_xml(xml, scratch, max_time=max_time)
    env = gym.make(**{**C.BASE_ENV_KWARGS, "settingxml": xml})

    for seed in task["seeds"]:
        ep_dir = os.path.join(run_dir, "episodes", f"{task['config_id']}__s{seed}")
        if os.path.exists(os.path.join(ep_dir, "meta.json")):
            continue  # --resume
        try:
            reason, last, wall = run_episode(
                env, tree, root, xml, task, seed, ep_dir
            )
            print(f"{task['config_id']} s{seed}: {reason} at step {last} "
                  f"({wall:.1f}s)", flush=True)
        except Exception:
            traceback.print_exc()
            C.dump_json(os.path.join(ep_dir, "meta.json"),
                        {"config_id": task["config_id"], "seed": seed,
                         "end_reason": "error"})
    env.close()


if __name__ == "__main__":
    main()
