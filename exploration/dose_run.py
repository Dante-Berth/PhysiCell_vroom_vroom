"""One worker process = one (arm, dose, initial-condition family), several seeds.

    python exploration/dose_run.py '<task json>'

Companion to run_one.py, which runs the same model with the drug off. Everything
structural is shared with it; the differences are all about the drug:

* The disc is set through `physicell.set_parameter`, NOT through the settings
  XML. `common.prepare_xml` pins drug_1_x/y/radius to 0 -- that pinning is
  load-bearing for the untreated study (add_local_substrate returns early when
  radius <= 0, which is one of its three dose-zero guards), so it is left alone
  and overridden here after every reset. physicell_start re-reads the XML on
  reset, so the override has to happen after reset, not before.

* Only the keys present in the action dict are written to PhysiCell
  (physicell_core.step). In action_mode="full" that is drug_1_dose alone, which
  is why x/y/radius have to be set out of band.

* The delivery guard is the mirror of run_one.py's dose-zero guard. For a
  commanded dose > 0 the drug MUST reach the field. A radius left at 0 would
  silently deliver nothing at every dose and produce a flat dose-response that
  looks like a drug-insensitive model.
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

# wrapper.py:652-660. "uniform" reproduces action_mode="full" exactly: the disc
# is centred on the domain with radius = the half-diagonal, so it contains every
# voxel of the rectangle. "targeted" uses the largest disc the targeted action
# space permits, radius_max_norm = 0.20 of the same half-diagonal.
RADIUS_MAX_NORM = 0.20


def m2_mask(u):
    """Which macrophages are M2, by the rule get_macrophage_polarization_scalars uses.

    Nearest voxel by cKDTree over the xy columns of the two factor fields;
    pro > anti means M2. Identical to common.m1m2_split, which returns only
    counts, so the per-cell mask is recomputed here.
    """
    from scipy.spatial import cKDTree

    mac = u.df_alive[u.df_alive["type"] == "macrophage"]
    if len(mac) == 0:
        return mac, np.zeros(0, dtype=bool)
    pro = np.asarray(physicell.get_microenv("pro_tumoral_factor"))
    anti = np.asarray(physicell.get_microenv("anti_tumoral_factor"))
    _, nearest = cKDTree(pro[:, :2]).query(mac[["x", "y"]].to_numpy())
    return mac, pro[:, -1][nearest] > anti[:, -1][nearest]


def best_centre(candidates, targets, radius):
    """The candidate centre covering the most targets."""
    if len(candidates) == 0 or len(targets) == 0:
        return None
    d2 = ((candidates[:, None, :] - targets[None, :, :]) ** 2).sum(-1)
    covered = (d2 <= radius**2).sum(1)
    best = int(covered.argmax())
    return float(candidates[best, 0]), float(candidates[best, 1])


def max_radius(u):
    return float(np.sqrt((u.width / 2) ** 2 + (u.height / 2) ** 2))


def domain_centre(u):
    return float(u.x_min + u.width / 2), float(u.y_min + u.height / 2)


_CAND = {}


def candidates(u, radius):
    """Injection centres whose disc lies entirely inside the domain.

    This is the candidate set of `figures/make_aim_comparison.py`, which
    produces fig:tme:aim, and the restriction is the reason that figure can
    claim to be dose-matched: a centre near an edge puts part of the disc off
    the mesh and administers less drug, so an unrestricted search confounds
    placement with dose. Measured on the first ladder run, cell-position
    candidates gave a covered fraction ranging over 0.0567 to 0.0632 instead of
    a constant 0.0630.
    """
    key = (u.x_min, u.x_max, u.y_min, u.y_max, round(radius, 6))
    if key not in _CAND:
        gx, gy = np.meshgrid(np.arange(u.x_min + radius, u.x_max - radius, 2.0),
                             np.arange(u.y_min + radius, u.y_max - radius, 2.0))
        _CAND[key] = np.column_stack((gx.ravel(), gy.ravel()))
    return _CAND[key]


def covered(cand, pts, radius):
    if len(pts) == 0:
        return np.zeros(len(cand))
    return (np.linalg.norm(cand[:, None] - pts[None], axis=2) <= radius).sum(1)


def aim_point(u, arm, radius_hint=(0.0,)):
    """Where the injection goes this step.

    "uniform" ignores the state, as the RL full action space does. Every other
    arm searches the same dose-matched candidate grid and differs only in what
    it counts as a target.
    """
    if arm == "uniform":
        return domain_centre(u)
    radius = radius_hint[0]
    df = u.df_alive

    if arm == "targeted":
        # centre of mass of the macrophages: the cheap aim, kept as the
        # baseline it turned out to be. It is the only arm that does not search.
        mac = df[df["type"] == "macrophage"][["x", "y"]].to_numpy()
        if len(mac):
            return float(mac[:, 0].mean()), float(mac[:, 1].mean())
        return domain_centre(u)

    cand = candidates(u, radius)
    if len(cand) == 0:
        return domain_centre(u)

    mac, is_m2 = m2_mask(u)
    mxy = mac[["x", "y"]].to_numpy() if len(mac) else np.zeros((0, 2))

    if arm == "grid_macro":
        # exactly fig:tme:aim's "aimed" arm: cover the most macrophages,
        # whatever their polarisation.
        score = covered(cand, mxy, radius)
    else:
        # The drug repolarises what it covers (sec:tme:drug), so dose landing on
        # a macrophage that is already M1 buys nothing. These two arms spend it
        # on the ones still M2.
        m2 = mxy[is_m2] if len(mxy) else mxy
        if len(m2) == 0:
            score = covered(cand, mxy, radius)  # none left: fall back
        elif arm == "grid_m2":
            score = covered(cand, m2, radius)
        else:
            # "grid_m2_tumor": between the tumour and the M2, but on the M2.
            # Lexicographic, M2 dominant and tumour cells covered as the
            # tie-break, so the disc is pulled toward the M2 macrophages that
            # sit against the tumour. Repolarising those puts the beacon where
            # the T cells are needed rather than at the population's edge.
            tum = df[df["type"] == "tumor"][["x", "y"]].to_numpy()
            score = 1000.0 * covered(cand, m2, radius) + covered(cand, tum, radius)

    cx, cy = cand[int(np.argmax(score))]
    return float(cx), float(cy)


def apply_disc(u, arm, radius):
    cx, cy = aim_point(u, arm, (radius,))
    physicell.set_parameter("drug_1_x", cx)
    physicell.set_parameter("drug_1_y", cy)
    physicell.set_parameter("drug_1_radius", radius)
    return cx, cy


def record(task, seed, step, dose_cmd, cx, cy, radius, env, reward, realised):
    u = env.unwrapped
    row = {
        "config_id": task["config_id"],
        "arm": task["arm"],
        "dose_mode": task["dose_mode"],
        "dose_cmd": float(dose_cmd),
        "ic_family": task["ic_family"],
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

    # ── dose accounting ──────────────────────────────────────────────────
    # dose_spent is what wrapper.py:723 charges the agent: the mass PhysiCell
    # actually consumed, divided by the whole domain volume. Because
    # add_local_substrate accumulates r_dose * voxel_volume over the voxels
    # inside the disc, this equals commanded dose x fraction of domain covered.
    # Recording it per step is what lets the composite reward be recomputed
    # offline for any (w_cell, w_dose) without re-running anything.
    amount_used = float(physicell.get_parameter("drug_1_amount_used"))
    row["drug_1_amount_used"] = amount_used
    row["drug_1_dose_param"] = float(physicell.get_parameter("drug_1_dose"))
    row["dose_spent"] = float(u.get_dose_spent())
    # what the disc actually covers this step. fig:tme:m1m2 reports 16 of 31
    # macrophages inside the disc, so this is the number to compare against
    # before reading anything into the downstream repolarisation.
    mac, is_m2 = m2_mask(u)
    if len(mac):
        dist = np.linalg.norm(mac[["x", "y"]].to_numpy() - np.array([cx, cy]), axis=1)
        inside = dist <= radius
        row["n_macro_covered"] = int(inside.sum())
        row["n_m2_covered"] = int((inside & is_m2).sum())
    else:
        row["n_macro_covered"] = 0
        row["n_m2_covered"] = 0
    row["drug_1_x"] = cx
    row["drug_1_y"] = cy
    row["drug_1_radius"] = radius
    row["coverage_frac"] = (
        amount_used / (dose_cmd * u.total_volume) if dose_cmd > 0 else float("nan")
    )
    # r_tumor as physicell_model.get_reward returns it; the composite reward the
    # sweeps optimised is w_cell * this - w_dose * dose_spent - w_smooth * (...).
    row["r_tumor"] = float(reward) if reward is not None else float("nan")

    # delivery guard: the mirror of run_one.py's dose-zero guard
    if dose_cmd > 0:
        row["delivery_ok"] = bool(amount_used > 0.0 and row["submax__drug_1"] > 0.0)
    else:
        row["delivery_ok"] = bool(amount_used == 0.0 and row["submax__drug_1"] == 0.0)
    return row


def run_episode(env, tree, root, xml, task, seed, ep_dir, rng):
    u = env.unwrapped
    os.makedirs(ep_dir, exist_ok=True)
    ic_csv = os.path.join(ep_dir, "ic.csv")

    df_ic, _ = generate_initial_condition(
        csv_path=ic_csv,
        mode=task["ic_family"],
        # the wrapper shrinks the domain to 90%; match it, as run_one.py does
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
    # physicell_core sets time_simulation in __init__ only and never resets it;
    # without this a short episode makes the NEXT one truncate at step 1.
    u.time_simulation = -1

    radius = (max_radius(u) if task["arm"] == "uniform"
              else RADIUS_MAX_NORM * max_radius(u))

    def next_dose():
        if task["dose_mode"] == "random":
            return float(rng.uniform(0.0, 1.0))
        return float(task["dose"])

    cx, cy = domain_centre(u)
    rows = [record(task, seed, 0, 0.0, cx, cy, radius, env, None, realised)]

    warmup = int(task.get("warmup", 0))
    end_reason, last = "truncation", 0
    t_start = time.perf_counter()
    for t in range(1, task["max_steps"] + 1):
        if t <= warmup:
            # untreated run-in, as make_aim_comparison.py does before aiming:
            # polarisation and position have to establish before "aim at the
            # macrophages" means anything, and before pro/anti are non-zero the
            # M1/M2 test cannot separate them at all.
            _, reward, term, trunc, _ = env.step(
                {"drug_1_dose": np.array([0.0], dtype=np.float32)})
            rows.append(record(task, seed, t, 0.0, cx, cy, radius, env, reward,
                               realised))
            last = t
            if term or trunc:
                end_reason = C.classify_end(term, trunc, u.c_t)
                break
            continue
        # the disc must be in place BEFORE the step: physicell_core.step writes
        # only drug_1_dose (the sole key of the full action space) and then runs,
        # so C++ reads whatever x/y/radius we last set.
        d = next_dose()
        cx, cy = apply_disc(u, task["arm"], radius)
        action = {"drug_1_dose": np.array([d], dtype=np.float32)}
        _, reward, term, trunc, _ = env.step(action)
        rows.append(record(task, seed, t, d, cx, cy, radius, env, reward, realised))
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
    df.to_csv(os.path.join(ep_dir, "steps.csv.gz"), index=False, compression="gzip")

    stepped = df[df["step"] > 0]
    C.dump_json(
        os.path.join(ep_dir, "meta.json"),
        {
            "config_id": task["config_id"],
            "arm": task["arm"],
            "dose_mode": task["dose_mode"],
            "dose": task["dose"],
            "ic_family": task["ic_family"],
            "counts": task["counts"],
            "seed": seed,
            "realised": realised,
            "end_reason": end_reason,
            "end_step": last,
            "wall_s": wall,
            "radius": radius,
            "no_terminate": bool(task.get("no_terminate")),
            "drug_diffusion": task.get("drug_diffusion"),
            "delivery_all_ok": bool(df["delivery_ok"].all()),
            "coverage_frac_median": float(stepped["coverage_frac"].median()),
            "dose_spent_sum": float(stepped["dose_spent"].sum()),
            "r_tumor_sum": float(stepped["r_tumor"].sum()),
            "c_T": int(df["c_t"].iloc[-1]),
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
    if task.get("drug_diffusion") is not None:
        # The whole point of the sweep: as the drug's diffusion length
        # sqrt(D/decay) grows past the injection radius, the field stops being
        # the disc that was commanded and placement stops mattering. At D = 0
        # (the shipped value) the field IS the disc.
        from lxml import etree as _et
        _t = _et.parse(xml)
        _r = _t.getroot()
        _r.xpath("//microenvironment_setup/variable[@name='drug_1']"
                 "/physical_parameter_set/diffusion_coefficient")[0].text = repr(
                     float(task["drug_diffusion"]))
        _t.write(xml, pretty_print=True)
        tree, root = _t, _r
    env = gym.make(**{**C.BASE_ENV_KWARGS, "settingxml": xml})

    if task.get("no_terminate"):
        # Let every run reach the horizon. physicell_model.get_terminated ends an
        # episode at c_t > 256 or c_t <= 3, which censors exactly the runs a dose
        # response wants to compare: an escaping run stops at ~257 whatever its
        # dose, so burden past that point is unobservable and every summary of it
        # reports the termination rule. Patched on the INSTANCE, so the model file
        # that produced the reported sweeps is untouched. physicell_core calls
        # self.get_terminated() (:706), which an instance attribute shadows.
        env.unwrapped.get_terminated = lambda: False
    rng = np.random.default_rng(abs(hash(task["config_id"])) % (2**32))

    for seed in task["seeds"]:
        ep_dir = os.path.join(run_dir, "episodes", f"{task['config_id']}__s{seed}")
        if os.path.exists(os.path.join(ep_dir, "meta.json")):
            continue  # --resume
        try:
            end_reason, last, wall = run_episode(
                env, tree, root, xml, task, seed, ep_dir, rng
            )
            print(f"{task['config_id']} s{seed}: {end_reason} @ {last} ({wall:.1f}s)",
                  flush=True)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    env.close()


if __name__ == "__main__":
    main()
