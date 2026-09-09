"""Tier-0 smoke test: measure per-step cost and prove the drug is really zero.

Everything downstream depends on the per-step wall time, which is currently
unknown. Run this before any sweep.

    python exploration/smoke.py [n_steps]
"""

import sys
import time

import bootstrap
import numpy as np

SO, PGYM = bootstrap.assert_all()

import common as C  # noqa: E402
import gymnasium as gym  # noqa: E402
import physigym  # noqa: F401,E402
from extending import physicell  # noqa: E402
from init_conds import generate_initial_condition  # noqa: E402

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
RUN = "exploration/output/_smoke"
XML = f"{RUN}/settings.xml"

print(f"so   : {SO}")
print(f"pgym : {PGYM}")

tree, root = C.prepare_xml(XML, f"{RUN}/pc_scratch", max_time=7200.0)

t0 = time.perf_counter()
env = gym.make(**{**C.BASE_ENV_KWARGS, "settingxml": XML})
u = env.unwrapped
print(f"construct: {time.perf_counter() - t0:.2f}s   domain "
      f"x[{u.x_min},{u.x_max}] y[{u.y_min},{u.y_max}]")

# Initial condition on the same bounds the wrapper would impose (0.9 * domain).
params = C.make_params(**C.PAPER_COUNTS)
ic_csv = f"{RUN}/ic.csv"
df_ic, mode = generate_initial_condition(
    csv_path=ic_csv, mode="network_field",
    x_min=u.x_min * 0.9, x_max=u.x_max * 0.9,
    y_min=u.y_min * 0.9, y_max=u.y_max * 0.9,
    params=params, seed=0,
)
realised = df_ic["type"].value_counts().to_dict()
print(f"IC {mode}: requested {C.PAPER_COUNTS} -> realised {realised}")

C.point_xml_at_ic(tree, root, XML, ic_csv, seed=0)

t0 = time.perf_counter()
obs, info = env.reset(seed=-1)
u.time_simulation = -1  # cross-episode truncation leak; harmless on a fresh env
print(f"reset: {time.perf_counter() - t0:.2f}s   alive at t=0: "
      f"{C.population_counts(u.df_alive)}")

# ── encoder inventory ───────────────────────────────────────────────────────
print("\nencoder dims and cost (median over steps):")
enc_times = {name: [] for name, _ in C.SCALAR_ENCODERS + C.IMAGE_ENCODERS}
enc_dims = {}

step_times = []
dose_violations = []

for t in range(1, N_STEPS + 1):
    ts = time.perf_counter()
    obs, r, term, trunc, info = env.step(C.NOOP_ACTION)
    step_times.append(time.perf_counter() - ts)

    for name, meth in C.SCALAR_ENCODERS + C.IMAGE_ENCODERS:
        te = time.perf_counter()
        v = np.asarray(getattr(u, meth)())
        enc_times[name].append(time.perf_counter() - te)
        enc_dims[name] = v.shape

    # three independent probes that no drug entered the system
    amt = float(physicell.get_parameter("drug_1_amount_used"))
    dose = float(physicell.get_parameter("drug_1_dose"))
    dmax = C.substrate_max("drug_1")
    if not (amt == 0.0 and dose == 0.0 and dmax == 0.0):
        dose_violations.append((t, amt, dose, dmax))

    if term or trunc:
        print(f"  episode ended at step {t}: "
              f"{C.classify_end(term, trunc, u.c_t)} (c_t={u.c_t})")
        break

for name, _ in C.SCALAR_ENCODERS + C.IMAGE_ENCODERS:
    med = np.median(enc_times[name]) * 1e3
    print(f"  {name:14s} {str(enc_dims[name]):16s} {med:7.2f} ms")

enc_total = sum(np.median(v) for v in enc_times.values())
sim_med = float(np.median(step_times[2:] or step_times))

print(f"\nsim step   : median {sim_med * 1e3:.1f} ms  "
      f"(p90 {np.percentile(step_times, 90) * 1e3:.1f} ms)")
print(f"encoders   : {enc_total * 1e3:.1f} ms total")
print(f"per step   : {(sim_med + enc_total) * 1e3:.1f} ms")

print(f"\ndose-zero probes: "
      f"{'PASS' if not dose_violations else f'FAIL {dose_violations[:3]}'}")

per_step = sim_med + enc_total
for horizon, label in ((480, "7200 min (full)"), (160, "2400 min"), (80, "1200 min")):
    ep = per_step * horizon
    print(f"  horizon {horizon:3d} steps ({label:15s}): {ep:6.1f}s/episode  "
          f"-> 128 episodes on 12 workers = {128 * ep / 12 / 60:5.1f} min")

env.close()
