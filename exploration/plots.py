"""Figures for the untreated parameter analysis.

    python exploration/plots.py output/dose0_v1
"""

import os
import sys

import bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "output/dose0_v1"
FIG = os.path.join(RUN, "figures")
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(RUN, "all_steps.csv.gz"))
df["hours"] = df["sim_time_min"] / 60.0
ep = df[df["step"] == 0].copy()

FAM_C = {"network_field": "#1f77b4", "rectangle": "#d62728",
         "circular": "#2ca02c", "random": "#9467bd"}
OUT_C = {"escape": "#d62728", "eradication": "#2ca02c",
         "truncation": "#7f7f7f"}


def band(ax, g, ycol, color, label=None):
    """Median and interquartile band across seeds."""
    piv = g.pivot_table(index="step", values=ycol, aggfunc=["median", q25, q75])
    x = piv.index / 4.0  # steps -> hours (15 min per step)
    ax.plot(x, piv[("median", ycol)], color=color, lw=1.6, label=label)
    ax.fill_between(x, piv[("q25", ycol)], piv[("q75", ycol)],
                    color=color, alpha=0.18, lw=0)


def q25(s):
    return s.quantile(0.25)


def q75(s):
    return s.quantile(0.75)


# ── 1. tumour trajectories: burden x geometry ───────────────────────────────
b = df[df["arm"] == "burden"]
burdens = sorted(b["req_tumor"].unique())
fams = [f for f in FAM_C if f in set(b["ic_family"])]
fig, axes = plt.subplots(len(burdens), len(fams), figsize=(4 * len(fams), 2.8 * len(burdens)),
                         sharex=True, sharey=True, squeeze=False)
for i, tum in enumerate(burdens):
    for j, fam in enumerate(fams):
        ax = axes[i][j]
        g = b[(b["req_tumor"] == tum) & (b["ic_family"] == fam)]
        if len(g):
            band(ax, g, "n_tumor", FAM_C[fam])
        ax.axhline(256, color="k", ls="--", lw=0.8)
        ax.axhline(3, color="k", ls=":", lw=0.8)
        if i == 0:
            ax.set_title(fam, fontsize=10)
        if j == 0:
            ax.set_ylabel(f"tumour = {tum}\n\ncells", fontsize=9)
        if i == len(burdens) - 1:
            ax.set_xlabel("hours")
fig.suptitle("Untreated tumour trajectories: initial burden x initial geometry\n"
             "median and IQR over seeds; dashed = escape threshold (256)",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/01_tumour_burden_geometry.png", dpi=150)
plt.close(fig)

# ── 2. all populations, paper parameter point, per family ───────────────────
p = b[b["req_tumor"] == 128]
fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.2),
                         sharey=True, squeeze=False)
for j, fam in enumerate(fams):
    ax = axes[0][j]
    g = p[p["ic_family"] == fam]
    for col, c, lab in (("n_tumor", "#444444", "tumour"),
                        ("n_M1", "#1f77b4", "M1"),
                        ("n_M2", "#ff7f0e", "M2"),
                        ("n_t_cell", "#2ca02c", "T cell")):
        if len(g):
            band(ax, g, col, c, lab)
    ax.set_title(fam, fontsize=10)
    ax.set_xlabel("hours")
    if j == 0:
        ax.set_ylabel("cells")
        ax.legend(fontsize=8, frameon=False)
fig.suptitle("Untreated populations at the paper parameter point (128/32/32)",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/02_populations_paper_point.png", dpi=150)
plt.close(fig)

# ── 3. growth relative to the unopposed reference ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.2))
for fam in fams:
    g = b[(b["ic_family"] == fam) & (b["req_tumor"] == 128)]
    if len(g):
        band(ax, g, "tumor_log_ratio", FAM_C[fam], fam)
t = np.linspace(0, df["hours"].max(), 50)
ax.plot(t, 2.99e-4 * 60 * t, "k--", lw=1.2, label="unopposed growth")
ax.set_xlabel("hours")
ax.set_ylabel(r"$\log(c_t / c_0)$")
ax.set_title("How much growth the immune compartment removes")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/03_growth_vs_unopposed.png", dpi=150)
plt.close(fig)

# ── 4. immune pressure axis ─────────────────────────────────────────────────
im = df[df["arm"].isin(["immune", "burden"])]
im = im[(im["req_tumor"] == 128)]
im["immune_level"] = im["req_macrophage"].astype(str) + "/" + im["req_t_cell"].astype(str)
levels = sorted(im["immune_level"].unique(),
                key=lambda s: int(s.split("/")[0]))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, fam in zip(axes, ["network_field", "rectangle"]):
    g = im[im["ic_family"] == fam]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(levels)))
    for lv, c in zip(levels, cmap):
        gg = g[g["immune_level"] == lv]
        if len(gg):
            band(ax, gg, "n_tumor", c, f"mac/T = {lv}")
    ax.axhline(256, color="k", ls="--", lw=0.8)
    ax.set_title(fam, fontsize=10)
    ax.set_xlabel("hours")
axes[0].set_ylabel("tumour cells")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Immune pressure axis, tumour fixed at 128", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/04_immune_pressure.png", dpi=150)
plt.close(fig)

# ── 5. mechanism ablations ──────────────────────────────────────────────────
ab = df[df["arm"] == "ablation"]
base = b[(b["ic_family"] == "network_field") & (b["req_tumor"] == 128)]
fig, ax = plt.subplots(figsize=(7, 4.2))
band(ax, base, "n_tumor", "#1f77b4", "full (128/32/32)")
for cid, c in zip(sorted(ab["config_id"].unique()),
                  ["#d62728", "#ff7f0e", "#8c564b"]):
    g = ab[ab["config_id"] == cid]
    band(ax, g, "n_tumor", c, cid.replace("ablation__", ""))
ax.axhline(256, color="k", ls="--", lw=0.8)
ax.set_xlabel("hours")
ax.set_ylabel("tumour cells")
ax.set_title("Which population is doing the work (network_field, tumour 128)")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/05_ablations.png", dpi=150)
plt.close(fig)

# ── 6. outcome composition + time to escape ─────────────────────────────────
last = df.sort_values("step").groupby(["config_id", "seed"]).tail(1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
comp = (last[last["arm"] == "burden"]
        .groupby(["ic_family", "req_tumor"])["end_reason"]
        .value_counts(normalize=True).unstack(fill_value=0))
comp.plot(kind="bar", stacked=True, ax=ax,
          color=[OUT_C.get(c, "#ccc") for c in comp.columns], width=0.85)
ax.set_ylabel("fraction of seeds")
ax.set_title("Outcome composition")
ax.tick_params(axis="x", labelsize=7, rotation=90)
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
esc = last[(last["end_reason"] == "escape") & (last["arm"] == "burden")]
if len(esc):
    for j, fam in enumerate(fams):
        g = esc[esc["ic_family"] == fam]
        if len(g):
            ax.scatter(g["req_tumor"] + (j - 1.5) * 6, g["end_step"] / 4.0,
                       color=FAM_C[fam], s=22, alpha=0.8, label=fam)
    ax.set_xlabel("initial tumour count")
    ax.set_ylabel("hours to escape")
    ax.set_title("Time to escape")
    ax.legend(fontsize=8, frameon=False)
else:
    ax.text(0.5, 0.5, "no escapes recorded", ha="center", transform=ax.transAxes)
fig.tight_layout()
fig.savefig(f"{FIG}/06_outcomes.png", dpi=150)
plt.close(fig)

# ── 7. initial-condition realisation ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
for fam in fams:
    g = ep[ep["ic_family"] == fam]
    ax.scatter(g["req_tumor"], g["ic_realised_tumor"], color=FAM_C[fam],
               s=26, alpha=0.75, label=fam)
lim = [0, ep["req_tumor"].max() * 1.05]
ax.plot(lim, lim, "k--", lw=0.9)
ax.set_xlabel("requested tumour cells")
ax.set_ylabel("realised (after grid snap + dedup)")
ax.set_title("Requested counts are an upper bound")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/07_ic_realisation.png", dpi=150)
plt.close(fig)

# ── 8. state-space comparison: cost, motion, and what they track ────────────
encs = sorted({c.split("__")[0] for c in df.columns if c.endswith("__l2")})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
cost = [(e, df[f"{e}__wall_s"].median() * 1e3) for e in encs]
cost.sort(key=lambda kv: kv[1])
ax.barh([c[0] for c in cost], [c[1] for c in cost], color="#4c78a8")
ax.set_xlabel("median ms per step")
ax.set_title("Cost of each state space")
ax.tick_params(labelsize=8)

ax = axes[1]
for e in encs:
    col = f"{e}__l2_delta_rel"
    if col in df:
        g = df[df["step"] > 1].groupby("step")[col].median()
        ax.plot(g.index / 4.0, g.values, lw=1.1, label=e)
ax.set_yscale("log")
ax.set_xlabel("hours")
ax.set_ylabel(r"$\|e_t - e_{t-1}\| / \|e_t\|$")
ax.set_title("Step-to-step jumpiness")
ax.legend(fontsize=6, frameon=False, ncol=2)

ax = axes[2]
# how well does each state space track the tumour count, linearly, per episode
from numpy.linalg import lstsq  # noqa: E402

rows = []
for e in encs:
    sub = df[df["step"] > 0]
    x = sub[f"{e}__l2"].to_numpy()
    y = sub["n_tumor"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 10 and np.std(x[ok]) > 0:
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        rows.append((e, abs(r)))
rows.sort(key=lambda kv: kv[1])
ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#e45756")
ax.set_xlabel("|correlation| with tumour count")
ax.set_title("Does the encoding norm track the tumour?")
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/08_state_spaces.png", dpi=150)
plt.close(fig)

print(f"figures -> {FIG}")
for f in sorted(os.listdir(FIG)):
    print("  ", f)
