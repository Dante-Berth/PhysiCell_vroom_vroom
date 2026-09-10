"""Figures for the drug-scenario report Vera asked for on 2026-09-10.

Reads output/vera_report (boundary arms, standalone binary) and, when present,
output/vera_disc (disc arms, PhysiGym), and writes three figures plus the numbers
the report quotes.

    python vera_figures.py [--run-dir output/vera_report] [--disc-dir output/vera_disc]

⚠️ Never TikZ and never a schematic: every panel is measured from a run.
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DECAY = 0.05            # drug_1 decay rate, so diffusion length is sqrt(D/DECAY)
HALF_MAX = 0.5          # cell_rules.csv:2-3, the drug's Hill half-max on macrophages
HILL = 4
DOMAIN = 63.0

C_ALL = "#1a4f7a"       # influx from all four faces
C_ONE = "#b03030"       # influx from xmin only
C_UNTREATED = "#666666"
C_DISC = "#2e7d32"


def load(run_dir):
    rows = []
    for mp in sorted(glob.glob(os.path.join(run_dir, "episodes", "*", "meta.json"))):
        m = json.load(open(mp))
        if not m.get("wrote_output"):
            continue
        m["_dir"] = os.path.dirname(mp)
        rows.append(m)
    return pd.DataFrame(rows)


def series(rowdir):
    f = os.path.join(rowdir, "field.csv.gz")
    return pd.read_csv(f) if os.path.exists(f) else None


def hill(c):
    return c ** HILL / (HALF_MAX ** HILL + c ** HILL)


# ── figure 1: how deep does a boundary influx reach? ────────────────────────
def fig_penetration(df, out):
    """The central measurement. Concentration at the domain centre against D.

    This is the figure that answers Vera's question directly, and the reason the
    answer is not a matter of taste: the drug only repolarises a macrophage above
    the Hill half-max, so the horizontal line is where the biology switches on.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, arm, colour, title in (
            (axes[0], "boundary_all", C_ALL, "influx from all four sides"),
            (axes[1], "boundary_one", C_ONE, "influx from one side (xmin)")):
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue
        for v, mk in zip(sorted(sub["boundary_value"].unique()), ("o", "s", "^")):
            s = (sub[sub["boundary_value"] == v]
                 .groupby("drug_diffusion")["drug_centre_mean"]
                 .median().reset_index())
            ax.plot(s["drug_diffusion"], s["drug_centre_mean"], marker=mk,
                    color=colour, alpha=0.45 + 0.28 * list(
                        sorted(sub["boundary_value"].unique())).index(v),
                    label=f"boundary = {v:g}")
        ax.axhline(HALF_MAX, color="k", ls="--", lw=0.9)
        ax.text(0.98, HALF_MAX, " Hill half-max: the drug does nothing below this",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7.2)
        ax.set_xscale("log")
        ax.set_xlabel("drug diffusion coefficient $D$  ($\\mu$m$^2$/min)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7.5, frameon=False)
    axes[0].set_ylabel("drug at the domain centre")

    # A second x axis in the units that actually govern the physics.
    for ax in axes:
        sec = ax.secondary_xaxis(
            "top", functions=(lambda D: np.sqrt(np.maximum(D, 1e-12) / DECAY),
                              lambda L: DECAY * L ** 2))
        sec.set_xlabel("diffusion length $\\sqrt{D/\\lambda}$  ($\\mu$m)"
                       f"      [domain is {DOMAIN:g} $\\mu$m]", fontsize=8)
    fig.suptitle("A boundary influx cannot be both localised and effective on this "
                 "domain", fontsize=10.5, y=1.06)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("wrote", out)


# ── figure 2: the tumour, which is what was actually asked for ──────────────
def fig_tumour(df, out, disc=None):
    """n_tumor against time, 0-50 h. Exactly the axes promised to Vera."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, fam in zip(axes, ["network_field", "rectangle"]):
        d = df[df["ic_family"] == fam]
        if d.empty:
            continue
        drawn = []
        for arm, colour, lab in (("untreated", C_UNTREATED, "untreated"),
                                 ("boundary_all", C_ALL, "all four sides"),
                                 ("boundary_one", C_ONE, "one side")):
            sub = d[d["arm"] == arm]
            if arm != "untreated":
                # Show the most favourable setting the grid contains, so the arm
                # is not made to look bad by a concentration that was never going
                # to work: strongest boundary value at the most penetrating D.
                sub = sub[(sub["boundary_value"] == sub["boundary_value"].max())
                          & (sub["drug_diffusion"] == sub["drug_diffusion"].max())]
            if sub.empty:
                continue
            curves = [s for s in (series(r) for r in sub["_dir"]) if s is not None]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            M = np.vstack([c["n_tumor"].to_numpy()[:n] for c in curves])
            t = curves[0]["sim_time_min"].to_numpy()[:n] / 60.0
            med = np.median(M, axis=0)
            lo, hi = np.percentile(M, [25, 75], axis=0)
            ax.plot(t, med, color=colour, lw=1.7, label=lab)
            ax.fill_between(t, lo, hi, color=colour, alpha=0.16, lw=0)
            drawn.append(arm)
        if disc is not None:
            dd = disc[disc["ic_family"] == fam]
            for arm, ls, lab in (("uniform", "--", "disc, uniform (PhysiGym)"),
                                 ("grid_m2", ":", "disc, aimed (PhysiGym)")):
                sub = dd[dd["arm"] == arm]
                if sub.empty:
                    continue
                curves = [s for s in (disc_series(r) for r in sub["_dir"])
                          if s is not None]
                if not curves:
                    continue
                n = min(len(c) for c in curves)
                M = np.vstack([c["n_tumor"].to_numpy()[:n] for c in curves])
                t = curves[0]["sim_time_min"].to_numpy()[:n] / 60.0
                ax.plot(t, np.median(M, axis=0), color=C_DISC, ls=ls, lw=1.6,
                        label=lab)
        ax.set_title(f"{fam.replace('_', ' ')} initial condition", fontsize=9.5)
        ax.set_xlabel("time (h)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("tumour cells")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Tumour burden under each way of delivering the drug "
                 "(median, IQR over seeds)", fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("wrote", out)


def disc_series(rowdir):
    f = os.path.join(rowdir, "steps.csv.gz")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, usecols=["step", "n_tumor", "sim_time_min"])
    return d.rename(columns={"step": "frame"})


# ── figure 3: what the field looks like, which is the intuition ────────────
def fig_fields(run_dir, out):
    """Drug field snapshots. One row per actuator, columns in time."""
    import scipy.io as sio
    picks = []
    for arm, D, v in (("untreated", 0.3, 0.0),
                      ("boundary_all", 30.0, 1.0),
                      ("boundary_one", 30.0, 1.0),
                      ("boundary_all", 0.3, 1.0)):
        cid = (f"untreated__network_field__s0" if arm == "untreated"
               else f"{arm}__D{D:g}__v{v:g}__network_field__s0")
        d = os.path.join(run_dir, "work", cid, "output", "episode00000000")
        if os.path.isdir(d):
            picks.append((arm, D, v, d))
    if not picks:
        print("no field data found")
        return
    cols = [0, 4, 20, 200]
    fig, axes = plt.subplots(len(picks), len(cols),
                             figsize=(2.05 * len(cols), 2.15 * len(picks)))
    axes = np.atleast_2d(axes)
    for i, (arm, D, v, d) in enumerate(picks):
        frames = sorted(glob.glob(os.path.join(d, "output*_microenvironment0.mat")))
        for j, ci in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            k = min(ci, len(frames) - 1)
            if not frames:
                continue
            m = sio.loadmat(frames[k])
            A = m[[x for x in m if not x.startswith("__")][0]]
            x, y, drug = A[0], A[1], A[6]
            g = pd.DataFrame({"x": x, "y": y, "c": drug}).pivot(
                index="y", columns="x", values="c")
            # One SHARED scale across the whole figure: unlike fig:tme:diffusion,
            # here the absolute level is the point (does it clear the half-max?),
            # so a per-panel scale would hide exactly what matters.
            ax.imshow(g.values, origin="lower", cmap="magma", vmin=0.0, vmax=1.0,
                      extent=[0, DOMAIN, 0, DOMAIN], interpolation="nearest")
            if i == 0:
                ax.set_title(f"t = {k * 15 / 60:.0f} h", fontsize=8.5)
            if j == 0:
                lab = ("untreated" if arm == "untreated"
                       else ("all sides" if arm == "boundary_all" else "one side")
                       + f"\n$D={D:g}$")
                ax.set_ylabel(lab, fontsize=8)
    fig.suptitle("The drug_1 field. Shared colour scale, 0 to 1; the Hill half-max "
                 "is 0.5", fontsize=10, y=1.005)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("wrote", out)


def numbers(df, out):
    """The table the report quotes, so no number in the prose is retyped by hand."""
    g = (df[df["arm"] != "untreated"]
         .groupby(["arm", "drug_diffusion", "boundary_value"])
         .agg(centre=("drug_centre_mean", "median"),
              interior=("drug_interior_mean", "median"),
              frac_half=("frac_above_half_max", "median"),
              n_tumor_T=("n_tumor_T", "median"),
              n=("seed", "count")).reset_index())
    g["diff_len"] = np.sqrt(g["drug_diffusion"] / DECAY)
    g["macrophage_response"] = hill(g["centre"])
    u = df[df["arm"] == "untreated"]
    g.to_csv(out, index=False)
    print("wrote", out)
    if len(u):
        print(f"untreated: n={len(u)}  drug_max={u['drug_max'].max():.3g}  "
              f"tumour {u['n_tumor_0'].median():.0f} -> {u['n_tumor_T'].median():.0f}")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "output/vera_report"))
    ap.add_argument("--disc-dir", default=os.path.join(HERE, "output/vera_disc"))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out = a.out_dir or a.run_dir
    os.makedirs(out, exist_ok=True)

    df = load(a.run_dir)
    print(f"{len(df)} boundary episodes")
    disc = None
    if os.path.isdir(a.disc_dir):
        disc = load_disc(a.disc_dir)
        print(f"{0 if disc is None else len(disc)} disc episodes")

    fig_penetration(df, os.path.join(out, "fig_penetration.pdf"))
    fig_tumour(df, os.path.join(out, "fig_tumour.pdf"), disc)
    fig_fields(a.run_dir, os.path.join(out, "fig_fields.pdf"))
    numbers(df, os.path.join(out, "numbers.csv"))


def load_disc(d):
    rows = []
    for mp in sorted(glob.glob(os.path.join(d, "episodes", "*", "meta.json"))):
        m = json.load(open(mp))
        m["_dir"] = os.path.dirname(mp)
        m.setdefault("ic_family", m.get("ic_family", ""))
        rows.append(m)
    return pd.DataFrame(rows) if rows else None


if __name__ == "__main__":
    main()
