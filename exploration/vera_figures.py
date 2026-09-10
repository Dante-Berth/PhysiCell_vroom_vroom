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
    rows, skipped = [], 0
    for mp in sorted(glob.glob(os.path.join(run_dir, "episodes", "*", "meta.json"))):
        m = json.load(open(mp))
        if not m.get("wrote_output") or m.get("error"):
            skipped += 1
            continue
        # ⚠️ Refuse any episode that ran without cell rules. drug_1 acts on cells
        # only through them, so a rules-free run is an untreated run under a
        # treated run's name and its numbers look entirely plausible. A whole
        # 156-episode grid was lost to this once; never plot one.
        if m.get("n_rules_loaded", None) == 0:
            skipped += 1
            continue
        m["_dir"] = os.path.dirname(mp)
        rows.append(m)
    if skipped:
        print(f"  skipped {skipped} unusable episodes in {os.path.basename(run_dir)}")
    df = pd.DataFrame(rows)
    if len(df) and "n_rules_loaded" in df.columns:
        missing = int((df["n_rules_loaded"].fillna(-1) < 0).sum())
        if missing:
            print(f"  ⚠️  {missing} episodes predate the rules check; verify them")
    return df


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
            # Data for this arm has not landed yet. Say so on the panel rather
            # than leaving an empty axes that reads as a measured zero.
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#999")
            ax.set_title(title, fontsize=9.5, pad=34)
            ax.set_xlabel("drug diffusion coefficient $D$  ($\\mu$m$^2$/min)")
            continue
        vals = sorted(sub["boundary_value"].unique())
        for i, (v, mk) in enumerate(zip(vals, ("o", "s", "^", "D"))):
            g = (sub[sub["boundary_value"] == v]
                 .groupby("drug_diffusion")["drug_centre_mean"]
                 .median().reset_index())
            # Darker line = stronger boundary concentration.
            alpha = 0.40 + 0.55 * (i / max(len(vals) - 1, 1))
            ax.plot(g["drug_diffusion"], g["drug_centre_mean"], marker=mk,
                    color=colour, alpha=min(alpha, 1.0), lw=1.6, ms=5,
                    label=f"boundary = {v:g}")
        ax.axhspan(0, HALF_MAX, color="k", alpha=0.055, lw=0)
        ax.axhline(HALF_MAX, color="k", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("drug diffusion coefficient $D$  ($\\mu$m$^2$/min)")
        ax.set_title(title, fontsize=9.5, pad=34)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        # Label the diffusion length at each measured D, rather than a second
        # continuous axis: only four values were run, and a log-scaled twin axis
        # collides with the tick labels underneath it.
        for D in sorted(df["drug_diffusion"].unique()):
            ax.annotate(f"{math.sqrt(D / DECAY):.1f}", xy=(D, 1.02),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="bottom", fontsize=7, color="#444")
        ax.annotate("diffusion length $\\sqrt{D/\\lambda}$ ($\\mu$m)",
                    xy=(0.5, 1.13), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=7.5, color="#444")
    axes[0].set_ylabel("drug at the domain centre")
    axes[0].text(0.03, HALF_MAX - 0.02,
                 "below this line the drug does essentially nothing",
                 transform=axes[0].get_yaxis_transform(), ha="left", va="top",
                 fontsize=7.2, color="#333")
    fig.suptitle(f"A boundary influx reaches the centre only once it is no longer "
                 f"localised\n(domain is {DOMAIN:g} $\\mu$m across)",
                 fontsize=10.5, y=1.10)
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
                      ("boundary_all", 0.3, 1.0),
                      ("boundary_all", 30.0, 1.0),
                      ("boundary_one", 30.0, 1.0)):
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
            im = ax.imshow(g.values, origin="lower", cmap="magma", vmin=0.0,
                           vmax=1.0, extent=[0, DOMAIN, 0, DOMAIN],
                           interpolation="nearest")
            if i == 0:
                ax.set_title(f"t = {k * 15 / 60:.0f} h", fontsize=8.5)
            if j == 0:
                lab = ("untreated" if arm == "untreated"
                       else ("all sides" if arm == "boundary_all" else "one side")
                       + f"\n$D={D:g}$")
                ax.set_ylabel(lab, fontsize=8)
    fig.suptitle("The drug field, on one shared colour scale", fontsize=10.5,
                 y=1.005)
    # A colourbar with the Hill half-max marked on it, so "is this dark region
    # doing anything?" can be read off the figure instead of inferred.
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.015)
    cb.set_label("drug concentration", fontsize=8, labelpad=26)
    cb.ax.axhline(HALF_MAX, color="w", lw=1.4)
    cb.ax.axhline(HALF_MAX, color="k", lw=0.7, ls="--")
    cb.ax.text(1.45, HALF_MAX, "Hill half-max", fontsize=6.8, va="bottom",
               ha="left", transform=cb.ax.get_yaxis_transform())
    cb.ax.text(1.45, HALF_MAX, "no effect below", fontsize=6.8, va="top",
               ha="left", color="#666", transform=cb.ax.get_yaxis_transform())
    cb.ax.tick_params(labelsize=7)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("wrote", out)


def with_delta(df):
    """Difference every treated run against the untreated run of its OWN seed.

    Untreated growth varies a lot between seeds (rectangle spans 144 to 274 over
    six), so differencing against a family median lets seed luck masquerade as a
    treatment effect. Pairing on the seed cancels it.
    """
    u = (df[df["arm"] == "untreated"]
         .set_index(["ic_family", "seed"])["n_tumor_T"].to_dict())
    t = df[df["arm"] != "untreated"].copy()
    t["untreated_same_seed"] = [u.get((f, s), float("nan"))
                                for f, s in zip(t.ic_family, t.seed)]
    t["delta"] = t.n_tumor_T - t.untreated_same_seed
    return t


def numbers(df, out):
    """The full grid, one row per configuration, keyed on the same-seed delta.

    Kept per initial-condition family rather than pooled: the two families have
    different untreated growth, so pooling them averages over the thing the
    delta is measuring.
    """
    t = with_delta(df)
    g = (t.groupby(["arm", "ic_family", "drug_diffusion", "boundary_value"])
         .agg(centre=("drug_centre_mean", "median"),
              interior=("drug_interior_mean", "median"),
              frac_half=("frac_above_half_max", "median"),
              n_tumor_T=("n_tumor_T", "median"),
              untreated_same_seed=("untreated_same_seed", "median"),
              delta=("delta", "median"),
              n=("seed", "count")).reset_index())
    g["diff_len"] = np.sqrt(g["drug_diffusion"] / DECAY)
    g["macrophage_response"] = hill(g["centre"])
    g = g.sort_values(["arm", "ic_family", "drug_diffusion", "boundary_value"])
    u = df[df["arm"] == "untreated"]
    g.to_csv(out, index=False)
    print("wrote", out)
    if len(u):
        print(f"untreated: n={len(u)}  drug_max={u['drug_max'].max():.3g}  "
              f"tumour {u['n_tumor_0'].median():.0f} -> {u['n_tumor_T'].median():.0f}")
    return g


def fig_grid(df, out):
    """The whole grid as a picture, so the report need not send anyone to a CSV.

    Rows are the two influx geometries, columns the two initial conditions. Each
    cell is diffusion coefficient against boundary concentration, coloured by how
    much the tumour changed against the same seed untreated, and annotated with
    that number. The hatched cells are the ones where the drug never reached the
    middle.
    """
    t = with_delta(df)
    arms = [a for a in ("boundary_all", "boundary_one") if (t.arm == a).any()]
    fams = sorted(t.ic_family.unique())
    if not arms or not fams:
        print("no data for the grid figure")
        return
    Ds = sorted(t.drug_diffusion.unique())
    Vs = sorted(t.boundary_value.unique())

    lim = float(np.nanmax(np.abs(t.delta))) or 1.0
    fig, axes = plt.subplots(len(arms), len(fams),
                             figsize=(3.5 * len(fams), 3.0 * len(arms)),
                             squeeze=False)
    for i, arm in enumerate(arms):
        for j, fam in enumerate(fams):
            ax = axes[i][j]
            sub = t[(t.arm == arm) & (t.ic_family == fam)]
            M = np.full((len(Vs), len(Ds)), np.nan)
            C = np.full((len(Vs), len(Ds)), np.nan)
            for r, v in enumerate(Vs):
                for c, D in enumerate(Ds):
                    g = sub[(sub.boundary_value == v) & (sub.drug_diffusion == D)]
                    if len(g):
                        M[r, c] = g.delta.median()
                        C[r, c] = g.drug_centre_mean.median()
            im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
                           aspect="auto")
            for r in range(len(Vs)):
                for c in range(len(Ds)):
                    if np.isnan(M[r, c]):
                        continue
                    # Mark the cells where the centre never cleared the half-max:
                    # a tumour change there cannot be the drug acting centrally.
                    if C[r, c] < HALF_MAX:
                        ax.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1, fill=False, lw=0,
                            hatch="///", edgecolor="#00000028"))
                    ax.text(c, r, f"{M[r, c]:+.0f}", ha="center", va="center",
                            fontsize=8.5,
                            color="w" if abs(M[r, c]) > 0.55 * lim else "#222")
            ax.set_xticks(range(len(Ds)))
            ax.set_xticklabels([f"{D:g}\n({math.sqrt(D/DECAY):.1f})" for D in Ds],
                               fontsize=7.5)
            ax.set_yticks(range(len(Vs)))
            ax.set_yticklabels([f"{v:g}" for v in Vs], fontsize=8)
            if i == 0:
                ax.set_title(fam.replace("_", " ") + " initial condition",
                             fontsize=9.5)
            if j == 0:
                ax.set_ylabel(("all four sides" if arm == "boundary_all"
                               else "one side")
                              + "\n\nboundary concentration", fontsize=8.5)
            if i == len(arms) - 1:
                ax.set_xlabel("diffusion coefficient $D$\n(diffusion length, "
                              "$\\mu$m)", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cb.set_label("change in tumour cells against the same seed untreated",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Every configuration that was run. Blue is fewer tumour cells, "
                 "white is no effect;\nhatched cells are those where the drug "
                 "never reached the middle of the domain", fontsize=10.5, y=1.05)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("wrote", out)


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
    fig_grid(df, os.path.join(out, "fig_grid.pdf"))
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
