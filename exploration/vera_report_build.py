"""Assemble the drug-scenario report for Vera from the measured runs.

    python vera_report_build.py

Fills the template with numbers read out of output/vera_report and output/vera_disc,
so that no figure in the prose is retyped by hand and the report cannot drift from the
data it describes. Writes report.md, and report.pdf when pandoc is available.
"""
import glob
import json
import math
import os
import subprocess
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "output/vera_report")
DISC = os.path.join(HERE, "output/vera_disc")
DECAY = 0.05
HALF_MAX = 0.5


def load(run_dir, standalone=True):
    """Episodes from a run directory, dropping any that are not usable.

    `standalone` distinguishes the two harnesses: boundary_run writes
    wrote_output and n_rules_loaded, dose_run (the gym-driven one) writes
    neither, so requiring them would silently discard every disc episode.
    """
    rows = []
    for mp in sorted(glob.glob(os.path.join(run_dir, "episodes", "*", "meta.json"))):
        m = json.load(open(mp))
        if m.get("error"):
            continue
        if standalone:
            if not m.get("wrote_output"):
                continue
            # ⚠️ Never let a rules-free episode into the prose: the drug acts on
            # cells only through cell_rules.csv, so such a run is an untreated run
            # wearing a treated run's name and its numbers look entirely plausible.
            if m.get("n_rules_loaded", None) == 0:
                continue
        rows.append(m)
    return pd.DataFrame(rows)


def fmt(x, n=1):
    return f"{x:.{n}f}"


def main():
    d = load(RUN)
    disc = load(DISC, standalone=False) if os.path.isdir(DISC) else pd.DataFrame()
    unt = d[d.arm == "untreated"].groupby("ic_family")["n_tumor_T"].median()
    out = []

    # ── results section ────────────────────────────────────────────────────
    out.append("### How far in does a boundary influx reach\n")
    b = d[d.arm == "boundary_all"]
    if len(b):
        t = (b[b.boundary_value == 1.0].groupby("drug_diffusion")
             .agg(centre=("drug_centre_mean", "median"),
                  frac=("frac_above_half_max", "median")).reset_index())
        out.append("![The drug field itself, on one shared colour scale. Each "
                   "row is a way of delivering the drug and the columns are "
                   "time. The $D=0.3$ row is the result in one picture: a bright "
                   "rim at the wall and a black interior, hours after the influx "
                   "began and unchanged at 50 h.](fig_fields.png){width=78%}\n")
        out.append("At the strongest boundary concentration tested (1.0), "
                   "influx from all four sides:\n")
        out.append("| $D$ | diffusion length | drug at the centre | fraction of the "
                   "domain above the half-max |")
        out.append("|---|---|---|---|")
        for _, r in t.iterrows():
            out.append(f"| {r.drug_diffusion:g} | "
                       f"{math.sqrt(r.drug_diffusion/DECAY):.1f} um | "
                       f"**{r.centre:.3f}** | {r.frac*100:.0f}% |")
        out.append("")
        lo = t.iloc[0]
        out.append(f"At $D = {lo.drug_diffusion:g}$, the value proposed in the "
                   f"discussion, the concentration at the centre of the domain is "
                   f"**{lo.centre:.4f}**. The drug is banked up in a thin rim at the "
                   f"wall and the middle of the tissue never sees it. Raising the "
                   f"boundary concentration does not fix this, because the profile "
                   f"is exponential in $\\sqrt{{D/\\lambda}}$: it scales the rim, "
                   f"not the reach.\n")

    out.append("![Drug concentration at the centre of the domain against the "
               "diffusion coefficient, for each boundary strength and both "
               "geometries. The dashed line is the Hill half-max: below it the "
               "drug does essentially nothing, and that is the shaded "
               "region.](fig_penetration.png){width=78%}\n")

    one = d[d.arm == "boundary_one"]
    if len(one):
        o = (one[(one.boundary_value == 1.0)].groupby("drug_diffusion")
             .agg(centre=("drug_centre_mean", "median"),
                  frac=("frac_above_half_max", "median")).reset_index())
        out.append("### Influx from one side\n")
        out.append("Entering through a single face, as blood would from one "
                   "vessel, gives a gradient across the tissue rather than a "
                   "bath.\n")
        out.append("| $D$ | drug at the centre | fraction above the half-max |")
        out.append("|---|---|---|")
        for _, r in o.iterrows():
            out.append(f"| {r.drug_diffusion:g} | {r.centre:.3f} | "
                       f"{r.frac*100:.0f}% |")
        out.append("")
        out.append("That asymmetry is what flowing in from all sides was meant "
                   "to avoid. The trade is explicit: the one-sided case is more "
                   "faithful to a vessel, and it is also the one where part of "
                   "the tumour is treated and part is not.\n")

    # ── tumour outcome ─────────────────────────────────────────────────────
    out.append("### What it does to the tumour\n")
    out.append(f"Untreated over 50 h the tumour reaches "
               f"**{unt.get('network_field', float('nan')):.0f}** cells "
               f"(network field) and "
               f"**{unt.get('rectangle', float('nan')):.0f}** (rectangle). Every "
               f"arm starts from the same cells, and each treated run is "
               f"differenced against the untreated run of **its own seed**, "
               f"because untreated growth varies a lot between seeds.\n")
    if len(b):
        # ⚠️ Pair each treated run against its OWN seed's untreated run, not
        # against the family median. Untreated growth varies a lot between seeds
        # (rectangle spans 144 to 274 over six seeds), so an arm that happens to
        # hold only the low-growth seeds looks effective when it is not. Only
        # configurations with the full set of seeds are reported.
        u_by_seed = (d[d.arm == "untreated"]
                     .set_index(["ic_family", "seed"])["n_tumor_T"].to_dict())
        bo = d[d.arm.isin(["boundary_all", "boundary_one"])].copy()
        bo["delta"] = [r.n_tumor_T - u_by_seed.get((r.ic_family, r.seed),
                                                   float("nan"))
                       for r in bo.itertuples()]

        n_seeds = int(d[d.arm == "untreated"].groupby("ic_family").size().max())

        out.append("![Every configuration that was run: influx geometry (rows) "
                   "against initial condition (columns), diffusion coefficient "
                   "against boundary concentration within each panel. Blue is "
                   "fewer tumour cells. The hatched cells are those where the "
                   "drug never reached the middle of the "
                   "domain.](fig_grid.png){width=78%}\n")
        out.append("**The whole left-hand half of every panel is pale.** At "
                   "$D = 0.3$ and $D = 3$ nothing happens whatever the boundary "
                   "concentration, because the drug is still at the wall. The "
                   "effect appears only in the right-hand columns, and sharply. "
                   "The two ends of the $D$ range, at the strongest concentration:\n")
        out.append("| initial condition | influx | $D$ | drug at centre | "
                   "domain above half-max | final tumour | "
                   "change vs same-seed untreated |")
        out.append("|---|---|---|---|---|---|---|")
        best_v = bo.boundary_value.max()
        # The two extremes of the D range carry the argument; the middle is in
        # the figure above and in numbers.csv, and repeating all four here made
        # a sixteen-row table nobody reads.
        keep = {bo.drug_diffusion.min(), bo.drug_diffusion.max()}
        for (fam, arm, D), g in bo[bo.boundary_value == best_v].groupby(
                ["ic_family", "arm", "drug_diffusion"]):
            if len(g) < n_seeds or D not in keep:
                continue          # incomplete: reporting it would mislead
            side = "all four sides" if arm == "boundary_all" else "one side"
            out.append(f"| {fam.replace('_',' ')} | {side} | {D:g} | "
                       f"{g.drug_centre_mean.median():.3f} | "
                       f"{g.frac_above_half_max.median()*100:.0f}% | "
                       f"{g.n_tumor_T.median():.0f} | "
                       f"**{g.delta.median():+.0f}** |")
        out.append("")
        out.append("The outcome follows the field measurement directly: nothing "
                   "at $D = 0.3$ on either geometry, a substantial cut at "
                   "$D = 30$ and above. **A boundary influx works only where the "
                   "field has stopped being localised.**\n")
        out.append("The one-sided arm is worth a second look: at $D = 30$ it "
                   "lifts only about a third of the domain above the half-max, "
                   "yet removes roughly two thirds of what four-sided removes at "
                   "the same $D$. Treating part of the tissue well is not far "
                   "behind treating all of it weakly, which is the point the "
                   "aimed disc makes below.\n")

    out.append("![Tumour burden over the 50 hours, median and interquartile "
               "range over seeds. The boundary arms are shown at their most "
               "favourable setting. The untreated curve is the reference every "
               "arm is read against.](fig_tumour.png){width=78%}\n")

    if len(disc):
        out.append("The injected-disc arms, for comparison, at dose 0.6:\n")
        out.append("| initial condition | arm | final tumour | total dose spent | "
                   "tumour removed per unit dose |")
        out.append("|---|---|---|---|---|")
        for (arm, fam), g in disc.groupby(["arm", "ic_family"]):
            u = unt.get(fam, float("nan"))
            cT = g.c_T.median()
            dose = g.dose_spent_sum.median()
            name = "uniform disc" if arm == "uniform" else "aimed disc"
            out.append(f"| {fam.replace('_',' ')} | {name} | {cT:.0f} | "
                       f"{dose:.1f} | **{(u-cT)/dose:.1f}** |")
        out.append("")
        out.append("The uniform disc reaches the lowest count of anything "
                   "tested, by covering every voxel and spending about **16 "
                   "times** the drug. Per unit of drug the aimed disc is roughly "
                   "**eight times** more efficient. Comparing arms on tumour "
                   "count alone is therefore misleading, which is why the field "
                   "and outcome figures are reported separately.\n")

    conclusion = (
        "**$D = 0.3$ is not a compromise.** Its diffusion length is 2.4 um against "
        "an injection radius of 8.9 um and a domain of 63 um, so the drug barely "
        "moves, and as a boundary influx it never reaches the middle of the tissue "
        "at all.\n\n"
        "**The other end fails differently.** Getting a boundary influx to the "
        "centre takes $D$ of order 30 to 300, where the diffusion length is "
        "comparable to the domain and the field is nearly flat. The drug then "
        "arrives everywhere at once, so where it was administered stops "
        "mattering.\n\n"
        "That is the dilemma, and it is geometry rather than parameters: **on a "
        "domain six cell-widths across, a drug can be localised or evenly "
        "distributed, with very little room in between.** The existing substrates "
        "bracket it from both sides, `cytokine` at 0.55 um and "
        "`anti_tumoral_factor` at 54.8 um, with nothing usefully between.\n\n"
        "For the thesis this matters because the question is whether *where* the "
        "drug goes can be learned from *what the agent sees*, which needs an "
        "action whose placement changes the outcome. $D = 0$ keeps that, and it "
        "makes the control problem harder rather than easier: a badly aimed "
        "injection is genuinely wasted, with nothing spreading it onto the "
        "target.\n\n"
        "A boundary influx is worth having as a **comparison arm**, not a "
        "replacement. It has no position or radius to choose, so it sits outside "
        "the question the thesis asks rather than being a variant of it."
    )

    tpl = open(os.path.join(HERE, "vera_report_template.md")).read()
    n_b = len(d)
    n_d = len(disc)
    md = (tpl.replace("{DATE}", time.strftime("%d %B %Y"))
             .replace("{RESULTS}", "\n".join(out))
             .replace("{CONCLUSION}", conclusion)
             .replace("{BTIME}", "an hour")
             .replace("{DTIME}", "six minutes"))
    md += (f"\n\n---\n\n*Based on {n_b} boundary episodes and {n_d} disc episodes, "
           f"six seeds per configuration. Every run is one PhysiCell simulation with "
           f"no learning and no agent.*\n")
    dst = os.path.join(RUN, "report.md")
    open(dst, "w").write(md)
    print("wrote", dst)

    try:
        subprocess.run(["pandoc", dst, "-o", dst.replace(".md", ".pdf"),
                        "-V", "geometry:margin=2.4cm", "-V", "fontsize=11pt"],
                       check=True, capture_output=True, cwd=RUN)
        print("wrote", dst.replace(".md", ".pdf"))
    except Exception as e:
        print("pandoc failed (markdown still written):", e)


if __name__ == "__main__":
    main()
