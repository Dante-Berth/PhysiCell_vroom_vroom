# Drug delivery scenarios in PhysiCell

*Alexandre Bertin, {DATE}. Simulations run in PhysiCell on the laptop; no reinforcement
learning, no trained policy, no observation spaces anywhere in this report.*

## What was asked

Vera asked for simulations of the drug in different scenarios, run in PhysiCell and
setting PhysiGym aside, with the drug arriving either uniformly or by diffusing in from
the boundary, "as if it was blood flowing in from the circulation", and from all sides
rather than one so that the geometry stays symmetric.

This report answers that. It compares five ways of delivering the same drug and reports
what each does to the drug field and to the tumour over 50 hours.

## One number first, because everything below follows from it

The simulated domain is **63 x 63 micrometres**. A cell is about 10 micrometres across,
so the whole domain is roughly six cells wide. That is small, and it is deliberate: it is
what makes an episode cheap enough to train on.

The consequence is that "how far does a molecule travel before it decays" is decisive.
That distance is the **diffusion length**, `sqrt(D / lambda)`, where `D` is the diffusion
coefficient and `lambda` the decay rate. For the substrates already in the model:

| substrate | D | decay | diffusion length | reaches |
|---|---|---|---|---|
| `anti_tumoral_factor` | 3 | 0.001 | **54.8 um** | most of the domain |
| `pro_tumoral_factor` | 3 | 0.001 | **54.8 um** | most of the domain |
| `tumor_molecule` | 2 | 0.1 | 4.5 um | a few cell widths |
| `cytokine` | 0.003 | 0.01 | 0.55 um | touching distance |
| `drug_1` **as shipped** | 0 | 0.05 | 0 | exactly where it is put |

So the existing substrates already span the full range from "touching distance" to
"the whole domain". There is no room in between on a 63 um domain: a molecule either
barely moves or it fills the box.

## How the drug works in this model, which constrains what any delivery can do

`drug_1` has **no direct effect on tumour cells at all**. It acts on macrophages only
(`config/cell_rules.csv`): it decreases their pro-tumoral secretion and increases their
anti-tumoral secretion. The anti-tumoral factor is the beacon T cells follow, and the
T cells do the killing.

Two consequences matter for everything below.

1. The dose-response is a **sharp threshold**, not a gradient: a Hill function with
   half-max **0.5** and coefficient **4**. At concentration 0.25 the response is 0.06,
   essentially nothing; at 0.5 it is half; at 0.75 it is 0.84. So the question for any
   delivery scheme is not "how much drug arrives" but **"what fraction of the domain
   clears 0.5"**.
2. The drug's effect is **indirect and slow**: drug, then repolarisation, then beacon,
   then T-cell migration, then killing. Fifty hours is not long for that chain.

## The five scenarios

| # | scenario | how the drug arrives | needs PhysiGym? |
|---|---|---|---|
| 1 | untreated | not at all | no |
| 2 | boundary influx, all four sides | Dirichlet condition on every face | **no** |
| 3 | boundary influx, one side | Dirichlet condition on `xmin` | **no** |
| 4 | uniform disc | injected over the whole domain | yes |
| 5 | aimed disc | injected on a small disc, placed on the M2 macrophages | yes |

Scenarios 2 and 3 are Vera's. They run on the **standalone PhysiCell binary**, driven by
nothing but a settings XML: `./project settings.xml`. No Python is in the loop.

### Why scenarios 4 and 5 still need PhysiGym, stated plainly

The disc injection is implemented by a function, `add_local_substrate`, that has **no
caller in the C++ code at all**. It is reached only from the Python bridge. So a disc
injection cannot be run standalone without writing new C++, whereas a boundary influx
needs none, because PhysiCell already parses per-face boundary conditions out of the XML.

That is a fact about where the actuator lives in the code, not a modelling choice. All
five scenarios share one set of cell rules and one set of initial-condition files, so the
comparison is fair; the report says which half ran which way rather than implying
everything ran the same.

## Results

{RESULTS}

## What this says about the drug diffusion question

{CONCLUSION}

## Reproducing this

```bash
cd PhysiCell_vroom_vroom/exploration
python boundary_driver.py --run-tag vera_report --workers 8 --seeds 6 --hours 50
python dose_driver.py --run-tag vera_disc --workers 4 --seeds 6 --max-steps 200 \
    --no-terminate --split-seeds --arms uniform,grid_m2 --only "d0.6__"
python vera_figures.py
```

Everything is on the laptop and needs no GPU. The boundary arms take about {BTIME} on
eight workers; the disc arms about {DTIME} on four.
