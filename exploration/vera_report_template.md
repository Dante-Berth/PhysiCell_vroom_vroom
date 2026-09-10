# Drug delivery scenarios in PhysiCell

*Alexandre Bertin, {DATE}. PhysiCell on the laptop; no reinforcement learning, no
trained policy, no observation spaces anywhere in this report.*

## What was asked, and the short answer

You asked for simulations of the drug arriving either uniformly or diffusing in from
the boundary, "as if it was blood flowing in from the circulation", and from all sides
rather than one so the geometry stays symmetric. Five delivery scenarios, 50 hours,
six seeds each.

**The short answer: a boundary influx works, but only once it is no longer localised.**
At a small diffusion coefficient the drug never reaches the middle of the tissue; by
the time it does, the field is nearly flat and where it was administered no longer
matters. That is a property of the geometry, not of the parameters.

## Why the domain size decides everything

The domain is **63 x 63 micrometres**, about six cell widths. So how far a molecule
travels before it decays, the **diffusion length** $\sqrt{D/\lambda}$, is decisive:

| substrate | D | decay | diffusion length | reaches |
|---|---|---|---|---|
| `anti_tumoral_factor`, `pro_tumoral_factor` | 3 | 0.001 | **54.8 um** | most of the domain |
| `tumor_molecule` | 2 | 0.1 | 4.5 um | a few cell widths |
| `cytokine` | 0.003 | 0.01 | 0.55 um | touching distance |
| `drug_1` **as shipped** | 0 | 0.05 | 0 | exactly where it is put |

The existing substrates already span touching distance to the whole box. There is very
little room in between.

**The drug acts on macrophages only**, not on tumour cells: it lowers their pro-tumoral
secretion and raises the anti-tumoral one, which is the beacon T cells follow. Two
consequences. The dose-response is a sharp threshold, a Hill function with half-max
**0.5** and coefficient 4, so at concentration 0.25 the response is 0.06, essentially
nothing. And the effect is indirect and slow: drug, repolarisation, beacon, T-cell
migration, killing.

## The five scenarios

| scenario | how the drug arrives | needs PhysiGym? |
|---|---|---|
| untreated | not at all | no |
| boundary influx, all four sides | Dirichlet condition on every face | **no** |
| boundary influx, one side | Dirichlet condition on `xmin` | **no** |
| uniform disc | injected over the whole domain | yes |
| aimed disc | injected on a small disc, on the M2 macrophages | yes |

The boundary arms run on the standalone binary, `./project settings.xml`, with no Python
in the loop. The disc arms cannot: `add_local_substrate` has no caller in the C++ at all
and is reached only from the Python bridge. That is a fact about where the actuator lives
in the code, not a modelling choice. All five share one set of cell rules and one set of
initial conditions.

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
python vera_figures.py && python vera_report_build.py
```

Laptop only, no GPU: about an hour for the boundary arms on eight workers, six minutes
for the disc arms on four.
