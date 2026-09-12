# Timed dosing under a drug budget: can images beat scalars without a spatial action?

_Written 2026-09-13. Design note for an experiment that is **not** in the thesis and is
not scheduled before deposit (30 Sep 2026)._

> **Status: designed, not run, not staged.** Nothing here has been measured. Every number
> below is either (a) carried in from a measurement recorded elsewhere and labelled as such,
> or (b) a prediction. Do not cite (b) anywhere.

---

## 0. The one-paragraph version

Under the uniform action the thesis's nine observation modes **collapse** — images buy
nothing over four scalars (§6.1, measured). That is the thesis's headline and it must not
be disturbed. This note designs a *different* experiment that reopens the observation
question **without restoring a spatial action**: keep the action non-spatial, but make the
drug **scarce** (an episode budget, so the decision is *when* to spend) and make the drug
field **anisotropic** (a one-sided boundary influx, so *where the tumour sits* decides
whether spending now is wasted). The prediction is that images beat scalars in that
setting, and that the advantage appears **graded** across three actuators — absent for the
uniform disc, weak for all-sides influx, strong for one-sided influx.

The thesis already names half of this in `ch7_discussion.tex:435-465` ("a weaker removal of
spatial freedom": uniform dose + episode budget + choose when). **The genuinely new part
of this note is the boundary influx as the anisotropy knob, and the three-arm graded
design that turns a single effect into a mechanism.**

---

## 1. Why the naive version of this idea is self-defeating

Worth stating first, because it is the trap.

"Give the drug uniformly / from all sides / from one side, and show images beat scalars"
**cannot** be the goal as stated. §6.1 (`ch6_results.tex:71-85`) argues that under a
uniform action the modes collapse to within **6.9 return units** (sd 2.0, measured) against
a **73.9** spread under targeted action, and that this collapse is the *confirmation* of
`fig:intro:claim`: spatial observation matters **iff** the action is spatially targeted.

So an image advantage under a plain uniform action would **falsify the thesis**, not
support it. The experiment below is only interesting because it sits on the *boundary* of
the claim's third condition, and `ch7_discussion.tex:449-458` says so explicitly: either
outcome sharpens the condition, and the claim predicts neither.

| outcome | what it would mean for the claim |
|---|---|
| gap **reappears** | "spatially targeted" must widen to cover actions whose *consequences* are spatial even when their *parameters* are not |
| gap **stays closed** | the condition means what it literally says — a **stronger** reading than current evidence supports |

Both are publishable. That is what makes it worth running rather than merely possible.

---

## 2. What is already measured, and where

Carried in from the memory notes and the manuscript. **These stand; they are not
predictions.**

### 2.1 The drug field geometry (standalone PhysiCell, no PhysiGym)

Boundary Dirichlet influx works natively, needs no C++ change
(`modules/PhysiCell_settings.cpp:764-853`). Measured at t=3000 min, boundary clamped 1.0,
four sides enabled:

| D | depth sqrt(D/0.05) | centre conc. | interior mean |
|---|---|---|---|
| 0.3 | 2.45 um | **0.0000** | 0.125 |
| 3 | 7.75 um | 0.0668 | 0.400 |
| 30 | 24.5 um | 0.6495 | 0.816 |
| 300 | 77.5 um | 0.9464 | 0.969 |

**One-sided (`xmin` only), D=30:** 1.0 at the wall falling monotonically to 0.153 at x=63.
Crosses the Hill half-max (0.5, `config/cell_rules.csv:2-3`) at **x ~ 16 um**, so only
**28% of the domain is above threshold** and 59% above 0.25.

⚠️ These are **field** numbers and they stand. Tumour-outcome numbers from the first
`vera_report` grid do **not** — those runs loaded zero cell rules. See the memory note
`cell-rules-load-silently-empty`.

### 2.2 Why the domain size forces the D choice

Domain is 64x64 um on 1 um voxels. Diffusion length vs the 8.9 um injection radius:

- D=0.3 -> 2.45 um, **below** the injection radius, so a commanded disc survives intact.
- D=3 (`anti_tumoral_factor`-like) -> 54.8 um, most of the domain, homogeneous by
  construction.

So on this domain "realistic diffusion" and "drug stays put" are nearly exclusive. **This
is why D=30 is the right choice for the one-sided arm**: 24.5 um is long enough to grade
across the domain and short enough not to homogenise it. It is the only D in the table that
is genuinely anisotropic at steady state.

### 2.3 The dose-penalty asymmetry (the confound to route around)

`wrapper.py:758` pays `w_cell*r_cancer - w_dose*dose_spent - w_smooth*smooth`, and
`dose_spent = commanded dose x (fraction of domain covered)`. Measured: uniform coverage
1.00000 vs targeted-largest-disc 0.06299 -> penalty ratio **15.9x**.

Consequence: **a boundary influx and a disc are not charged comparably.** Any cross-arm
comparison on return confounds placement with quantity — the trap `sec:tme:nodiffusion`
already records. **Mitigation: compare within an arm, across observation modes.** That is
the comparison the claim needs anyway, and it is immune to the charging difference.

⚠️ Also: `drug_1_amount_used` **leaks across episodes inside a worker** (`env.reset()`
does not clear it). Exclude step 0 from all dose accounting.

### 2.4 Dose-efficiency, measured 2026-09-10

At matched commanded dose 0.6 over 50 h, 6 seeds x 2 IC families: uniform reaches a lower
final count but spends **15.75x** the drug; per unit dose the aimed disc is ~**8x** more
efficient. Both arms beat untreated. Spread is wide (uniform/rectangle 3--81); six seeds is
enough for a report, not for a claim of separation.

**This is the strongest single argument that a budget changes the problem**: under scarcity
the 8x efficiency gap is what the agent is forced to optimise, and efficiency is where
geometry lives.

---

## 3. The mechanism this experiment turns on

Two independent reasons a *uniform-in-space* dose can still have a *spatially determined*
value. The first is the thesis's; the second is this note's.

**(a) Contact-dependent killing (`ch7_discussion.tex:441-447`, already written).** The drug
does not diffuse and killing is contact-dependent, so repolarising macrophages buys a great
deal when T cells are already close to the tumour and much less when they are not. That is a
spatial fact; aggregate counts cannot report it; an image can. So *timing* is spatially
determined even for a perfectly uniform dose.

**(b) Field anisotropy (new here).** With a one-sided influx, only 28% of the domain is
above the kill threshold. A scalar tumour count cannot distinguish:

- 80 cells hugging the `xmin` wall -> dose now, it will work; and
- 80 cells at the far edge -> dose now is **budget thrown away**.

The scalar modes see the same observation in both cases. The image modes do not. **This is a
clean, constructed POMDP separation that requires no spatial action parameter at all.**

Mechanism (b) is much stronger than (a) because its magnitude is controllable: turning D
from 300 down to 30 to 3 slides the field from homogeneous to sharply graded, which is the
experimental knob that makes the graded prediction in §4 testable.

---

## 4. The design

### 4.1 Three arms, one action space

All arms share a **non-spatial** action: `dose in [0,1]` (plus, in every arm, the same
episode budget). No coordinates anywhere. The arms differ **only** in the actuator geometry.

| # | arm | field shape | prediction |
|---|---|---|---|
| 1 | **uniform disc + budget** | isotropic, whole domain | **no** image advantage (must reproduce §6.1's collapse) |
| 2 | **all-sides influx + budget**, D=30 | radially symmetric, edge-to-centre gradient | **weak** advantage (a radius scalar captures most of it) |
| 3 | **one-sided influx + budget**, D=30 | monotone gradient, 28% above threshold | **strong** advantage |

**Arm 1 is the control and it is not optional.** It is what makes the result a mechanism
rather than a demo. Run only arm 3 and a reviewer says "you changed two things at once"
(budget *and* geometry) — and they would be right. The graded prediction across 1-2-3
isolates anisotropy as the cause, because the budget is held constant across all three.

Arm 2 is the genuinely informative middle: it has a budget and a gradient but **no
asymmetry**. If the advantage tracks anisotropy rather than merely "non-uniform field",
arm 2 should sit near arm 1. If arm 2 matches arm 3, the driver is the gradient, not the
asymmetry, and mechanism (b) needs restating.

### 4.2 The budget, and the three ways to get it wrong

- **It must bind.** The dose ladder shows uniform **cures** at commanded dose >= 0.4. A
  budget near `0.4 x n_steps` is not a constraint. Set it well below, so the greedy
  dose-every-step policy exhausts it before episode end and genuine timing choices exist.
  ⚠️ **Calibrate this by measurement before the sweep** (see §6, step 1) — a budget guessed
  wrong produces either the §6.1 collapse again (too generous) or an all-seeds-fail floor
  (too tight), and both look like "no effect".
- **It must be observable.** Put remaining budget in the observation vector of **every**
  mode, images included. Otherwise the agent not knowing its own budget is a *second*,
  unrelated partial observability, and it confounds exactly the comparison being made.
- **It must not silently re-price the arms.** See §2.3: budget *consumption* should be
  measured in commanded dose, not in volume-weighted `dose_spent`, or arm 1 spends its
  budget ~16x faster than arm 3 and the arms are no longer comparable.

### 4.3 IC geometry is where the effect gets its teeth

⭐ **Randomise tumour position relative to the dosed wall, per episode.** If the tumour is
always centred, the optimal policy is a fixed schedule and a scalar mode learns it — the
image is never *needed*. For the image to pay, the thing it sees must **vary per episode**.

The machinery exists: IC families `network_field` and `rectangle`, plus the boundary
IC-geometry grid (commit `4ca2431` in the thesis repo, `boundary_ic_driver.py` here).

This is the single highest-leverage design decision in the note. An otherwise correct
experiment with centred ICs will measure nothing.

### 4.4 Observation modes

No new modes. Use the existing families so the comparison inherits §6's framing:

- image: `I1`, `I1m`, `I2`, `I2m`
- spatial-scalar: `S3s`, `S3m`, `S3sm`, `S5m`
- blind baseline: `S1` (four scalars, no space) — **the mode that sat inside the collapsed
  band in §6.1**, so it is the reference for whether the gap reopens.

---

## 5. What would be proved, and what would not

### Proved if the prediction holds (graded 1 < 2 < 3)

Actions whose **parameters** are non-spatial but whose **consequences** are spatially
structured still require spatial observation. That widens the claim's third condition and
`ch7_discussion.tex:455` says so in advance — which matters, because it means the result
was predicted as a possibility before it was measured, not rationalised after.

### Proved if the gap stays closed in all three arms

The condition means what it literally says: spatial *parameters* in the action, not merely
spatial consequences. `ch7_discussion.tex:456-458` calls this "a stronger reading than the
evidence currently supports". A null here is a real result and must be written up as one.

### NOT proved either way

- Anything about the **level** difference between arms. See §2.3 — cross-arm return
  comparisons are confounded by the volume-weighted charge. Only within-arm, across-mode
  comparisons are clean.
- Anything about the §6.1 collapse itself. This experiment **leaves §6.1 intact** rather
  than replacing it, which `ch7_discussion.tex:461-465` names as one of two reasons to
  prefer it over richer action spaces.
- Anything about real drug delivery. The boundary influx is "as if from the circulation"
  by analogy only; a 64 um domain with a clamped wall is not a vasculature model.

---

## 6. Execution order

Staged so that each step can kill the experiment cheaply before the expensive one.

1. **Calibrate the budget** (laptop, ~1 h). Sweep budget on arm 1 only, one observation
   mode, and find the range where the greedy policy is forced to stop early but outcomes are
   not floored. **Gate: if no such range exists, the experiment is dead** and that is worth
   knowing for one hour of compute.
2. **Verify the harness supports it** (laptop, ~2 h). ⚠️ **Unverified as of writing:**
   whether `boundary_run.py` supports (a) an episode dose budget and (b) per-episode IC
   randomisation relative to the dosed wall. Both are assumed by this design; neither has
   been checked. Do this before anything else that costs compute.
3. **Field sanity check** (laptop, minutes). Re-confirm the one-sided profile at D=30 with
   cell rules actually loaded — see `cell-rules-load-silently-empty`. Use the
   `which boundaries?` print (`BioFVM_microenvironment.cpp:1497`) as ground truth: `1 0 0 0 0 0`
   = xmin only.
4. **Single-mode pilot on arm 3** (GPU, ~hours). One image mode vs `S1`, 3 seeds. If the
   strongest predicted contrast shows nothing, stop.
5. **Full sweep** (GPU, days). Three arms x nine modes x five seeds.

⚠️ **Steps 4-5 need the workstations, and `NOW.md`'s one starred irreversible item —
backing up the 7,783 unbacked per-episode traces on `sureli9`/`sureli11` — must be done
first.** Do not launch training onto machines whose only copy of the §6.4/§6.6 evidence is
still unbacked.

---

## 7. Scheduling verdict

**Not before deposit.** 30 Sep 2026 is ~17 days out; this is a new multi-day training sweep
plus harness work, and the machines are under a backup-only decision (`NOW.md`, taken
2026-09-03).

**Where it goes instead:** it is already half-written as future work in
`ch7_discussion.tex:435-465`. The cheapest possible gain is to add the boundary-influx
anisotropy knob (§3b, §4.1) to that existing paragraph — it converts "uniform dose +
budget" into a *graded* three-arm design with a controllable knob, and the field numbers in
§2.1 are already measured and citable. That is a paragraph edit, not an experiment.

**Post-thesis:** this is a good first paper-sized experiment. It needs no new action
machinery, no new observation modes, and no simulator change.

---

## 8. Provenance

- §6.1 collapse, spreads 6.9 / 73.9 / 2.0 / 0.005 / 0.0123 — `chapters/ch6_results.tex:71-104`
- the claim and its three conditions — `chapters/ch7_discussion.tex:5-40` (`sec:discussion:scope`)
- the budget experiment as already-named future work — `chapters/ch7_discussion.tex:435-465`
- boundary influx field measurements — memory `boundary-influx-works-and-is-measured`
- D vs domain size — memory `domain-is-64um-so-any-drug-diffusion-homogenises`
- dose penalty 15.9x, episode leak — memory `dose-penalty-is-volume-weighted`
- 8x dose efficiency, 15.75x spend — memory `aimed-disc-is-8x-more-dose-efficient`
- zero-cell-rules trap — memory `cell-rules-load-silently-empty`
