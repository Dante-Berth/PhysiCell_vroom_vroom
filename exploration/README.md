# Untreated (dose-0) parameter analysis

Exploratory study of the TME model with **no drug at all**: how the populations
evolve on their own, how that depends on the parameters that were chosen rather
than measured, and what each observation mode encodes of the same trajectory.

This is the measurement half of the thesis's `sec:tme:sensitivity` gap (Task T5).

## Why the drug is off

With `dose = 0` the drug never enters the microenvironment, so the trajectory is
decided entirely by the initial condition and the cell rules. That isolates the
question "does immunosuppression establish itself on its own?" from any question
about control.

Three independent probes assert this on every recorded step, and all must be
exactly zero:

| probe | what it proves |
|---|---|
| `drug_1_dose_param` | what we sent |
| `drug_1_amount_used` | what `add_local_substrate` consumed |
| `submax__drug_1` | max over every voxel of the `drug_1` field |

`add_local_substrate` returns early when dose <= 0 **or** radius <= 0, and both
are pinned to 0 in our XML copy, so zero drug is guaranteed twice over.

## Why one simulation covers every state space

With dose 0 the observation mode does not affect the simulation. It only changes
what gets read out. So running the same untreated trajectory under twenty
observation modes would produce twenty identical trajectories.

Instead each episode runs **once** and every feature builder is called directly
on the same instant. All twelve primitives are recorded, and every published
observation mode is a concatenation of them (the map is in `manifest.json`
under `composites`), so any state space can be reassembled offline without
re-running anything.

## The import problem this works around

There is no `.venv` in this tree. Every venv on this machine resolves
`extending.physicell` to a `.so` built from `~/Documents/Git/PhysiCell`, a
different model whose `custom.cpp` has **no `add_local_substrate` at all**, and
resolves `physigym` to that tree as well. Running against those would silently
simulate the wrong model.

`bootstrap.py` prepends this tree's own build directory and physigym package and
then asserts, by mangled C++ symbol, that the right ones won. Import it first,
before anything else.

⚠️ The correct `.so` lives under `custom_modules/extending/build/`, which is
git-ignored. A `make clean` or a fresh clone destroys it and the bootstrap will
fail loudly with the rebuild command rather than silently using the wrong one.

## The grid

Three axes, kept separate so tumour burden and immune pressure are not
confounded the way a single "scale everything" axis would confound them.

| arm | varies | fixed |
|---|---|---|
| `burden` | 4 IC families x tumour {32, 64, 128, 192} | immune 32/32 |
| `immune` | macrophage/T cell {8/8, 16/16, 64/64} x {network_field, rectangle} | tumour 128 |
| `ablation` | remove T cells, macrophages, or both | network_field, tumour 128 |

6 seeds each. The paper's point is tumour 128 / macrophage 32 / T cell 32 on
`network_field` (training) and `rectangle` (held out).

Requested counts are an **upper bound**: `init_conds.py` rounds to the integer
grid and drops duplicates, so realised counts are recorded per episode and the
analysis is keyed on those.

## Study 2: the parameter screening

Morris elementary effects over the parameters that were chosen rather than measured:
four in the settings XML, six in `cell_rules.csv`, plus one deliberate negative
control. Sampling and indices are SALib's, the same library the PhysiCell community's
UQ-PhysiCell wraps.

The **same design matrix** is evaluated under every initial condition (both families,
three seeds), so a difference in the indices is attributable to the initial condition
rather than to resampling. That makes rank stability measurable.

Morris rather than Sobol because the thesis scopes this as a screening and says so:
`sec:tme:sensitivity` states it is not a full global sensitivity analysis.

⚠️ **`growth_rate_reward_only` is a negative control, not an oversight.**
`//user_parameters/growth_rate` is read only at `physicell_model.py:112`, to build
`lambda_dt` for the **reward**. No C++ code reads it, so it cannot move the untreated
dynamics, and a working screening must return mu* = 0 for it. The real proliferation
driver is `tumor_cycle_rate`, the tumour's cycle phase transition rate. An earlier run
swept only the reward parameter and its exact zero is what exposed the mistake.

⚠️ **Read mu\* as screening, not as a strength ordering.** It is not normalised for
range width, and the ranges here differ: `cytokine_diffusion` spans four orders of
magnitude while most others move by a factor of four to sixteen.

## Running it

```bash
PY=/home/disc/a.bertin/Documents/Git/.venv/bin/python
cd /home/disc/a.bertin/Documents/Git/PhysiCell_vroom_vroom/exploration

$PY smoke.py 40                 # timing + dose-zero check, ~1 min
$PY driver.py --run-tag dose0_v1 --workers 12 --seeds 6 --max-steps 480
$PY plots.py "$PWD/output/dose0_v1"

$PY sa_driver.py --run-tag sa_v4 --workers 12 --trajectories 12 --horizon 200 \
    --ic-families network_field,rectangle --ic-seeds 0,1,2
$PY sa_driver.py --analyse-only "$PWD/output/sa_v4"     # re-analyse without re-running
```

`plots.py` and `--analyse-only` need an **absolute** path: `bootstrap` chdirs to the
project root, so a relative one resolves against the wrong directory.

Measured cost: **103 ms** per simulation step plus **50 ms** for all twelve
encoders. Episodes self-terminate on escape well before the 480-step cap, so the
full 7200-minute horizon is affordable and shortening it would save little.

## Layout

```
bootstrap.py   import first: sys.path, cwd, libgomp, and the loud assertions
common.py      XML preparation, encoder registry, ground-truth measurements
smoke.py       tier-0: per-step cost, encoder dims, dose-zero probes
run_one.py     one worker process = one parameter point, several seeds
driver.py      grid fan-out, concatenation, manifest
plots.py       figures
sa_params.py   the parameters swept, and how each is written into XML or rules
sa_run.py      one process = one Morris sample (see the singleton note below)
sa_driver.py   Morris sampling, fan-out, per-IC indices and rank stability
probe_polarisation.py  why every macrophage reads M2 from step 1
output/        git-ignored (.gitignore:55 "output*/")
```

## Traps already paid for

- **`time_simulation` is never reset.** `physicell_core` sets it in `__init__`
  and mutates it in `get_truncated`, but `reset()` leaves it. After a short
  episode the next one truncates spuriously at step 1. Every reset here is
  followed by `env.unwrapped.time_simulation = -1`.
- **The wrapper is not used.** `PhysiCellModelWrapper.reset()` deletes files in
  its output directory, rewrites the settings XML on disk, and anchors its
  action-delta clipping at dose 0.5. We drive the raw env with `action_mode="full"`,
  whose action space is the single key `drug_1_dose`.
- **`render_mode` must stay `None`.** `get_img()` queries substrate names
  (`debris`, `pro-tumoral factor`) that no longer exist in the XML.
- **`get_microenv(name)` returns `(N,4)` rows** of x, y, z, concentration, not a
  2-D field. There is no `get_microenvironment`.
- **`relational` is 62-dimensional**, not the 48 its docstring claims.
- **One env per interpreter.** The compiled module is a per-process singleton,
  so the driver runs one OS process per parameter point.
- The tracked `config/PhysiCell_settings.xml` is never mutated; each worker
  copies it to `output/<tag>/cfg/`.
