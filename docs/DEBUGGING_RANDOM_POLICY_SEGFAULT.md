# Debugging the random-policy segfault

The random policy (`run.py --mode random`) intermittently **segfaults inside the
compiled PhysiCell `physicell.step()`**, killing a couple of worker
subprocesses per run. `ResilientSubprocVecEnv` disables the dead envs and the
run continues, so it is survivable — but the underlying C++ crash is real.

This doc records **what we know, how it was tracked, and how to finish the
diagnosis**. It is the map for the next person (or session).

---

## RESOLUTION (2026-07-07)

**Why cells go out of the domain at all** — this is *normal* PhysiCell
behavior, not corruption. Cells are point agents whose positions are
integrated by the mechanics ODE (`Cell::update_position`,
`core/PhysiCell_cell.cpp`), and division places daughters at
`parent ± radius·rand_vec`. Either can land a cell outside the box, at which
point it is flagged `is_out_of_domain = true` and its
`current_mechanics_voxel_index` is set to `-1`. `virtual_wall_at_domain_edge`
is only a **soft repulsive force**, not a hard clamp, so it can be overpowered
in one timestep — especially in the thin `63×63×1.0` (dz=1) 2D slab, and
especially under random RL actions that drive cells into the walls far more
often than a scripted sim.

So out-of-domain cells are a recurring population. The crash was **any code
path that indexes a voxel/agent structure with that `-1` without guarding**.

### Root cause of the `step()` crash — FIXED ✅ (verified)

In `Cell_Container::update_all_cells` (`core/PhysiCell_cell_container.cpp`),
every per-cell loop guards on `is_out_of_domain == false` **except two** that
were added later:

- the `dynamic_spring_attachments` loop (March 2023)
- the `standard_cell_cell_interactions` loop (March 2022)

Both call `find_nearby_cells` / `find_nearby_interacting_cells`
(`core/PhysiCell_cell.cpp`), which do
`agent_grid[pCell->get_current_mechanics_voxel_index()]`. For an out-of-domain
cell that index is `-1` → out-of-bounds `std::vector` access → segfault. This
matches the forensics exactly (`n_out_of_domain ≥ 1` on every captured crash).

**Fix applied:**
1. Added `... && pC->is_out_of_domain == false` to both loops in
   `core/PhysiCell_cell_container.cpp` (matching the other loops there).
2. Belt-and-suspenders: added `if (get_current_mechanics_voxel_index() < 0)
   return neighbors;` at the top of **both** `find_nearby_cells` and
   `find_nearby_interacting_cells` in `core/PhysiCell_cell.cpp`, so no future
   caller can re-trip it.

**Verification:** after rebuilding the extension `.so`, a 5000-step
`run.py --mode random` completed **12 clean episodes** (`ep_len=480`) with **no
`step()` crash**.

### A second, separate crash in the reset path — OPEN ⚠️

The same run still segfaulted once, but the faulthandler trace shows it is
**not** in `step()` — it is in `physicell.start(settingxml, reload=True)` at
`physicell_core.py:487`, the **between-episode reinitialization**
(`physicellmodule.cpp:130–176`: clears cells via
`while(!(*all_cells).empty()) back()->die()`, resets mesh/microenvironment,
creates a *fresh* cell container, re-runs `setup_tissue()`). This is a distinct
bug from the step() one and is still being pinned (see Tool 2 — the `-O0`
import blocker is now solved, build with `-O2 -g`).

### Rebuild recipe used

```bash
cd custom_modules/extending
rm -f ../../core/*.o *.o
../../.venv/bin/python setup.py build_ext --inplace -j4   # release
# inplace copy step errors on a missing local extending/ dir — harmless; copy manually:
cp build/lib.linux-x86_64-cpython-311/extending/physicell*.so \
   ../../.venv/lib/python3.11/site-packages/extending/
```

---

## TL;DR — what we established

- The crash is in `physicell.step()` (C++), **not** in the observation code and
  **not** in the drug injection. Observation mode never touches the sim step, so
  "it only crashes for state space X" is a sampling illusion — different runs use
  different seeds / `num_envs`, and random actions reach bad states far more
  often than a trained policy.
- **Trigger, confirmed 4/4:** every captured crash had **out-of-domain cells
  present** (`n_out_of_domain ≥ 1`) when the step ran. Zero crashes occurred with
  `n_out_of_domain == 0`. The triggering *actions* were unremarkable (mid-range
  dose, in-domain injection center, normal radius).
- It is the **same bug family** as the already-fixed
  `release_internalized_substrates` voxel(-1) crash, but a **different**
  unguarded code path that runs during the normal mechanics/diffusion step
  (not the cell-death path).
- **PINNED & FIXED** — see the RESOLUTION section above. The `step()` crash was
  two unguarded loops in `update_all_cells` (`dynamic_spring_attachments` +
  `standard_cell_cell_interactions`) reaching `agent_grid[-1]` via
  `find_nearby_cells`. A second, separate crash in the `start(reload=True)`
  reset path remains open.

---

## Tool 1 — Python-side crash forensics (already wired in)

`wrapper.py::step()` has an **opt-in** forensics block, gated on the
`PHYSIGYM_CRASH_LOG` env var. When set to a directory, it dumps — and
`fsync`s — the pre-step action + sim state to `laststep_pid<PID>.txt`
**before** calling `self.env.step()`, then truncates it on success.

A segfault leaves no Python traceback, so the last non-empty `laststep_pid*.txt`
is the action/state that killed each worker.

### How to run it

```bash
export PHYSIGYM_CRASH_LOG=/tmp/crashlog
mkdir -p "$PHYSIGYM_CRASH_LOG"
PHYSIGYM_QUIET=1 PYTHONPATH="custom_modules/physigym/physigym/envs:custom_modules:." \
  .venv/bin/python custom_modules/physigym/physigym/envs/run.py \
  --mode random --seed 1 --observation_mode spatial_scalars_cells \
  --action_mode targeted --action_repeat 6 --total_timesteps 5000 --wandb false

# after it finishes, read the culprits (non-empty markers):
for f in "$PHYSIGYM_CRASH_LOG"/laststep_pid*.txt; do
  [ -s "$f" ] && { echo "--- $f ---"; cat "$f"; }
done
```

Each non-empty file looks like:

```
episode=1 step=251
n_cells=242 n_out_of_domain=2         <-- the smoking gun
raw_action(dose,x,y,r)=[0.58, 0.64, 0.33, 0.52]
applied drug_1_x=40.6 drug_1_y=20.8 drug_1_radius=5.67 drug_1_dose=0.58
domain x[0,63] y[0,63]
```

**What to look at:** `n_out_of_domain`. If it is ≥1 on the crashing steps, the
out-of-domain hypothesis holds.

---

## Tool 2 — gdb backtrace for the exact C++ line (the unfinished part)

Python's faulthandler only shows the C boundary (`physicell.step()` at
`physicell_core.py:692`). To get the exact C++ line you need a **debug build +
gdb**, driven **in-process** (SubprocVecEnv workers are separate processes gdb
won't follow, so use a single in-process env).

### 2a. Build a debug `.so`

`custom_modules/extending/setup.py` honors `PHYSIGYM_DEBUG_BUILD=1` (adds `-g`,
drops `-fomit-frame-pointer`, uses `-O0`). Build **inplace** — the pip editable
build fails on a relative-path (`../../BioFVM`) permission issue, so avoid it:

```bash
cd custom_modules/extending
# back up the working release .so first!
VENV=../../.venv/lib/python3.11/site-packages/extending/physicell.cpython-311-x86_64-linux-gnu.so
cp "$VENV" "$VENV.release_bak"

PHYSIGYM_DEBUG_BUILD=1 ../../.venv/bin/python setup.py build_ext --inplace -j4
cp build/lib.linux-x86_64-cpython-311/extending/physicell*.so "$VENV"
```

> **KNOWN ISSUE (blocker):** the `-O0` debug `.so` built this way **crashed at
> import** (`PyInit_physicell` → `PyModule_Create` → `PyErr_Format` → `strlen`),
> before the sim loop — an ABI/flag-mix artifact, unrelated to the real bug.
> **Next step to try:** build with `-O2 -g` instead of `-O0` (keeps ABI closer
> to the release build), or drop `-march=native`. Only proceed to 2b once
> `from extending import physicell` imports cleanly with the debug `.so`.

### 2b. Drive it under gdb, in-process, single-threaded

A standalone driver that builds ONE wrapped env (no subprocess) and loops random
actions until it faults lives at
`scripts/debug/segfault_driver.py`.
Key settings: `num_envs=1`, `rl_threads = cpu_count-1` (→ `threads_per_env=1`),
`OMP_NUM_THREADS=1`.

```bash
PHYSIGYM_QUIET=1 OMP_NUM_THREADS=1 \
PYTHONPATH="custom_modules/physigym/physigym/envs:custom_modules:." \
gdb -batch \
  -ex 'set pagination off' -ex 'handle SIGSEGV stop print' \
  -ex run -ex 'bt' -ex 'info locals' -ex quit \
  --args .venv/bin/python scripts/debug/segfault_driver.py
```

Read frame `#0..#3` of the backtrace: the first frame inside PhysiCell/BioFVM
source (not libpython, not libc) is the crash site. Expect an unguarded
`voxels[idx]` / `microenvironment(idx)` / `agent_grid[idx]` access where `idx`
came from an out-of-domain cell (`current_mechanics_voxel_index == -1`).

### 2c. Restore the release build when done

```bash
cp "$VENV.release_bak" "$VENV" && rm "$VENV.release_bak"
```

---

## The fix, once the line is known

Guard that specific access with the same pattern already used elsewhere in
`core/PhysiCell_cell_container.cpp` (`... && pC->is_out_of_domain == false`) and
in `BioFVM_basic_agent.cpp::release_internalized_substrates`
(`if (current_voxel_index < 0 || >= number_of_voxels()) return;`).

> **Note — why a pure-Python cull does NOT work:** the `physicell` module exposes
> only `start/step/stop`, parameters, variables, vectors, `get_cell`,
> `get_microenv` — **no cell-removal API**. So Python can read `df_alive`
> positions but cannot delete out-of-domain cells from the sim. The real cull /
> guard must be in C++ and requires a rebuild.

---

## Related

- `core/PhysiCell_cell_container.cpp` — most per-cell loops already guard on
  `is_out_of_domain == false`; the crashing path is one that does not.
- `BioFVM/BioFVM_basic_agent.cpp::release_internalized_substrates` — the
  previously-fixed sibling bug (voxel(-1) on death).
- `custom_modules/physigym/physigym/envs/resilient_sub_vec_env.py` — `close()`
  override that lets runs finish cleanly despite the crash.
