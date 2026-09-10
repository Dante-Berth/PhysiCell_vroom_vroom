# Where this tree runs, and where it does not

_Written 2026-09-10._

**This checkout is on the laptop.** `port-bertin-l`, at
`~/Documents/Git/PhysiCell_vroom_vroom`.

⚠️ **It is not the same thing as the `PhysiCell_vroom_vroom` on `sureli9` / `sureli11`.**
The directory name is identical on every one of them and the contents are not. Confusing
them has already cost three weeks once (looking for per-episode traces on one host when
half of them were on the other).

| machine | what its copy holds | path |
|---|---|---|
| **`port-bertin-l`** (this one) | the **model + `exploration/`** analysis harness | `~/Documents/Git/PhysiCell_vroom_vroom` |
| `sureli11` | training runs, **image** observation modes | `~/PhysiCell_vroom_vroom/data` |
| `sureli9` | training runs, **spatial-scalar** observation modes | `~/Physi/PhysiCell_vroom_vroom/data` |
| `sureli1` | the whole `net2rect` **uniform-action** sweep (52 runs) | -- |
| `sureli3` | a staging clone, edited `run.py`, **no runs on wandb** | -- |

So five machines carry a tree of this name. The thesis repo's `MACHINES.md` is the
authority on which produced which wandb project; do not re-derive it here.

The per-episode traces on the two `sureli` hosts are the sole surviving record behind
two written sections of the thesis (§6.4 and §6.6) and are backed up nowhere. Searching
one host finds half the data and looks exactly like absence.

## What lives here: `exploration/`

The analysis harness for everything that is **not** training: the dose-zero study, the
dose ladder, the Morris parameter screening and the drug-diffusion sweep. See
`exploration/README.md` for the layout and for the traps already paid for.

⚠️ **`exploration/output/` is git-ignored** (`.gitignore:55`, `output*/`). Every result
that harness has ever produced is laptop-local and unbacked. The scripts are tracked; the
numbers are not.

## The simulator does not need a GPU

Measured at **~103 ms per simulation step** plus ~50 ms for all twelve observation
encoders. A full exploration study is minutes to an hour on this laptop.

The GPU on the `sureli` machines is only ever needed for **training** (the RL agent), never
for **simulation**. So exploration and analysis work belongs here, on the laptop.

⚠️ Note also that the CUDA path never ran in any reported experiment.
`BioFVM_microenvironment.cpp:1395` selects the GPU solver only when `simulate_2D == false`
**and** `gpu_backend_is_cuda()`, then only above a 1,000,000-voxel threshold. This model is
2D on a 64x64 grid (4,096 voxels) and fails on the first test alone.

## ⚠️ Standing instruction: launch nothing on the `sureli` machines

Set by the user on 2026-08-31 and still binding. The sweeps are finished, and what remains
for the thesis is writing and analysis of runs already in hand. Read-only analysis of the
stored traces is fine. Training, re-runs and new sweeps are not to be started. If something
seems to need a run, say so and let the user decide rather than launching it.

Also: `nvidia-smi` alone does not tell you whether those machines are busy. These runs are
CPU-bound, so a job can pin a core at 99% while the GPU reads 0% and 4 MiB. Check
`pgrep -a -u $USER -f "envs/run.py"` as well.

## Two things that break silently on this machine

- ⚠️ **The compiled `.so` is git-ignored.** It lives at
  `custom_modules/extending/build/lib.linux-x86_64-cpython-310/` and a `make clean` or a
  fresh clone destroys it (`Makefile:513` does `rm -f project*`). `exploration/bootstrap.py`
  asserts by mangled C++ symbol that the right one loaded, and fails loudly with the
  rebuild command rather than silently simulating the wrong model.
- ⚠️ **Every venv on this machine resolves `extending.physicell` to a different tree**
  (`~/Documents/Git/PhysiCell`), whose `custom.cpp` has no `add_local_substrate` at all.
  Import `exploration/bootstrap.py` **first**, before numpy or gymnasium, in anything that
  drives the model.

## The tracked settings XML is runtime residue in places

`config/PhysiCell_settings.xml` is the template, but three of its values are leftovers from
whatever ran last and must not be read as configuration:

- `//save/folder` points into a `data/random_baseline_.../run_000055` directory.
- `//initial_conditions/cell_positions` points at `ic_000055.csv`, **which does not exist**.
  A run that inherits it starts with **zero cells**, silently, because
  `//user_parameters/number_of_cells` is 0 and nothing scatters any.
- `//save/SVG/enable` and `//save/full_data/enable` are both `false`, so a standalone run
  from this file writes no output at all.

`exploration/common.py:prepare_xml` exists to deal with this: it copies the tracked file,
pins everything the harness controls, and never mutates the original. `point_xml_at_ic`
then sets a real initial-condition path and seed.
