# PhysiGym end-to-end random-policy benchmark: optimized vs baseline BioFVM

Date: 2026-07-02. Measures the **full PhysiGym step path** (Python `env.step` →
C++ `physicellmodule`: BioFVM diffusion + PhysiCell cell update + observation),
not just the isolated diffusion loop that `make benchmark` / `bench3` time.

## Setup

- Config: `config/PhysiCell_settings_bench_random.xml` — a copy of
  `config/PhysiCell_settings.xml` with `max_time=7200` min (→ 480 gym steps of
  `dt_gym=15` min = one full episode), `omp_num_threads=8`, full_data + SVG
  output **disabled** (removes I/O noise).
- Driver: minimal gymnasium loop, `action_space.sample()` each step, cheapest
  observation mode (`scalars_cells`) to maximize the BioFVM/core signal. No
  torch / wandb / vectorization (unlike `run.py`), so timing is the sim, not the
  RL machinery. (driver kept in the session scratchpad: `bench_random.py`)
- Machine: 32 cores, 8 OpenMP threads.

## How the two builds were produced

PhysiGym compiles BioFVM into the `extending.physicell` C++ extension via
`custom_modules/extending/setup.py` (untracked). Swapping whole BioFVM trees is
an **ABI trap**: `core/*.h` hardcode `#include "../BioFVM/BioFVM_microenvironment.h"`,
so `BioFVM copy/`'s sources would mismatch the header the core objects see and
crash. Same reason `make benchmark` avoids it.

Instead — identical methodology to `make benchmark` — the baseline swaps only
the two hot sources for their original-algorithm `*_baseline.cpp` variants
(same current header/ABI):
- `BioFVM_solvers.cpp`      → `BioFVM_solvers_baseline.cpp`
- `BioFVM_basic_agent.cpp`  → `BioFVM_basic_agent_baseline.cpp`

`setup.py` was made to honor `PHYSIGYM_BIOFVM_BASELINE=1` for this swap:

```
PHYSIGYM_BIOFVM_BASELINE=1 python setup.py build_ext --inplace   # baseline
python setup.py build_ext --inplace                               # optimized
```

The built `.so` was copied over
`…/.venv/lib/python3.11/site-packages/extending/physicell.…so` for each run.
The optimized `.so` was restored afterward (verified).

## Results (480 steps = 1 full 7200-min episode)

| Build                | run 1   | run 2    | per step  |
|----------------------|---------|----------|-----------|
| optimized BioFVM/    | 24.29 s | 25.40 s  | ~51–53 ms |
| baseline (original)  | 102.97 s| 104.30 s | ~215–217 ms |

**End-to-end speedup ≈ 4.1x** (25.3s vs 103.6s median) for a full random-policy
episode through PhysiGym.

## Correctness: same result, not just faster

Ran one deterministic episode (`omp_num_threads=1`, config seed 42) on each build
and compared final **live cell counts** (`physicell.get_cell()`, `dead<0.1`):

| Build      | steps run | total alive | tumor | t_cell | macrophage |
|------------|-----------|-------------|-------|--------|------------|
| optimized  | 156       | 323         | 259   | 48     | 16         |
| baseline   | 156       | 323         | 259   | 48     | 16         |

**Identical** — same terminal tumor count (259), same total, same episode length
(both terminate at step 156). So the ~4x speedup changes nothing about the
biology. (This matches `verify-discrepancy.md`: the CPU BioFVM is bit-equivalent
to baseline within 5e-14 at 1 thread.)

Caveat worth recording: `env.action_space.sample()` uses an **unseeded** RNG —
`reset(seed=42)` seeds PhysiCell's `random_seed` but not the Gymnasium action
sampler. So `action_sum` varies run-to-run *within the same build* (81.3 vs 74.2
on two optimized runs). Yet the cell counts are invariant, which shows that for
this config the terminal tumor count is governed by the PhysiCell seed, not by
the random drug schedule. If you need a fully controlled A/B on actions, seed the
action space explicitly (`env.action_space.seed(...)`).

## Controlled max-drug variant (no RNG)

Re-ran with a **fixed** action instead of random: `action_mode="full"` applies
maximum dose (`drug_1_dose=1.0`), centered on the domain, with the full-domain
radius `sqrt((w/2)²+(h/2)²)` — exactly what `PhysiCellModelWrapper`'s `full`
branch does. No RNG at all, so it's a clean A/B.

Correctness (omp=1, seed 42) — **identical** again:

| Build     | steps | tumor | t_cell | macrophage | total |
|-----------|-------|-------|--------|------------|-------|
| optimized | 156   | 259   | 48     | 16         | 323   |
| baseline  | 156   | 259   | 48     | 16         | 323   |

Speed — compare **per-step** (not total wall): under omp=8 the episode
terminates a few steps early at slightly different points because multi-thread
secretion is non-deterministic, so total wall is noisy but per-step is stable:

| Build     | omp=1 per-step | omp=8 per-step |
|-----------|----------------|----------------|
| optimized | 111 ms         | ~48.6 ms       |
| baseline  | 180 ms         | ~213 ms        |
| speedup   | 1.6x           | ~4.4x          |

## Videos (visual dynamics check)

Generated `video.mp4` for one full max-drug episode per build to eyeball that
the spatial dynamics match. Method: build the wrapper with
`frequence_episode_test=1` (every episode is a "test" → frames captured),
`action_mode="full"`, `mode_test=["network_field"]` (single geometry so both
builds start identically), constant max action each step. The video is compiled
by `save_data()` on the *next* reset (ffmpeg + cv2). Outputs saved to
`bench_videos/{optimized,baseline}_maxdrug{,_omp1}.mp4`.

**Trajectory comparison (omp=1, byte-identical starting IC — verified 62 tumor /
7 t_cell / 30 macrophage, same `ic_000001.csv`):**

- `n_tcell`, `n_macrophage`: **bit-identical every step** (max|diff| = 0).
- `number_tumor`: identical for the first **236 steps**, then a slow bounded
  drift (max|diff| = 24 of ~200 cells, mean ≈ 5 over 376 steps).

This drift is *expected and not a BioFVM regression*: the CPU-optimized fields
match baseline only to ~5e-14 (see `verify-discrepancy.md`), and PhysiCell tumor
**division timing** is stochastic, so on an exponentially growing tumor those
femto-scale field differences get amplified through cell-division events after a
few hundred steps — classic sensitive-dependence, not systematic divergence. The
tighter `bench_maxdrug.py` correctness runs (smaller `config/cells.csv` start,
terminating at step 156) matched *exactly* (259 tumor) because they end before
this amplification sets in.

## Reading the number

This is *larger* than the ~1.63x reported for the isolated diffusion loop
(`verify-discrepancy.md`) because the end-to-end path also benefits from the
optimized 3-step→fused secretion in `BioFVM_basic_agent`, and because this
config (10080→7200 min, `dt_diffusion=0.01`) is diffusion-heavy: 1500 diffusion
solves per gym step dominate the wall clock, so the BioFVM speedup shows through
strongly rather than being diluted by Python/observation overhead. (Contrast the
1.04x in `physigym-gpu-fit.md`, which used an image-observation mode where the
per-step obs download dominated.)
