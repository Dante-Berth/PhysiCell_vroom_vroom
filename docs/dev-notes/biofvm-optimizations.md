# BioFVM CPU optimizations


Applied the following optimizations to BioFVM/ (2026-05-22):

**BioFVM_vector.cpp**
- `axpy`, `naxpy`: added `#pragma omp simd` + `__restrict__` pointers via `.data()`
- `operator+=/-=/*=/=/=` (all in-place variants): same SIMD treatment
- `operator/=(scalar)`: replaced division with `1/a` reciprocal multiply
- `norm_squared`: added `#pragma omp simd reduction(+:out)`

**BioFVM_microenvironment.cpp**
- Added SoA (Structure-of-Arrays) flat buffers: `soa_density1/2`, `soa_p`, `soa_old`
- `pack_to_soa()`: AoS→SoA transpose before diffusion solver
- `unpack_from_soa()`: SoA→AoS transpose after diffusion solver
- `compute_all_gradient_vectors`: reads from `soa_p` directly (stride-1 per substrate)
- `compute_gradient_vector` (lazy per-voxel): updated to read `soa_p` directly
- Added `soa_is_authoritative` flag, `sync_to_aos_if_needed()` (public), `get_soa_p()` (public accessor)

**BioFVM_solvers.cpp** (LOD_2D and LOD_3D, the hot paths)
- **Opt 1 (SoA layout)**: Replaced AoS pointer indirection with flat `soa_p[substrate*nv+voxel]` access
  - Pre-built flat arrays: `denom_x`, `cx_flat`, `denom_y`, `cy_flat` (stride-1 per substrate)
  - Single `#pragma omp parallel for collapse(2)` per sweep direction (row-outer)
  - Substrate loop INSIDE the parallel region (no fork/join per substrate)
- **Opt 2 (compact index lists)**: Separated D>0 and D=0 substrates
  - `diff_subs2[]`/`nodiff_subs2[]` — for D=0: pure scalar decay `soa[s*nv+v] *= decay_factor2[s]`
  - `decay_factor2[s] = 1/(1+c2)^2` for LOD_2D (2 passes), `1/(1+c2)^3` for LOD_3D (3 passes)
  - Skips all Thomas forward/back substitution for zero-diffusion substrates

**TIMING (omp=1, 1440min sim, 64x64 domain, 5 substrates, ~300 cells):**
- Baseline (original AoS solver, O2): ~2m45s
- Opt 1+2 (SoA + compact indices, O3): ~85s (omp=1), ~52s (omp=2)
- Opt 2 alone improved: from ~95s (buggy substrate-outer loop) to ~85s (correct row-outer)

**BENCHMARK INFRASTRUCTURE:**
- `BioFVM/BioFVM_solvers_baseline.cpp`: original AoS solver compiled with current header (avoids ABI mismatch)
- `BioFVM/BioFVM_basic_agent_baseline.cpp`: original 3-step secretion compiled with current header
- `config/PhysiCell_settings_benchmark.xml`: max_time=1440
- `config/PhysiCell_settings_benchmark_omp2.xml`: omp=2 variant
- `main_timed.cpp`: diffusion vs cell-update timing split
- Makefile targets: `classic`, `project_opt`, `project_orig`, `benchmark`

**IMPORTANT - STALE OBJECT FILES:**
- When `BioFVM_microenvironment.h` changes, ALL `.o` files that include it must be recompiled
- The Makefile does NOT have header dependencies — run `rm -f *.o && make classic` after any header change
- This caused a "double free or corruption" crash that was misdiagnosed as a code bug

**OPT 3 (ABANDONED):**
- Goal: skip `unpack_from_soa()` after diffusion to save one AoS←SoA transpose per step
- Problem: cell agents read substrate concentrations via `(*pS)(voxel_index)` (AoS path)
  - Skipping unpack means cells see stale pre-diffusion concentrations → wrong biology (divergent cell counts)
- Full Opt 3 requires converting ALL cell signal reads to use `soa_p` directly — large scope change affecting PhysiCell core
- Current state: pack+unpack kept as before; gradient and file I/O already read soa_p natively

**CRITICAL BUG FOUND + FIXED (2026-06-05):**
- The fused secretion loop (`simulate_secretion_and_uptake`) AND `release_internalized_substrates`
  were writing sources **into the SoA buffer** via `get_soa_p()` instead of the AoS density vector.
- `simulate_diffusion_decay` calls `pack_to_soa()` (AoS->SoA) EVERY step, so it overwrote the
  secretion in SoA with stale (zero) AoS values before the solver ran. Result: all D>0 substrates
  were silently **zeroed** for the whole sim. AoS is authoritative, NOT SoA.
- The earlier "cell counts diverge, expected, both valid" note was WRONG — that divergence (295 vs
  298) was this bug, not FP noise.
- FIX: both functions now write `(*pS)(current_voxel_index).data()` (AoS), keeping the fused SIMD
  loop. After fix, fields match baseline bit-for-bit (omp=1) and cell counts are identical (298 vs 298).

**VERIFICATION HARNESS (use this to check correctness):**
- `config/PhysiCell_settings_verify.xml` (max_time=60, omp=1), `/tmp/compare_me.py` loads both runs'
  `output*_microenvironment0.mat` via scipy and reports max|abs|/|rel| diff per substrate row.
  Rows 0-3 = x,y,z,vol; rows 4.. = substrates. PASS = max|rel| < ~1e-12.
- Build the two changed TUs explicitly if needed:
  `g++ -march=native -O3 -fopenmp -m64 -std=c++11 -c ./BioFVM/BioFVM_<tu>.cpp` then `make project_opt`.

**y-SWEEP TILING (BioFVM_solvers.cpp LOD_2D, 2026-06-05):**
- y-diffusion sweep is column-blocked (BW=8 adjacent x-columns) to fix stride-nx cache thrashing.
  Thomas recurrence stays sequential in j; all BW columns advance j in lockstep. Verified == untiled.
- Reciprocal-multiply (1/dy2) in the tiled sweep gives ~1.5e-13 rel diff vs baseline (FP reassoc, fine).

**LAZY TRANSPOSE ELIMINATION (§11, 2026-06-05) — SUPERSEDES the "secretion writes AoS" fix above:**
- Profiling: diffusion = 76% solve + 24% per-step AoS<->SoA transpose (pack 13% + unpack 11%), run ~every step.
- Made SoA authoritative across steps via dirty flags `aos_dirty`/`soa_dirty` (in BioFVM_microenvironment.h).
  Secretion + release_internalized now read/write SoA directly (`soa_p[s*nv+voxel]`); simulate_diffusion_decay
  packs only if aos_dirty, never eager-unpacks (sets soa_dirty). AoS accessors (operator()/density_vector/
  nearest_density_vector) + write_to_matlab call sync_aos_from_soa_if_dirty() before exposing AoS.
  In-solver apply_dirichlet_conditions writes RAW p_density_vectors/soa_p (NOT accessor) to avoid nested-parallel
  sync mid-solve.
- Result: pack 24000->112, unpack 24000->64 calls (~99.6% gone). Solo speedup 1.4x -> 1.6x vs pristine.
- BIG CORRECTNESS FINDING: lazy build matches PRISTINE upstream PhysiCell bit-for-bit for >1500 min
  (field ~1e-14), unlike the pre-§11 SoA build which drifted from pristine (296 vs 298, then -32 over long runs).
  Root cause of the OLD drift: gradients read SoA while secretion wrote AoS -> compute_all_gradient_vectors ran on
  PRE-secretion field (latent timing bug). §11 makes secretion write SoA so gradients see post-secretion = upstream.
  So the 296<->298 gap I earlier called "pre-existing microenv FP chaos" was partly THIS gradient-timing bug.
- Late drift (after ~t=2145min) lazy vs pristine is genuine chaotic FP (the y-sweep 1e-14 flips a cell decision);
  pristine itself is deterministic run-to-run (Δ=0), so it's a real but benign sub-ULP sensitivity, not a logic bug.

**HARDWARE: i7-13800H = 14 physical cores (6 P + 8 E), 20 logical. 9 SubprocVecEnv envs fit without oversubscription.**
**9-ENV THROUGHPUT (the real RL metric): ~27-37% gain (single-thread per-process speedup multiplies across envs).**
**Multi-thread scaling of ONE sim is poor (1.77x @ 4 threads = 44% eff) — irrelevant if each env stays omp=1.**

**VERIFIED TIMING (omp=1, vs pristine "BioFVM copy"): ~1.6x solo after §11 (was 1.4x). Correct vs upstream.**

**Why:** accelerating PhysiCell for RL training via PhysiGym / stable-baselines3 SubprocVecEnv (9 parallel envs)
