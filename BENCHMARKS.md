# BioFVM acceleration — benchmarking & correctness

This fork accelerates BioFVM (PhysiCell's diffusion solver). The guiding rule:
**an acceleration is only valid if it produces the same results as the reference.**
So every speed comparison has a matching correctness check.

## The reference vs. accelerated layout

| Tree | Role |
|------|------|
| `BioFVM copy/` | **Original / reference** BioFVM, untouched. The ground truth. *(Untracked on purpose — it's a local reference copy. Keep a pristine copy here to diff against.)* |
| `BioFVM/` | **Accelerated** BioFVM: CPU optimizations (SoA layout, flattened Thomas coeffs, OpenMP) **plus** the dual-backend GPU-resident solver. |

There are two independent acceleration axes, each with its own comparison:

### Axis 1 — CPU optimizations: original vs optimized
- Original build: `project_orig` (compiled from `BioFVM copy/`, -O2, original algorithm).
- Optimized build: `project_opt` (compiled from `BioFVM/`).
- The `*_baseline.cpp` files in `BioFVM/` are the original algorithm compiled against
  the *current* header (same ABI) — used so the two builds are link-compatible.

```
make benchmark        # wall-time: project_orig vs project_opt on one episode
make verify           # CORRECTNESS: same run, diff the output fields (see below)
```

### Axis 2 — GPU port: CPU solver vs CUDA solver
The GPU solver is dual-backend: real CUDA via `nvcc`, identical OpenMP fallback
otherwise (so it is correct and testable on a CPU-only machine). See
`BioFVM/CUDA_GPU_NOTES.md`.

```
make test-cuda              # unit: kernel vs independent reference Thomas solver
make test-cuda-integration  # engine: diffusion / diffusion+secretion / interleaved reads
make bench-cuda             # wall-time + transfer counts, CPU two-call vs GPU-resident
make test-cuda-gpu          # (needs nvcc) build & run kernels on a real device
make bench-cuda-gpu         # (needs nvcc) real-GPU wall time vs CPU
make test-gpu-autoselect[-gpu]  # size-based GPU solver selection (CPU/GPU build)
```

All Axis-2 correctness tests assert **bit-for-bit** (0.000e+00) agreement against
the CPU solver, except where BioFVM's own CPU secretion is nondeterministic (see
the multi-cell-per-voxel note in `BioFVM/CUDA_GPU_NOTES.md`).

The GPU only wins on large 3D grids (it loses to the CPU-opt solver on small ones,
where kernel-launch/transpose overhead dominates). `initialize_microenvironment`
therefore auto-selects the GPU solver only when the build has a CUDA backend AND
the grid is >= `BIOFVM_GPU_MIN_VOXELS` (default 1,000,000; env-overridable). A
normal CPU build never selects it. Real-GPU speedup (RTX 4090, vs 32-thread CPU):
~1.1x @256K voxels, ~1.6x @2M, ~2.0x @4M.

### Three-way micro-benchmark: reference vs CPU-opt vs GPU
`make bench3` (or `bench3-ref` / `bench3-cpu` / `bench3-gpu`) times JUST the
diffusion+secretion loop on an identical problem across all three solvers, so the
numbers are directly comparable. Pass `ARGS="NX NY NZ AGENTS STEPS"`. Source:
`benchmark_biofvm.cpp`. Sample (32-thread CPU, ms/step): 4M voxels = ref 84.1 /
cpu 36.2 (2.3x) / gpu 21.2 (4.0x vs ref).

## Axis 3 — PhysiCell mechanics (cell-cell forces)

PhysiCell's per-step cost is dominated by mechanics (~86% of `update_all_cells`;
`update_velocity` alone ~52%). Optimizations here must leave cell trajectories
bit-identical.

```
make verify-mech   # baseline vs optimized mechanics, 1 thread; diffs BOTH cell
                   #   state (*_cells.mat) AND microenvironment -> must be 0.000e+00
make bench-mech    # wall-time A/B (baseline vs optimized) on the benchmark config
```

Both use the SAME `project` binary; `PHYSICELL_MECH_BASELINE=1` selects the frozen
original force kernel (`Cell::add_potentials_baseline`). The optimized path is
verified bit-identical. To find the next hotspot, rebuild
`PhysiCell_cell_container.o` with `-DPHYSICELL_PROFILE_STEP` for a per-phase
wall-time breakdown of `update_all_cells` (zero cost otherwise).

## In-process micro-benchmark: `benchmark_biofvm.cpp`

A standalone harness that builds a microenvironment + agents and times the
diffusion+secretion loop directly, without the full PhysiCell loop. The *same
source* is compiled against `BioFVM copy/` (reference), `BioFVM/` (CPU-opt), and
`BioFVM/` + nvcc (GPU) — this is exactly what the `make bench3*` targets do
(see "Three-way micro-benchmark" above). It is also the model for adding new
micro-benchmarks of a specific optimization.

## How to add a new acceleration + its test

1. Implement the optimization in `BioFVM/` (leave `BioFVM copy/` as the reference).
2. Add a **correctness** comparison first: either
   - extend `BioFVM/tests/test_diffusion_cuda_integration.cpp` (in-process, bit-exact), or
   - use `make verify` (whole-simulation output diff), or
   - add a micro-test modeled on `benchmark_biofvm.cpp`.
3. Only once it matches the reference, report the speedup (`make benchmark` /
   `make bench-cuda`).
4. **Pin OpenMP threads** when benchmarking (`OMP_NUM_THREADS=N`) — default
   oversubscription produces wildly noisy timings.

For a **PhysiCell core** (e.g. mechanics) optimization, the same discipline with
an in-binary A/B: keep the original as a `*_baseline` function selected by an env
var, gate correctness on `make verify-mech` (cells + microenvironment, 1 thread,
must be 0.000e+00), then time with `make bench-mech`. Profile first with
`-DPHYSICELL_PROFILE_STEP` so you optimize the phase that actually dominates —
not every plausible-looking change moves the profiler (see the dev-notes).

## Repo hygiene

Large research/experiment outputs (`data/`, `wandb_csv_exports/`, root-level
`*.csv`/`*.png`/`*.pdf`/`*.py`) are **git-ignored** — they are RL experiment data,
not part of the BioFVM C++ work, and must never enter history. Real simulation
inputs under `config/` (e.g. `config/cell_rules.csv`) are kept.
