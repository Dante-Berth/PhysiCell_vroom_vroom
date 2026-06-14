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
```

All Axis-2 correctness tests assert **bit-for-bit** (0.000e+00) agreement against
the CPU solver, except where BioFVM's own CPU secretion is nondeterministic (see
the multi-cell-per-voxel note in `BioFVM/CUDA_GPU_NOTES.md`).

## In-process micro-benchmark: `benchmark_biofvm.cpp`

A standalone harness that builds a microenvironment + agents and times
`simulate_diffusion_decay` / secretion directly, without the full PhysiCell loop.
It includes BioFVM via a compile-time include path, so the *same source* can be
compiled against either `BioFVM/` or `BioFVM copy/` to compare solvers in isolation.
This is the model for adding new micro-benchmarks of a specific optimization.

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

## Repo hygiene

Large research/experiment outputs (`data/`, `wandb_csv_exports/`, root-level
`*.csv`/`*.png`/`*.pdf`/`*.py`) are **git-ignored** — they are RL experiment data,
not part of the BioFVM C++ work, and must never enter history. Real simulation
inputs under `config/` (e.g. `config/cell_rules.csv`) are kept.
