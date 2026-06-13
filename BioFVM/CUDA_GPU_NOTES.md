# BioFVM CUDA / GPU port — resume runbook

This document is the portable, version-controlled record of the GPU acceleration
work on BioFVM. It is written so that a future session (on real GPU hardware, a
different machine, or a fresh checkout) can pick the work up immediately. It lives
in the repo on purpose: it travels with the code and survives machine/path changes.

Last validated on a **CPU-only machine** (no NVIDIA GPU) using the dual-backend
fallback. Everything below compiles, runs, and passes there; the CUDA kernels are
written but have **not yet run on a device**.

---

## 1. What this is (scope)

- **Goal:** keep BioFVM's *field* solver (diffusion/decay/secretion/uptake)
  resident on the GPU to avoid per-step CPU↔GPU transfer. **Cell decision logic
  stays on the CPU** — porting it would mean rewriting PhysiCell's programming
  model (branchy, dynamic cell birth/death, user C++ rules) and is explicitly out
  of scope.
- **Dual-backend design:** one source file compiles two ways.
  - `nvcc -D BioFVM_USE_CUDA` → real CUDA kernels, field in device memory.
  - normal `g++ -x c++` → identical-results OpenMP fallback, "device" = host
    memory. This is what lets the code be correct on a CPU-only machine today and
    run on a GPU later **with zero source changes**.

## 2. File map

| File | Role |
|------|------|
| `BioFVM/BioFVM_diffusion_cuda.h` | Backend-agnostic interface (no CUDA toolkit needed to include). `gpu_field` opaque handle, alloc/free, upload/download, `gpu_solve_3D_LOD`, `gpu_apply_secretion`, transfer counters, `gpu_backend_is_cuda()`. |
| `BioFVM/BioFVM_diffusion_cuda.cu` | Both backends. `#if defined(__CUDACC__) && defined(BioFVM_USE_CUDA)` → CUDA kernels (Thomas sweeps + transpose-for-coalescing, decay, secretion); `#else` → OpenMP fallback reproducing the production CPU solver bit-for-bit. |
| `BioFVM/BioFVM_solvers.{h,cpp}` | `diffusion_decay_solver__constant_coefficients_LOD_3D_GPU` (drop-in solver, same signature as CPU LOD_3D) and `gpu_ensure_field(M)` (builds/caches the device field on `M.gpu_field_handle`). |
| `BioFVM/BioFVM_microenvironment.{h,cpp}` | `simulate_diffusion_and_secretion_gpu(dt)` (resident combined step), `gpu_resident` flag + `sync_soa_from_gpu_if_resident()` (lazy device→host download), `gpu_field_handle`. |
| `BioFVM/BioFVM_basic_agent.{h,cpp}` | `pack_secretion_row(...)` packs one cell's secretion/uptake row into the GPU batch. |
| `BioFVM/tests/test_diffusion_cuda.cpp` | Unit test vs. an independent reference Thomas solver. |
| `BioFVM/tests/test_diffusion_cuda_integration.cpp` | 3 end-to-end tests through the live engine (diffusion / diffusion+secretion / interleaved host reads). |
| `BioFVM/tests/bench_diffusion_cuda.cpp` | CPU two-call vs GPU-resident benchmark + transfer counters. |

## 3. Build & test targets (in `Makefile`)

CPU-only (works today, no toolkit):
```
make test-cuda             # unit test, CPU fallback
make test-cuda-integration # 3 integration tests, CPU fallback
make bench-cuda            # benchmark (PIN OMP THREADS — see gotcha #1)
make classic               # full PhysiCell build (GPU solver linked, compiled as C++)
```

GPU (on a machine with `nvcc`):
```
make test-cuda-gpu         # builds the .cu with nvcc -D BioFVM_USE_CUDA, runs unit test
# NVCC ?= nvcc and NVCC_FLAGS ?= -O3 -std=c++14 are overridable on the make line.
```

## 4. How to USE the GPU path from a simulation

The GPU solver is **opt-in** (not yet wired into `auto_choose_diffusion_decay_solver`).
To use it:
```cpp
M.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D_GPU;
// then each step, instead of the usual two calls, use the resident combined step:
M.simulate_diffusion_and_secretion_gpu( dt );
// (this fuses diffusion + cell secretion/uptake on the resident device field)
```
Host reads (sensing, SVG/MultiCellDS output, `M(v)[s]`) work unchanged — they
trigger exactly one lazy device→host download via the existing
`sync_aos_from_soa_if_dirty` path.

## 5. Architecture in one paragraph

The SoA density buffer (`soa[s*nv + v]`) lives on the device. A three-tier
authority model extends BioFVM's existing AoS/SoA dirty-bit protocol:
`aos_dirty` (host AoS written) → `soa_dirty` (host SoA current) → **`gpu_resident`
(device authoritative, host stale)**. The combined step uploads only when the host
changed something (external dose/init/Dirichlet or first step), solves + secretes
on the device, and leaves the field resident — it does **not** download. The next
genuine host read pulls it down once. Secretion uses a compact per-cell batch
(`voxel[], temp1[], temp2[], temp_export2[]`, size ∝ cells×substrates, not field
size), so cell↔field coupling never moves the whole grid.

## 6. ⚠️ Gotchas (read before benchmarking on GPU)

1. **Pin OpenMP threads when benchmarking.** Default oversubscription produced
   wildly noisy timings (phantom 6–11× slowdowns). With pinned threads + best-of-N,
   the CPU fallback overhead is ~0% (the abstraction is essentially free on CPU).
   The benchmark now pins `omp_get_max_threads()` and takes best-of-3.
2. **Multi-cell-per-voxel ordering.** BioFVM's CPU secretion is an *unordered*
   `#pragma omp parallel for` over cells; when two cells share a voxel, the result
   is order-dependent and **nondeterministic run-to-run** (~3e-3). The GPU kernel
   processes cells in index order. Tests force single-threaded CPU secretion for
   apples-to-apples comparison. This is a property of BioFVM, not a bug in the port.
3. **Dirichlet is still host-routed.** A device kernel hook (`k_apply_dirichlet`)
   exists but is a stub; when Dirichlet nodes are present the combined step routes
   the field through the host to re-impose them. The `dirichlet_indices` member in
   the header is **commented out** — detect presence via `mesh.voxels[i].is_Dirichlet`.
4. **Internalized-substrate tracking** branch of secretion is **not** ported — only
   the common non-tracking path. If `track_internalized_substrates_in_each_agent`
   is true, the GPU secretion kernel will be wrong. Add that branch before using it
   with internalized tracking.

## 7. Validation status (CPU fallback)

- `make classic` — builds clean, GPU solver linked.
- `make test-cuda` — PASS (0.000e+00 vs reference).
- `make test-cuda-integration` — all 3 tests 0.000e+00.
- `make bench-cuda` — transfer counters confirm **1 upload + 1 download over 200
  steps** (≈0 per step): residency works.

## 8. TODO when on real GPU hardware (priority order)

1. **Build with nvcc and run `make test-cuda-gpu`** — first time the kernels touch a
   device. Fix any launch/config issues. Then `make bench-cuda` (built via nvcc) for
   the *real* speedup number (CPU-only build can't measure device acceleration).
2. **Wire into `auto_choose_diffusion_decay_solver`** so 3D sims pick the GPU solver
   automatically (guard on `gpu_backend_is_cuda()` / a runtime flag).
3. **Dirichlet-on-device kernel** — fill in `k_apply_dirichlet`, upload the boolean
   mask + value vectors once, drop the host round-trip.
4. **Internalized-substrate-tracking** secretion branch in the kernel.
5. **Gradient computation on GPU** (`compute_all_gradient_vectors`) so gradients
   never force a field download.
6. **Profile transpose vs. cuSPARSE `gtsv` batched tridiagonal** for the Thomas
   sweeps; tune block sizes.

## 9. Prior art

BioFVM-X (published GPU/MPI BioFVM) — don't reinvent; compare approach/perf.
