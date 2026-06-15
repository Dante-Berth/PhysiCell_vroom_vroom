# Setup & build — fresh machine (CPU or GPU)

Onboarding guide for picking up this repo (an accelerated BioFVM fork of
PhysiCell, v1.14.2) on a new machine. Written to be followed top-to-bottom by a
human **or another LLM agent**. After this you can build the simulator, run the
correctness/benchmark suite, and (on a GPU box) build the CUDA backend.

Repo: <https://github.com/Dante-Berth/PhysiCell_vroom_vroom> (branch `main`).

---

## 0. What this repo is (orient first)

- PhysiCell + BioFVM, with the **BioFVM field solver accelerated** two ways:
  1. **CPU optimizations** in `BioFVM/` (SoA layout, flattened Thomas coeffs,
     OpenMP, lazy AoS↔SoA transpose) — see [biofvm-optimizations.md](biofvm-optimizations.md).
  2. A **dual-backend CUDA GPU-resident solver** — see
     [cuda-biofvm.md](cuda-biofvm.md) and the canonical runbook
     [`BioFVM/CUDA_GPU_NOTES.md`](../../BioFVM/CUDA_GPU_NOTES.md).
- `BioFVM copy/` is the **pristine upstream reference** kept for bit-for-bit
  diffing. Never edit it; it's the ground truth (see [`BENCHMARKS.md`](../../BENCHMARKS.md)).
- The simulator is also driven from Python via **physigym** (a Gymnasium env) for
  RL training. That's the `make all` path; the plain C++ build is `make classic`.
- ⚠️ There is a known **open correctness bug** in the optimized CPU path on
  low/zero-diffusion substrates — read [verify-discrepancy.md](verify-discrepancy.md)
  before trusting optimized output in production.

---

## 1. System prerequisites

### Both CPU-only and GPU machines

| Need | Why | Check |
|------|-----|-------|
| `g++` ≥ 7 with OpenMP | C++11, `-fopenmp`, `-march=native` | `g++ --version` |
| `make` | build | `make --version` |
| `git` | clone/push | `git --version` |
| Python ≥ 3.9 + `pip` | physigym / RL driver (optional for pure C++) | `python3 --version` |

Debian/Ubuntu:
```bash
sudo apt update && sudo apt install -y build-essential g++ make git python3 python3-pip python3-venv
```

The Makefile compiles with `-march=native -O3 -fopenmp -m64 -std=c++11`
(`CFLAGS` in [`Makefile`](../../Makefile)). Override the compiler with
`PHYSICELL_CPP=<path>` if `g++` isn't default (e.g. on macOS use a real GCC, not
Apple clang, for OpenMP).

### GPU machine — additionally

| Need | Why | Check |
|------|-----|-------|
| NVIDIA GPU + driver | run CUDA kernels | `nvidia-smi` |
| CUDA Toolkit (`nvcc`) | compile the `.cu` with `-D BioFVM_USE_CUDA` | `nvcc --version` |

The CUDA path uses `nvcc` with `-O3 -std=c++14` (`NVCC` / `NVCC_FLAGS` in the
Makefile, both overridable). **No GPU is required to build or test** — the
dual-backend falls back to an identical-results OpenMP path. You only need
`nvcc` to exercise the real device kernels.

---

## 2. Clone

```bash
git clone https://github.com/Dante-Berth/PhysiCell_vroom_vroom.git
cd PhysiCell_vroom_vroom
```

Note: `BioFVM copy/` is tracked in the repo, so it arrives with the clone. The
`output*/` folders are git-ignored — they are simulation results, not source.

---

## 3. Build the C++ simulator (CPU — works everywhere)

```bash
rm -f *.o            # always clean .o after a header change (see gotcha below)
make classic         # builds ./project, GPU solver linked as plain C++ (CPU fallback)
```

`make classic` produces the `project` executable. The CUDA solver
(`BioFVM_diffusion_cuda.cu`) is compiled as ordinary C++ here, so it links fine
on a CPU-only box.

> **Gotcha — stale object files.** The Makefile has no header dependency
> tracking. After editing any `BioFVM/*.h` (especially
> `BioFVM_microenvironment.h`), you **must** `rm -f *.o` before rebuilding, or
> you get silent ABI mismatches / "double free" crashes. When in doubt, clean.

Run a simulation:
```bash
./project config/PhysiCell_settings.xml
```

---

## 4. Correctness & benchmark suite (do this to confirm a good build)

All work today is validated on CPU via the fallback backend; targets live in the
[`Makefile`](../../Makefile). **Pin OpenMP threads** for any timing
(`OMP_NUM_THREADS=N`) — default oversubscription gives wildly noisy numbers.

```bash
# CUDA solver correctness (CPU fallback — no GPU needed):
make test-cuda              # unit test vs independent Thomas solver  -> expect 0.000e+00 PASS
make test-cuda-integration  # 3 end-to-end engine tests             -> expect all 0.000e+00 PASS

# Optimized-vs-reference whole-sim diff (CPU optimizations correctness):
OMP_NUM_THREADS=1 make verify     # diffs optimized BioFVM/ vs reference BioFVM copy/
                                  #   NOTE: currently FAILS on low/zero-D substrates,
                                  #   see verify-discrepancy.md — this is the open bug.

# Wall-time benchmarks:
OMP_NUM_THREADS=1 make benchmark   # whole-sim: project_orig vs project_opt
OMP_NUM_THREADS=$(nproc) make bench-cuda  # CPU two-call vs GPU-resident step + transfer counts
```

Expected good state on a healthy CPU build: `test-cuda` and
`test-cuda-integration` all report `0.000e+00` and `PASS`. The benchmark
methodology is documented in [`BENCHMARKS.md`](../../BENCHMARKS.md).

---

## 5. GPU build (only on a machine with `nvcc`)

This is the **first time the kernels touch a device** — expect to debug launch
config. Follow [`BioFVM/CUDA_GPU_NOTES.md`](../../BioFVM/CUDA_GPU_NOTES.md) §8
(prioritized TODO) which was written for exactly this moment.

```bash
nvcc --version              # confirm toolkit present
make test-cuda-gpu          # builds BioFVM_diffusion_cuda.cu with nvcc -D BioFVM_USE_CUDA, runs unit test
# override toolkit flags if needed:
make test-cuda-gpu NVCC=/usr/local/cuda/bin/nvcc NVCC_FLAGS="-O3 -std=c++14 -arch=sm_80"
```

To actually **use** the GPU path from a simulation (it is opt-in, not yet auto-
selected):
```cpp
M.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D_GPU;
// each step, replace the two CPU calls with the resident combined step:
M.simulate_diffusion_and_secretion_gpu( dt );
```
Host reads (sensing, output, `M(v)[s]`) work unchanged — they trigger one lazy
device→host download. See `CUDA_GPU_NOTES.md` §4–6 for the residency model and
the four gotchas (Dirichlet host-routing, multi-cell-per-voxel ordering,
internalized-substrate tracking not ported, pin OMP threads when benchmarking).

---

## 6. Python / physigym (RL driver — optional)

Only needed if running the Gymnasium/RL training loop, not for pure C++ sims.

```bash
python3 -m venv .venv && source .venv/bin/activate
make all     # compiles project AND pip-installs custom_modules/extending + physigym
```

Then, per the Makefile's own instructions:
```python
import gymnasium, physigym
gymnasium.envs.pprint_registry()
env = gymnasium.make("physigym/ModelPhysiCellEnv")
```

The packages are editable installs from `custom_modules/extending/` and
`custom_modules/physigym/` (each has its own `pyproject.toml`).

---

## 7. First-day checklist for a new agent/dev

1. Read this file, then [`BioFVM/CUDA_GPU_NOTES.md`](../../BioFVM/CUDA_GPU_NOTES.md)
   and [verify-discrepancy.md](verify-discrepancy.md).
2. `rm -f *.o && make classic` — confirm it builds.
3. `make test-cuda && make test-cuda-integration` — confirm `0.000e+00` / PASS.
4. If on GPU: `make test-cuda-gpu`, then work `CUDA_GPU_NOTES.md` §8 TODO list.
5. If touching the optimized CPU solver: the open bug in
   `verify-discrepancy.md` is the highest-value correctness target.

> All notes here are point-in-time. Verify any `file:line` citation against the
> current code before relying on it.
