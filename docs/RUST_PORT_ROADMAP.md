# Porting BioFVM + PhysiCell to Rust — assessment and roadmap

Status: **assessment / planning document.** Nothing here has been implemented.
Written 2026-08-03 against PhysiCell 1.14.2 as vendored in this fork.

---

## TL;DR

**Do not do the full rewrite.** Reaching validated parity with what is in this
tree is a 12–24 month single-developer project, and the two things a rewrite
usually buys — speed and safety — split unevenly here:

- **Speed: already won, in C++.** [BENCHMARKS.md](../BENCHMARKS.md) records 2.3x
  (CPU-opt) and 4.0x (CUDA) against the reference solver, gated on bit-for-bit
  correctness. Rust and C++ both lower through LLVM; on a Thomas solver over
  flat `f64` arrays the language delta is noise. **Rust buys no speed here.**
- **Safety: a genuine, documented, repeat-offender problem in this repo** — see
  "The empirical case" below. This is real and Rust would eliminate the entire
  bug family by construction.

The recommended path is therefore **not** "rewrite" and **not** "do nothing",
but a bounded strangler port of Stages 1–2 (~6 weeks) that kills the global
singletons, followed by an explicit go/no-go. See
[Recommended minimal path](#recommended-minimal-path).

---

## The empirical case (what this repo actually says)

### Three instances of one bug family

[DEBUGGING_RANDOM_POLICY_SEGFAULT.md](DEBUGGING_RANDOM_POLICY_SEGFAULT.md)
documents the same defect three separate times:

| Site | Status |
|---|---|
| `release_internalized_substrates` | fixed earlier |
| `dynamic_spring_attachments` loop in `update_all_cells` | fixed 2026-07-07 |
| `standard_cell_cell_interactions` loop in `update_all_cells` | fixed 2026-07-07 |

All three are the identical mistake: an out-of-domain cell carries
`current_mechanics_voxel_index == -1`, that `-1` reaches
`agent_grid[...]` as a `std::vector` subscript, and the process dies. The fix
each time was to remember to add a guard.

This is the single most persuasive argument for Rust in this codebase, and it
has nothing to do with performance. The C++ design encodes "no voxel" as a
`-1` in an `int` that is *also* a valid array subscript type. The Rust
equivalent — `Option<VoxelIdx>`, with `VoxelIdx(u32)` and bounds-checked
indexing — makes all three bugs **compile errors**, not runtime segfaults. You
cannot forget the guard, because you cannot index with an `Option` without
unwrapping it.

A fourth crash **remains open**: `physicell.start(settingxml, reload=True)`,
the between-episode reinitialization at `physicellmodule.cpp:130–176`, which
tears down global state (`while(!(*all_cells).empty()) back()->die()`, reset
mesh/microenvironment, fresh cell container, re-run `setup_tissue()`). That is
not a coincidence — it is the *global singleton* problem, and it is precisely
what Stage 2 of this roadmap deletes.

### The global-state tax on the RL work

[resilient_sub_vec_env.py](../custom_modules/physigym/physigym/envs/resilient_sub_vec_env.py)
exists because a crashed PhysiCell cannot be restarted in-process — its
docstring notes that spawning a new instance "is forbidden". The cause is
visible in the headers:

```
core/PhysiCell_cell_container.h:121   extern std::vector<Cell*> *all_cells;
BioFVM/BioFVM_basic_agent.h:142       extern std::vector<Basic_Agent*> all_basic_agents;
core/PhysiCell_constants.h:154-159    extern double diffusion_dt, mechanics_dt, phenotype_dt, ...
core/PhysiCell_standard_models.h:79   extern Cycle_Model Ki67_advanced, live, apoptosis, ...
```

One simulation per process, forever. For vectorized RL that means one OS
process per environment, no in-process reset, and a bespoke supervisor to nurse
dead workers. Making `Microenvironment` and the cell arena into *values* rather
than globals is the highest-leverage structural change available, and it is
worth doing **whether or not** the language changes.

---

## What you are actually buying, and what you are giving up

| | Gain | Cost |
|---|---|---|
| Speed | ~0% (both are LLVM; hot loops already SoA + OpenMP + CUDA) | — |
| Memory safety | Entire `-1`-index / dangling-`Cell*` family eliminated at compile time | — |
| Data races | `omp critical` (10 sites) replaced by types that prove exclusion | — |
| N sims per process | Yes — but achievable in C++ too, more cheaply | — |
| Build/dep story | `cargo` vs. hand-rolled Makefile | Must still keep g++ for addons |
| Ecosystem | — | **Lose** the model zoo, PhysiCell Studio, upstream merges |
| User models | — | **Every** C++ `custom_modules` model stops compiling |
| Validation | — | Re-validate everything against published results |

The last three rows are the ones that decide it. **A Rust PhysiCell is a fork
with zero users on day one that cannot consume the existing model corpus.**

---

## Ground rules

### 1. The oracle comes first

This fork already has the right discipline — every acceleration is gated on
matching the reference. Keep it, and extend it before writing any Rust:

- Add a **golden-trace dump**: cell positions/state + substrate field, every N
  steps, serialized as `f64` hex (`%a`), not decimal. A whole-simulation
  end-state diff (`make verify`) is too coarse to localize a port bug to a
  stage.
- Trace must be emitted by the *current* C++ build first, and checked in as the
  reference artifact.

### 2. Decide the tolerance gate NOW — bit-exact is not attainable cross-language

This is the single most underestimated risk in the project. The current
bit-exact discipline works because both sides are g++ with identical flags.
That property does not survive a language boundary:

- `CFLAGS` here is `-march=native -O3 -mfpmath=both -fopenmp -std=c++11`.
  At `-O3 -march=native`, GCC will contract `a*b+c` into FMA and may route
  `exp`/`log`/`pow` through **libmvec** (vectorized libm) inside loops.
- Rust/LLVM does **not** enable FP contraction by default and has no stable
  `-ffast-math`. Scalar `f64::exp` goes to the system libm.
- Net: identical algebra, different last bits. Reduction order under `rayon`
  vs. `omp parallel for` differs again.

**Decision required before Stage 1:** either
(a) gate on a ULP/relative-error bound (e.g. `rel_err < 1e-12` per field, with
a documented drift budget over N steps), or
(b) build the C++ reference with `-ffp-contract=off -fno-fast-math` and
single-threaded reductions to create a bit-exact-able baseline, accepting that
it is slower than the shipping build.

Option (b) is more work up front and is the only one that lets you keep the
"0.000e+00" gate you have today. Pick one and write it down; do not discover it
in Stage 3.

### 3. Strangler, never big-bang

Rust is built as a `staticlib` and linked *into* the existing C++ binary. The
C++ build stays green at every commit. There is no branch where the simulation
does not run.

---

## Build integration

Rust produces a `staticlib`; `g++` links it directly.

```toml
# rust/Cargo.toml
[package]
name = "physicell_core"
edition = "2021"

[lib]
crate-type = ["staticlib"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"     # never unwind across the FFI boundary
```

Makefile additions (this fork's `CC := g++`, so no linker surprises):

```make
RUST_LIB := rust/target/release/libphysicell_core.a

$(RUST_LIB): $(shell find rust/src -name '*.rs')
	cd rust && cargo build --release

# append to the existing link lines (classic:, static:, project_opt:, ...)
#   $(RUST_LIB) -lpthread -ldl
```

Notes specific to this tree:

- `panic = "abort"` is mandatory. Unwinding a Rust panic through a C++ frame is
  UB. Every `extern "C"` entry point should additionally be wrapped in
  `catch_unwind` returning an error code, because a panic reaching Python via
  [physicellmodule.cpp](../custom_modules/extending/physicellmodule.cpp) is
  exactly the crash-in-a-worker situation you already have tooling for.
- The CUDA targets link with `nvcc` rather than `g++` (Makefile:166, 201, 219).
  `nvcc` delegates host linking to g++, so the `.a` still works, but the CUDA
  and Rust targets must be kept in sync by hand.
- Do **not** let Rust own OpenMP threads. Until Stage 3, Rust code called from
  inside an `omp parallel for` must be leaf, thread-agnostic, and allocation-free
  on the hot path.

---

## Stage-by-stage roadmap

Durations assume one experienced developer, full-time, including validation.

---

### Stage 0 — Harness and baseline

**Duration:** 1–2 weeks · **Ports nothing**

1. Implement the golden-trace dump (Ground rule 1) in the C++ build.
2. Resolve the tolerance-gate decision (Ground rule 2) and record it here.
3. Build a `-ffp-contract=off`, single-threaded reference binary if option (b).
4. Stand up `rust/` crate, link an empty `staticlib` into `classic:`, prove the
   build is still green and `make verify` still passes.
5. Set up `cargo test` + a `make test-rust` target wired into CI.

**Exit criteria:** an empty Rust lib is linked into the shipping binary; a
golden trace exists; the tolerance gate is written down.

**Why first:** every later stage is validated against this. Skipping it means
discovering in Stage 3 that you cannot tell a port bug from FP drift.

---

### Stage 1 — Diffusion solver

**Duration:** 3–6 weeks · **~1.5k LOC** ·
[BioFVM_solvers.cpp](../BioFVM/BioFVM_solvers.cpp) (892) +
[BioFVM_vector.cpp](../BioFVM/BioFVM_vector.cpp) (555)

The ideal beachhead: pure numerics, no cell graph, no globals, and this fork has
*already* flattened the Thomas coefficients into SoA arrays.

**Approach — borrow, don't own.** Rust does not allocate anything. It receives
raw pointers to the C++-owned arrays:

```rust
#[no_mangle]
pub unsafe extern "C" fn biofvm_lod_3d(
    p:  *mut f64,       // density field, len = n_voxels * n_substrates
    thomas_denom: *const f64,
    thomas_c:     *const f64,
    n_voxels: usize, n_substrates: usize,
    nx: usize, ny: usize, nz: usize,
) -> i32
```

Internally: `slice::from_raw_parts_mut`, then safe Rust for the whole solve.
The `unsafe` surface is three lines per entry point.

**Validation:** `make bench3` already times reference / CPU-opt / GPU on an
identical problem via [benchmark_biofvm.cpp](../benchmark_biofvm.cpp). Add a
fourth column. Gate on the Stage 0 tolerance.

**Risks:**
- The x/y/z sweeps have different memory-access patterns; the z-sweep is
  stride-`nx*ny` and is where the C++ optimizations live. Expect to need
  explicit chunking to match, and expect the naive Rust port to be *slower*
  before it is faster.
- Bounds-check overhead in the inner Thomas loop is real. Use
  `chunks_exact_mut` / iterator forms rather than `get_unchecked` — LLVM elides
  the checks when the iterator proves the range, and you keep the safety.

**Exit criteria:** `bench3` shows Rust within noise of CPU-opt, and the
tolerance gate passes. **This is the first real go/no-go.** If Rust is not
within ~10% of the tuned C++ here, on the easiest possible kernel, the rest of
the project will not go better.

---

### Stage 2 — Microenvironment and mesh: kill the singletons

**Duration:** 6–8 weeks · **~2.9k LOC** ·
[BioFVM_microenvironment.cpp](../BioFVM/BioFVM_microenvironment.cpp) (1804) +
[BioFVM_mesh.cpp](../BioFVM/BioFVM_mesh.cpp) (1127)

Ownership crosses the boundary: **Rust owns the substrate arrays**, C++ receives
a view. This is the stage that pays for itself independently of the rest.

```rust
pub struct Microenvironment {
    mesh: CartesianMesh,
    density: Vec<f64>,          // n_voxels * n_substrates, substrate-major
    substrates: Vec<SubstrateDef>,
    solver: SolverState,
}
// C++ gets an opaque handle:
//   me_create(...) -> *mut Microenvironment
//   me_density_ptr(h) -> *mut f64
//   me_destroy(h)
```

**What this deletes:** the global `microenvironment` singleton. Multiple
`Microenvironment` values can coexist in one process. This is a direct attack on
the **open** `start(reload=True)` crash — the reset path stops being "tear down
global state and hope" and becomes "drop the old value, construct a new one".

**Risks:**
- `Microenvironment` is referenced pervasively from `core/` and `modules/`. The
  C++ side needs a compatibility shim so the ~hundreds of `microenvironment.foo`
  call sites keep compiling. Budget real time for this; it is the bulk of the
  stage.
- The Dirichlet-condition machinery (per-voxel activation vectors) is fiddly and
  under-tested upstream. Write Rust tests for it before porting.

**Exit criteria:** two independent `Microenvironment`s run in one process with
correct, independent results; `make verify` still passes; the reset-path crash
is retested.

---

### Stage 3 — Cell arena and mechanics

**Duration:** 3–4 months · **~4k LOC** ·
[PhysiCell_cell.cpp](../core/PhysiCell_cell.cpp) (3574) +
[PhysiCell_cell_container.cpp](../core/PhysiCell_cell_container.cpp) (430)

**The crux of the whole project.** Also the largest single payoff: per
[BENCHMARKS.md](../BENCHMARKS.md), mechanics is ~86% of `update_all_cells` and
`update_velocity` alone is ~52%.

**The central transformation** — every `Cell*` becomes an index:

```rust
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct CellId(u32);

pub struct CellArena {
    cells: Vec<Cell>,
    free:  Vec<CellId>,          // recycled slots
    gen:   Vec<u32>,             // generation counter -> detects stale ids
}

pub struct Cell {
    pos: [f64; 3],
    mechanics_voxel: Option<VoxelIdx>,   // <- the -1 bug, eliminated
    attached:  Vec<CellId>,
    springs:   Vec<CellId>,
    neighbors: Vec<CellId>,
}
```

This is the standard Rust answer to a cyclic object graph
([PhysiCell_cell.h:145-148](../core/PhysiCell_cell.h#L145-L148)), and it fixes
two distinct bug families at once:

1. `Option<VoxelIdx>` — the `-1` subscript cannot be written.
2. Generational indices — a `CellId` held in a neighbor list after
   `divide()`/`delete_cell()` recycled the slot is *detected*, not silently
   dereferenced into a live-but-wrong cell. (C++ `Cell*` in `attached_cells`
   after deletion is a dangling pointer with no such protection.)

**Parallelism.** Replace `omp parallel for` + `omp critical` with the
read-phase/write-phase split that `rayon` makes natural: compute all forces from
`&CellArena` in parallel into a per-cell force buffer, then apply with `&mut` in
a serial (or disjoint-partitioned) pass. This is both faster than lock-based
mutation and deterministic — which the current code is not, per the
multi-cell-per-voxel note in [CUDA_GPU_NOTES.md](../BioFVM/CUDA_GPU_NOTES.md).

**Risks — this is where projects die:**
- `Cell::divide()` mutates the arena *while iterating it*. Needs a deferred
  birth/death queue drained between phases. Changes the order in which cells
  are created, which **changes RNG consumption order**, which changes
  trajectories. Plan for this: either replicate the exact upstream ordering, or
  accept the tolerance gate can no longer be bit-exact and re-validate
  statistically.
- `is_out_of_domain` semantics must be preserved exactly; it is load-bearing for
  the guards described above.
- Mechanics must stay bit-identical to keep `make verify-mech` meaningful, and
  that target is the most valuable safety net you have.

**Exit criteria:** `make verify-mech` passes (or its documented successor);
`bench-mech` shows no regression; trajectories match the golden trace.

---

### Stage 4 — Phenotype, cycle, and death models

**Duration:** 2–3 months · **~4k LOC** ·
[PhysiCell_phenotype.cpp](../core/PhysiCell_phenotype.cpp) (1466) +
[PhysiCell_phenotype.h](../core/PhysiCell_phenotype.h) (812) +
[PhysiCell_standard_models.cpp](../core/PhysiCell_standard_models.cpp) (1509)

Mechanically straightforward — state machines over transition-rate matrices —
but this is where the **extension API** must be designed (see below), because
`Phenotype` is where the ~20 user-facing function pointers live
([PhysiCell_phenotype.h:500-529](../core/PhysiCell_phenotype.h#L500-L529)).

**Risk:** the global `extern Cycle_Model Ki67_advanced, live, apoptosis, ...`
([PhysiCell_standard_models.h:79-81](../core/PhysiCell_standard_models.h#L79-L81))
are shared mutable templates that user code is *expected* to mutate at setup.
Reproducing that pattern safely needs a registry owned by the simulation value,
not a `static mut`.

---

### Stage 5 — Rules, signals, and behaviors

**Duration:** 2–3 months · **~5.2k LOC** ·
[PhysiCell_rules.cpp](../core/PhysiCell_rules.cpp) (2383) +
[PhysiCell_signal_behavior.cpp](../core/PhysiCell_signal_behavior.cpp) (2820)

Large but shallow: string→index dispatch tables and Hill-function response
curves. Genuinely *nicer* in Rust (enums, exhaustive `match`, `HashMap`), and
low-risk because the semantics are well-specified by the rules CSV format.

Good candidate to parallelize with Stage 4 if a second developer exists.

**Risk:** the signal/behavior index space is constructed at runtime from the
substrate and cell-definition lists; the index arithmetic is implicit and
under-documented. Port the *tests* first.

---

### Stage 6 — IO and configuration

**Duration:** ~2 months · **~7k LOC** · [modules/](../modules/)

| File | LOC | Rust replacement |
|---|---|---|
| pugixml (vendored) | 12444 | **delete** → `quick-xml` or `roxmltree` |
| [PhysiCell_pathology.cpp](../modules/PhysiCell_pathology.cpp) | 1851 | SVG rendering — tedious, zero risk |
| [PhysiCell_MultiCellDS.cpp](../modules/PhysiCell_MultiCellDS.cpp) | 1297 | must stay **byte-compatible** |
| [PhysiCell_settings.cpp](../modules/PhysiCell_settings.cpp) | 997 | `serde` + `quick-xml` |
| [BioFVM_matlab.cpp](../BioFVM/BioFVM_matlab.cpp) | 625 | `.mat` v4 writer — small, self-contained |

Deleting 12.4k LOC of vendored pugixml is the single largest LOC win in the
project.

**Hard constraint:** `.mat` and MultiCellDS output must remain byte-identical.
Every downstream analysis script, and both of
[diff_cells_mat.py](../BioFVM/tests/diff_cells_mat.py) /
[diff_microenvironment_mat.py](../BioFVM/tests/diff_microenvironment_mat.py),
depend on it.

---

### Stage 7 — The extension API (design, not code)

**This decision defines the project.** It should be made during Stage 4, not
deferred to the end.

PhysiCell's user API *is* the ~20 function pointers on `Phenotype`:
`update_phenotype`, `custom_cell_rule`, `update_velocity`, `contact_function`,
`cell_division_function`, `volume_update_function`, … Every published PhysiCell
model is C++ written against them.

Three options:

| Option | Shape | Consequence |
|---|---|---|
| **A. Trait objects** | `Box<dyn PhenotypeModel>` per cell definition | Most idiomatic; total break with C++ models |
| **B. `extern "C"` callbacks** | keep the fn-pointer table, C ABI | C++ models still work; keeps `unsafe` at the core |
| **C. Scripted rules only** | no user code — rules CSV + config | Cleanest; only viable if your models fit the rules grammar |

**Recommendation: B during migration, A as the end state.** B is what keeps the
C++ build green through Stages 3–6; A is what you want if the Rust version
becomes the product.

For this fork specifically, **C is worth serious consideration** — if the RL
work only ever drives the simulation through
[physigym](../custom_modules/physigym/) actions and the rules grammar, you may
not need a general C++ extension API at all, which removes the largest single
source of both `unsafe` and community breakage.

---

### Stage 8 — Addons: never port

**Duration:** ongoing FFI maintenance

| Addon | External dependency | Verdict |
|---|---|---|
| [libRoadrunner](../addons/libRoadrunner/) | libRoadRunner — SBML with an **LLVM JIT** | FFI only |
| [PhysiBoSS](../addons/PhysiBoSS/) | MaBoSS (C++ Boolean-network engine) | FFI only |
| [dFBA](../addons/dFBA/) | an LP solver | FFI only |
| [PhysiMeSS](../addons/PhysiMeSS/) | none (1157 LOC, in-tree) | portable, low priority |

`cxx` or `bindgen` bridges, indefinitely. **This is why the "one clean language"
payoff never fully arrives** — the g++ toolchain stays in the build regardless.

---

## Cost summary

| Stage | Scope | LOC | Duration |
|---|---|---|---|
| 0 | Harness, tolerance gate, crate skeleton | — | 1–2 wk |
| 1 | Diffusion solver | ~1.5k | 3–6 wk |
| 2 | Microenvironment + mesh | ~2.9k | 6–8 wk |
| 3 | **Cell arena + mechanics** | ~4k | 3–4 mo |
| 4 | Phenotype + cycle models | ~4k | 2–3 mo |
| 5 | Rules + signals | ~5.2k | 2–3 mo |
| 6 | IO + config | ~7k (−12.4k pugixml) | 2 mo |
| 7 | Extension API | design | (within 4) |
| 8 | Addons | — | never |
| | **Total to validated parity** | **~25k** | **12–24 months** |

"Parity" means matching undocumented behavior, including the nondeterminism
already catalogued in this fork. Upstream 1.14.x continues moving throughout.

---

## Decision points

Three explicit off-ramps. Each is a legitimate stopping place.

**After Stage 1 — the performance reality check.** If Rust is not within ~10% of
the tuned C++ on the easiest kernel in the codebase, stop. Nothing downstream
gets easier.

**After Stage 2 — the highest-value stopping point.** You have multi-sim per
process, the singleton is gone, the reset-path crash is addressed, and you still
have a fully working PhysiCell with the entire C++ ecosystem intact. **This is
the recommended stopping point unless something has changed.**

**After Stage 3 — the point of no return.** Once the cell arena is Rust, the C++
`Cell` API is gone and every user model must be rewritten. Do not cross this
line without having answered: *is the product the RL platform, or is it
PhysiCell compatibility?* You cannot have both.

---

## Recommended minimal path

1. **Verify the open reset-path crash is not a 2-line C++ fix.** It is the last
   known crash, it is in the between-episode reinit, and the step() crash in the
   same family turned out to be two missing guards. Rewriting 25k LOC to escape
   a bug you have not root-caused risks reproducing it as a `panic!`.
2. **Do Stage 0 + Stage 1 + Stage 2** (~3–4 months). Bounded, reversible,
   independently valuable, and it produces the thing the RL work actually needs:
   N simulations in one process.
3. **Stop and reassess at the Stage 2 off-ramp.**
4. Proceed to Stage 3+ **only** on an explicit decision that this fork's product
   is the RL platform rather than PhysiCell compatibility.

Two cheaper alternatives that deliver much of the same value and should be
priced before committing to any of the above:

- **De-globalize in C++.** Wrapping the singletons in a `Simulation` context
  object is weeks, not months, and buys the multi-sim-per-process win with zero
  ecosystem cost. It is also a strict prerequisite for a clean Stage 2, so it is
  not wasted work either way.
- **Check the prior art first.** There is existing work on distributed PhysiCell
  (PhysiCell-X, MPI) and on third-party BioFVM diffusion-solver acceleration.
  *Verify the current state of both* before writing any Rust — if Stages 1–2 are
  already solved by someone else, the calculus changes entirely.
