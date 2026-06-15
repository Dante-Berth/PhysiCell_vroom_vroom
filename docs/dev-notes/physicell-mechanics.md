# PhysiCell mechanics acceleration

Applying the BioFVM methodology (every speedup has a matching bit-identical
correctness check) to PhysiCell's core, whose per-step cost is dominated by cell
mechanics, not diffusion.

## Where the time goes

Per-phase profile of `update_all_cells` (compile `PhysiCell_cell_container.o`
with `-DPHYSICELL_PROFILE_STEP`; prints a stderr breakdown at exit; zero cost
otherwise). First measurement (prof config, 311 cells, 8 threads):

| phase | share |
|-------|-------|
| `update_velocity` (cell-cell forces) | ~52% |
| `std_interactions` (phago/attack/fusion) | ~19% |
| gradients | ~12% |
| secretion | ~12% |
| everything else | <6% |

**Mechanics is ~86% of cell-update time.** `update_velocity` (which calls
`Cell::add_potentials` per cell pair over the Moore-neighbor voxels) is the
single biggest slice.

## Optimizations (all bit-identical, verified by `make verify-mech`)

1. **Squared-distance early-out in `add_potentials`** — the function contributes
   nothing once a pair is beyond the adhesion cutoff `max_interactive_distance`,
   so rejecting on `distance^2 >= max_interactive_distance^2` *before* the `sqrt`
   is exact. Gain scales with cell density: ~5% at ~320 cells, ~11% at 736.
2. **`is_neighbor_voxel` voxel centers by `const&`** — it took two
   `std::vector<double>` *by value* (two heap allocs/copies per call), called per
   Moore-neighbor voxel per cell per step, and only reads `[0..2]`. By-ref cut
   `update_velocity` 3.81s -> 2.98s (-22%) and `update_all_cells` -10%.

### A rejected change (kept as a lesson)
A per-thread memo of `find_cell_definition_index(type)` gave **zero** measurable
gain — the map lookups weren't the cost. It was removed. Profile, then measure
each change; don't keep optimizations that don't move the profiler.

## Harness

- `make verify-mech` — runs the SAME `project` binary twice (env
  `PHYSICELL_MECH_BASELINE=1` selects the frozen `Cell::add_potentials_baseline`),
  1 thread, diffs both `*_cells.mat` and microenvironment. Must be 0.000e+00.
- `make bench-mech` — wall-time A/B on the benchmark config.
- `BioFVM/tests/diff_cells_mat.py` — cell-state `.mat` differ (companion to the
  microenvironment one).

## Not done / not viable
- `std_interactions` (~19%) — another neighbor scan; candidate for the same
  early-out treatment. Open.
- GPU mechanics — impractical: the neighbor grid is pointer-linked and the
  per-cell work goes through user function pointers, neither GPU-resident-friendly.

> Point-in-time notes. Verify `file:line` citations against current code.
