# RESOLVED (2026-06-15): optimized-vs-reference verify discrepancy

**FIXED.** `make verify` now PASSES at 5.16e-14 (pure FP reassociation, well under tol).
Two-part fix:

1. **Real bug — stale SoA read in gradient computation.** `compute_all_gradient_vectors`
   and `compute_gradient_vector` (BioFVM_microenvironment.cpp) read `soa_p` directly
   WITHOUT first flushing pending AoS writes. Cells sense the field via AoS accessors,
   which set `aos_dirty=true` (an accessor may be written through). With `aos_dirty` set,
   `soa_p` is stale, so gradients were computed on a stale field. The reference computes
   gradients from the live field → cells made different chemotaxis decisions → the field
   diverged field-wide over ~1500 steps. FIX: both gradient functions now call
   `sync_soa_from_aos_if_dirty()` before reading `soa_p` (no-op when aos_dirty is false).
2. **Test harness footgun — `make verify` defaulted to 4 threads.** BioFVM's
   multi-cell-per-voxel secretion is `#pragma omp parallel` and order-dependent, so >1
   thread makes the REFERENCE itself nondeterministic (~1e-3 run-to-run) and the diff
   reports false ~0.99 "swaps". `make verify` is now pinned to OMP_NUM_THREADS=1.

**Two diagnostic traps that cost time (avoid next time):**
- **Stale `.o` files:** changing a header without `rm -f *.o` left old objects linked, so a
  correct fix appeared to have "no effect". Always `rm -f *.o` after editing any BioFVM
  header. (This is the same hazard noted in biofvm-optimizations.md.)
- **Stale output dirs:** the original 0.99 measurement came from `output_orig`/`output_opt`
  left over from a 4-thread run; re-diffing without regenerating compared old files.

**Correctness + speed now established (2026-06-15, omp=1):** `make verify` PASS (5e-14);
`make benchmark` orig ~16.6s vs opt ~10.2s = **1.63x** solo speedup. CUDA integration
tests still 0.000e+00.

---
## Original (INCORRECT) analysis, kept for the record

The "low/zero-D substrate" pattern below was a red herring: it came from diffing against
4-thread (nondeterministic) reference output. The real bug was diffusion-coefficient-
independent (the gradient stale-read). Original notes follow.

`make verify` (added 2026-06-13: runs project_orig from `BioFVM copy/` vs project_opt from `BioFVM/` on config/PhysiCell_settings_verify.xml, diffs microenvironment .mat via BioFVM/tests/diff_microenvironment_mat.py) FOUND A REAL DISCREPANCY between the optimized and reference BioFVM.

**Symptom:** max abs diff 0.99 (rel 585x). Deterministic — project_orig is bit-for-bit reproducible against itself (0.000e+00), so it is NOT RL randomness. Initial frame + frame 0 MATCH; divergence starts at frame 1.

**Localized pattern:** only LOW/ZERO-diffusion substrates diverge. In the verify config: drug_1 (D=0), cytokine (D=0.003), and a 6th substrate diverge; high-D substrates (anti/pro_tumoral D=3, tumor_molecule D=2) match exactly. Optimized decays these toward ~0 while original holds them ~0.99.

**Ruled out:**
- NOT the diffusion solver in isolation: a standalone 1-step D=0 probe gives 1.0000000000 in BOTH BioFVM copy and BioFVM/ (identical; neither decays a D=0 substrate in isolation — note auto_choose may route D=0 oddly).
- NOT Dirichlet: all substrates have Dirichlet disabled in the verify config.
- NOT the CUDA work: all 3 CUDA integration tests still pass 0.000e+00.

**Strong suspect:** interaction bug in the optimized path under the FULL multi-cell app — most likely the optimized SoA-direct fused secretion (Basic_Agent::simulate_secretion_and_uptake writing soa_p directly) + the lazy AoS<->SoA sync protocol (soa_dirty/aos_dirty/soa_is_authoritative), vs the original's AoS-based secretion ((*pS)(voxel) += ...). The baseline secretion (BioFVM_basic_agent_baseline.cpp) is verified identical-algorithm to BioFVM copy. Only manifests with cells present over multiple steps.

**REFINED 2026-06-13 (extensive probing):**
- App uses **LOD_2D** (use_2D=true in verify config), not LOD_3D. Confirmed both builds use LOD_2D.
- Divergence is **field-wide** at frame 1: substrate rows 5,7,8 (drug_1 D=0, cytokine D=0.003, and a 6th substrate — note the .mat has 6 substrate rows but XML configures only 5; custom code adds one) differ in 3967-3969 of 3969 voxels. NOT clustered at cells or edges → not a local secretion effect. Pattern = small/zero diffusion-LENGTH-SCALE substrates (drug_1 len 0, cytokine len 0.55 sub-voxel) diverge; len>=4.47 substrates match exactly.
- **Solver RULED OUT definitively:** isolated LOD_2D probes (D=0, uniform AND non-uniform init field, with and without secretion) give BIT-IDENTICAL results between BioFVM copy and BioFVM/. So neither the solver nor isolated secretion is the bug.
- **Root cause is an interaction only present in the full app:** the app reads densities/gradients mid-loop via Basic_Agent::nearest_density_vector / nearest_gradient (core/PhysiCell_standard_models.cpp:870,917,941,962 in phenotype updates), which triggers the optimized lazy sync_aos_from_soa_if_dirty -> unpack_from_soa. The isolated probes don't exercise this mid-loop unpack. Suspect: lazy SoA<->AoS sync produces a stale/double-applied state for low-D substrates field-wide (e.g. an unpack/re-pack round-trip that loses or re-applies decay).

**Next concrete step:** instrument the sync flags (soa_dirty/aos_dirty/soa_is_authoritative) across ONE full app step with a mid-loop density read, comparing low-D substrate SoA vs AoS values, to find the stale/double point. OR add a debug build that forces eager unpack after every solve and see if divergence vanishes (would confirm lazy-sync as culprit).

This is PRE-CUDA optimization work, separate from the GPU port. See biofvm-optimizations.md and cuda-biofvm.md. Repo is now standalone at github.com/Dante-Berth/PhysiCell_vroom_vroom (origin/main).
