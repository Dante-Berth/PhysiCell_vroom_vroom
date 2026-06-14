# CUDA / GPU port of BioFVM


User wants a GPU port of BioFVM/PhysiCell to avoid CPU↔GPU transfer. Agreed scope: **GPU-resident fields (Scope B)** — the whole *field* solver (diffusion/decay/secretion/uptake/gradients) on GPU, but **cell decision logic stays on CPU** (3500-line PhysiCell_cell.cpp has 341 branches + 113 dynamic-alloc/function-pointer sites → warp divergence, dynamic topology, user C++ rules don't exist on GPU; porting it = rewriting PhysiCell's programming model). Avoiding per-step field transfer comes from full field-residency + compact source/sink list, NOT from putting cells on GPU.

User has **no GPU currently** → hard constraint: must compile + be correct on CPU today, run fast on GPU later with zero source changes. Solution = compile-time backend switch.

**Architecture (dual-backend):** biofvm-optimizations.md
- `BioFVM/BioFVM_diffusion_cuda.h` — backend-agnostic interface (gpu_field opaque handle, gpu_field_alloc/free, gpu_upload/download, gpu_solve_3D_LOD, gpu_set_dirichlet).
- `BioFVM/BioFVM_diffusion_cuda.cu` — `#ifdef __CUDACC__ && BioFVM_USE_CUDA` → CUDA kernels (Thomas sweeps + transpose-between-sweeps for coalescing); `#else` → OpenMP fallback that reproduces the production CPU solver bit-for-bit. Built as plain C++ via `g++ -x c++` in default build.
- Solver fn `diffusion_decay_solver__constant_coefficients_LOD_3D_GPU` in BioFVM_solvers.cpp — same `void(Microenvironment&,double)` signature as CPU solver, opt-in (assign to M.diffusion_decay_solver). Added as `friend` in BioFVM_microenvironment.h to read thomas_* coeffs. Keeps SoA residency via static gpu_field keyed to (M,nv,ns).
- Reuses existing SoA dirty-bit protocol (soa_dirty/aos_dirty/soa_is_authoritative) in simulate_diffusion_decay.

**Makefile targets:** `make test-cuda` (CPU unit test), `make test-cuda-gpu` (nvcc), `make test-cuda-integration` (end-to-end CPU-solver vs GPU-solver through live engine). BioFVM_diffusion_cuda.o added to BioFVM_OBJECTS.

**Status (2026-06-12):** PoC + live-engine diffusion integration + secretion/uptake kernel all DONE and validated (CPU fallback, 0.000e+00 diff). Full `make classic` builds clean.

**Secretion/uptake kernel (added 2026-06-12):**
- `gpu_secretion_batch` (SoA: voxel[], temp1[], temp2[], temp_export2[], all per-cell*ns) + `gpu_apply_secretion(g,batch)` in BioFVM_diffusion_cuda.{h,cu} (CUDA kernel one-thread-per-cell + CPU fallback). Math: rho=(rho+t1)/t2+ex at cell voxel, exactly Basic_Agent::simulate_secretion_and_uptake non-internalized path on SoA.
- `Basic_Agent::pack_secretion_row(dt, voxel_out, t1*, t2*, ex*)` packs one cell's row (recomputes consts if volume_is_changed). Added to BioFVM_basic_agent.{h,cpp}.
- `Microenvironment::simulate_diffusion_and_secretion_gpu(dt)` = combined resident step (BioFVM_microenvironment.cpp). Packs batch from all_basic_agents each step, uploads only the batch, diffuses+secretes on resident device field, downloads field to soa_p (the one remaining transfer not yet eliminated). `void* gpu_field_handle` added to Microenvironment to share the device field. `gpu_ensure_field(M)` (BioFVM_solvers.cpp, friend) builds/caches the field.
- Tests: BioFVM/tests/test_diffusion_cuda_integration.cpp now has Test1 (diffusion) + Test2 (diff+secretion), both 0.000e+00. `make test-cuda-integration`.

**KEY FINDING — BioFVM CPU secretion is nondeterministic:** simulate_cell_sources_and_sinks is `#pragma omp parallel for` over agents; when 2 cells share a voxel, rho=(rho+t1)/t2+ex is order-dependent → CPU path differs ~3e-3 run-to-run (verified). GPU kernel processes cells in index order. So Test2 forces omp_set_num_threads(1) on the CPU reference for apples-to-apples; collision-ordering differences are a property of BioFVM, not the GPU port.

**Zero field-transfer residency (added 2026-06-12):** DONE. `Microenvironment::gpu_resident` flag (+ `sync_soa_from_gpu_if_resident()`) means device field is authoritative & soa_p stale. unpack_from_soa() and sync_aos_from_soa_if_dirty() trigger one lazy device->host download only on real host reads. simulate_diffusion_and_secretion_gpu skips upload when already resident (uploads only on aos_dirty/dirichlet/first step). Dirichlet path routes through host (cheap O(nv) scan via mesh.voxels[i].is_Dirichlet — note: dirichlet_indices member is COMMENTED OUT in the header, don't use it). Validated: integration Test3 interleaves mid-sim host reads, all 3 tests still 0.000e+00.

**Transfer counters:** gpu_field_upload_count/download_count/transfer_bytes/reset in BioFVM_diffusion_cuda. Benchmark confirms 1 upload + 1 download over 200 steps (0.01 transfers/step) — residency works.

**Benchmark (`make bench-cuda`, BioFVM/tests/bench_diffusion_cuda.cpp):** 80x80x40, 4 substrates, 5000 agents. CRITICAL: must pin OMP threads — default oversubscription gave wild noise (saw 9.6/17.6 ms phantom slowdowns). Real CPU-fallback numbers (best-of-3): 1 thread GPU 1.07x FASTER, 4thr 1.05x faster, 8thr 0.92x, 16thr 0.91x. I.e. the dual-backend abstraction adds ~0% overhead on CPU; on a real GPU the on-device kernels are the actual speedup (unmeasurable here, no GPU). Batch buffer made persistent (static local in simulate_diffusion_and_secretion_gpu) to avoid per-step realloc of n*ns arrays.

**NOT yet done / next steps:** (1) wire into auto_choose_diffusion_decay_solver; (2) Dirichlet-on-device kernel (still host-routed; stub kernel exists); (3) gradient computation on GPU; (4) internalized-substrate-tracking branch in secretion kernel (only non-tracking path ported); (5) benchmark on real GPU (need hardware); (6) multi-cell-per-voxel: GPU does index-order, CPU omp races — see KEY FINDING above.

Prior art to not reinvent: BioFVM-X (published GPU/MPI BioFVM).
