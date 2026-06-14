# BioFVM acceleration — developer notes

Working notes on the BioFVM acceleration effort (CPU optimizations + CUDA/GPU
port). These are kept in the repo so they travel with the code across machines.

The **canonical GPU runbook** is [`BioFVM/CUDA_GPU_NOTES.md`](../../BioFVM/CUDA_GPU_NOTES.md)
(file map, build/test targets, how to use the GPU path, gotchas, TODO for real
GPU hardware). Start there when resuming GPU work.

| File | What it covers |
|------|----------------|
| [biofvm-optimizations.md](biofvm-optimizations.md) | CPU optimizations: SoA layout, flattened Thomas coeffs, OpenMP, lazy AoS↔SoA transpose. Timings. |
| [cuda-biofvm.md](cuda-biofvm.md) | CUDA/GPU port (Scope B, dual-backend): architecture, secretion kernel, field residency, status, next steps. |
| [verify-discrepancy.md](verify-discrepancy.md) | **OPEN BUG**: optimized BioFVM diverges from the reference on low/zero-diffusion substrates in the full RL sim. |

Benchmarking & correctness methodology lives in [`BENCHMARKS.md`](../../BENCHMARKS.md).

> These started as session notes and are point-in-time. Verify `file:line`
> citations against current code before trusting them.
