# BioFVM acceleration — developer notes

Working notes on the BioFVM acceleration effort (CPU optimizations + CUDA/GPU
port). These are kept in the repo so they travel with the code across machines.

The **canonical GPU runbook** is [`BioFVM/CUDA_GPU_NOTES.md`](../../BioFVM/CUDA_GPU_NOTES.md)
(file map, build/test targets, how to use the GPU path, gotchas, TODO for real
GPU hardware). Start there when resuming GPU work.

| File | What it covers |
|------|----------------|
| [SETUP.md](SETUP.md) | **Start here on a fresh machine** — prerequisites, clone, build, test/benchmark, GPU build, physigym. Written for a human or LLM agent. |
| [biofvm-optimizations.md](biofvm-optimizations.md) | CPU optimizations: SoA layout, flattened Thomas coeffs, OpenMP, lazy AoS↔SoA transpose. Timings. |
| [cuda-biofvm.md](cuda-biofvm.md) | CUDA/GPU port (Scope B, dual-backend): architecture, secretion kernel, field residency, status, next steps. |
| [verify-discrepancy.md](verify-discrepancy.md) | **RESOLVED (2026-06-15)**: the verify discrepancy was a stale-SoA read in gradient computation (+ a 4-thread test footgun). `make verify` now PASSES at 5e-14; CPU is 1.63x faster and bit-equivalent. |

Benchmarking & correctness methodology lives in [`BENCHMARKS.md`](../../BENCHMARKS.md).

> These started as session notes and are point-in-time. Verify `file:line`
> citations against current code before trusting them.
