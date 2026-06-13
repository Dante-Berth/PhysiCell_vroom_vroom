/*
###############################################################################
# BioFVM_diffusion_cuda.h
#
# GPU-resident 3D LOD (Locally One-Dimensional) reaction-diffusion solver for
# BioFVM, with a compile-time backend switch:
#
#   - Built with nvcc (-D BioFVM_USE_CUDA, __CUDACC__ defined):
#       fields live in device memory; the LOD Thomas sweeps run as CUDA kernels.
#
#   - Built with a normal C++ compiler (no CUDA toolkit required):
#       the exact same code path compiles, "device memory" is host memory, and
#       the kernels are plain OpenMP loops that reproduce the current CPU solver
#       bit-for-bit. This lets the implementation be developed and validated on
#       a CPU-only machine and run unchanged on a GPU later.
#
# The solver is "GPU-resident" (Scope B): the SoA density buffer is kept on the
# device across timesteps. A dirty flag (gpu_field::host_dirty) records when the
# host SoA was modified (e.g. by cell secretion/uptake on the CPU) so the next
# solve uploads only when necessary; downloads happen lazily before host reads
# (I/O, gradients). This mirrors the AoS/SoA dirty-bit protocol already present
# in Microenvironment.
###############################################################################
*/

#ifndef __BioFVM_diffusion_cuda_h__
#define __BioFVM_diffusion_cuda_h__

#include <vector>

namespace BioFVM {

class Microenvironment;

// Opaque per-microenvironment GPU state. The definition is private to the .cu
// translation unit so that callers never need a CUDA toolkit to include this
// header.
struct gpu_field;

// Geometry + precomputed Thomas coefficients needed by the device solver.
// Coefficient arrays are flattened SoA: index [axis_coord * ns + s], matching
// the layout already built in BioFVM_solvers.cpp.
struct gpu_solver_params
{
    unsigned int nx = 0, ny = 0, nz = 0; // grid dimensions
    unsigned int ns = 0;                 // number of substrates (densities)
    unsigned int nv = 0;                 // nx*ny*nz voxels

    // Per-substrate constant c1 = dt*D/dx^2 (length ns).
    std::vector<double> c1;

    // Thomas forward-elimination denominators and back-substitution coeffs,
    // flattened [coord*ns + s] for each axis.
    std::vector<double> denom_x, cx;   // length nx*ns
    std::vector<double> denom_y, cy;   // length ny*ns
    std::vector<double> denom_z, cz;   // length nz*ns

    // Indices of substrates that actually diffuse (D>0); the solver only sweeps
    // these. Non-diffusing substrates get a scalar decay factor instead.
    std::vector<unsigned int> diff_subs;
    std::vector<unsigned int> nodiff_subs;
    std::vector<double>       nodiff_decay; // per nodiff substrate: 1/(1+c2)^3
};

// ---- backend lifecycle ------------------------------------------------------

// Allocate device buffers for params.nv * params.ns doubles plus scratch, and
// copy the (small) coefficient arrays to the device once. Returns an opaque
// handle owned by the caller; free with gpu_field_free.
gpu_field* gpu_field_alloc( const gpu_solver_params& params );
void       gpu_field_free( gpu_field* g );

// Host SoA <-> device transfers. `host_soa` is the contiguous SoA buffer
// (soa[s*nv + v]) already maintained by Microenvironment.
void gpu_upload( gpu_field* g, const double* host_soa );
void gpu_download( gpu_field* g, double* host_soa );

// ---- secretion / uptake (field stays resident) ------------------------------

// Per-cell secretion/uptake data, packed Structure-of-Arrays for the device.
// For cell c and substrate s the update applied at the cell's voxel is:
//     rho = (rho + temp1[c*ns+s]) / temp2[c*ns+s] + temp_export2[c*ns+s]
// which is exactly Basic_Agent::simulate_secretion_and_uptake on the SoA buffer
// (non-internalized-tracking path). `voxel[c]` is the cell's current_voxel_index.
//
// The arrays are sized by the *current* number of active cells, which changes
// every step; the host re-packs and re-uploads them each step. This transfer is
// O(cells * ns), independent of the (typically far larger) field size, so the
// field itself never has to leave the device.
struct gpu_secretion_batch
{
    unsigned int n_cells = 0;
    std::vector<int>    voxel;        // length n_cells
    std::vector<double> temp1;        // length n_cells*ns
    std::vector<double> temp2;        // length n_cells*ns
    std::vector<double> temp_export2; // length n_cells*ns
};

// Apply one secretion/uptake step on the device, in place, on the resident field.
// No field transfer occurs; only the (small) per-cell batch is uploaded.
void gpu_apply_secretion( gpu_field* g, const gpu_secretion_batch& batch );

// ---- solve ------------------------------------------------------------------

// Perform one full 3D LOD diffusion-decay step on whatever buffer is currently
// authoritative on the device. Dirichlet nodes are enforced via the boolean
// mask uploaded through gpu_set_dirichlet (host length nv; true = fixed).
void gpu_set_dirichlet( gpu_field* g, const char* host_mask /*len nv, or null*/ );
void gpu_solve_3D_LOD( gpu_field* g );

// True when this build actually targets a CUDA device (vs. the CPU fallback).
bool gpu_backend_is_cuda();

// Lightweight global counters of full-field host<->device transfers, for benchmarking
// the residency optimization (a resident steady state should show zero per step).
// gpu_transfer_bytes accumulates field bytes moved (uploads + downloads).
unsigned long long gpu_field_upload_count();
unsigned long long gpu_field_download_count();
unsigned long long gpu_field_transfer_bytes();
void gpu_reset_transfer_counters();

}; // namespace BioFVM

#endif
