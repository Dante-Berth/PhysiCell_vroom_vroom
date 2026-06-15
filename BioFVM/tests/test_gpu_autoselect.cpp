// Verifies size-based GPU solver auto-selection in initialize_microenvironment.
// Built two ways (see Makefile test-gpu-autoselect / test-gpu-autoselect-gpu):
//   CPU-fallback build: gpu_backend_is_cuda()==false -> NEVER selects GPU (any size).
//   CUDA build        : selects GPU iff voxels >= BIOFVM_GPU_MIN_VOXELS threshold.
#include "../BioFVM.h"
#include "../BioFVM_solvers.h"
#include "../BioFVM_diffusion_cuda.h"
#include <cstdio>
#include <cstdlib>

using namespace BioFVM;

static bool is_gpu_solver( Microenvironment& M )
{ return M.diffusion_decay_solver == diffusion_decay_solver__constant_coefficients_LOD_3D_GPU; }

// Build a fresh default 3D microenvironment of the requested cubic-ish size.
static bool selects_gpu( int nx, int ny, int nz )
{
    default_microenvironment_options = Microenvironment_Options(); // reset to defaults
    default_microenvironment_options.initial_condition_from_file_enabled = false; // ctor leaves this indeterminate
    default_microenvironment_options.simulate_2D = false;
    default_microenvironment_options.X_range = {0, (double)nx};
    default_microenvironment_options.Y_range = {0, (double)ny};
    default_microenvironment_options.Z_range = {0, (double)nz};
    default_microenvironment_options.dx = 1; default_microenvironment_options.dy = 1; default_microenvironment_options.dz = 1;
    initialize_microenvironment();
    return is_gpu_solver( microenvironment );
}

int main()
{
    const bool cuda = gpu_backend_is_cuda();
    std::printf( "backend: %s\n", cuda ? "CUDA" : "CPU-fallback" );

    // Force a known threshold so the test is deterministic regardless of default.
    setenv( "BIOFVM_GPU_MIN_VOXELS", "1000000", 1 );

    bool small_gpu = selects_gpu( 50, 50, 50 );    // 125,000 voxels  (< 1e6)
    bool big_gpu   = selects_gpu( 110, 110, 110 ); // 1,331,000 voxels (>= 1e6)

    std::printf( "small grid (125k) -> %s\n", small_gpu ? "GPU" : "CPU" );
    std::printf( "big grid  (1.33M) -> %s\n", big_gpu   ? "GPU" : "CPU" );

    bool ok;
    if( cuda )
        ok = (!small_gpu) && big_gpu;   // small=CPU, big=GPU
    else
        ok = (!small_gpu) && (!big_gpu); // never GPU in a CPU-only build

    std::printf( "RESULT: %s\n", ok ? "PASS" : "FAIL" );
    return ok ? 0 : 1;
}
