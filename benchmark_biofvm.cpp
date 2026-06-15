/*
 * benchmark_biofvm.cpp
 *
 * Measures wall-clock time of BioFVM diffusion + secretion/uptake over N
 * timesteps. Compile twice — once against BioFVM/ (optimized) and once
 * against "BioFVM copy"/ (original) — via the Makefile targets below.
 *
 * What is timed:
 *   - simulate_diffusion_decay()   (Thomas solver LOD)
 *   - simulate_cell_sources_and_sinks()  (secretion/uptake)
 *
 * Domain: user config-like: 64x64x1 voxels, dx=1 µm, 5 substrates, N_AGENTS
 * agents uniformly distributed.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Use include path set at compile time via -DBIOFVM_INCLUDE_DIR
#include "BioFVM.h"
#include "BioFVM_microenvironment.h"
#include "BioFVM_basic_agent.h"
#ifdef BENCH_HAVE_GPU
#include "BioFVM_solvers.h"
#include "BioFVM_diffusion_cuda.h"
#endif

using namespace BioFVM;

// ── tuneable parameters (overridable on the command line) ───────────────
//   argv: NX NY NZ [N_AGENTS] [N_STEPS]
static int    N_STEPS    = 500;      // diffusion timesteps
static const double DT    = 0.01;    // minutes
static const int N_SUBSTRATES = 5;
static int    N_AGENTS   = 300;
static int    NX = 64, NY = 64, NZ = 1;
// In the GPU build, time the resident combined step instead of two-call.
static bool   USE_GPU    = false;
// ───────────────────────────────────────────────────────────────────────

int main( int argc, char** argv )
{
    if( argc >= 4 ) { NX = atoi(argv[1]); NY = atoi(argv[2]); NZ = atoi(argv[3]); }
    if( argc >= 5 ) N_AGENTS = atoi(argv[4]);
    if( argc >= 6 ) N_STEPS  = atoi(argv[5]);
#ifdef BENCH_HAVE_GPU
    USE_GPU = true;
#endif
    // ── build microenvironment (identical setup to bench_diffusion_cuda.cpp
    //    so the reference / CPU-opt / GPU numbers are directly comparable) ──
    Microenvironment M;
    M.name = "benchmark_env";
    M.set_density( 0, "s0", "dimensionless" );
    M.diffusion_coefficients[0] = 600.0;
    M.decay_rates[0]            = 0.1;

    for( int s = 1; s < N_SUBSTRATES; s++ )
    {
        M.add_density( "s" + std::to_string(s), "dimensionless" );
        M.diffusion_coefficients[s] = (s == 2) ? 0.0 : (100.0 + 50.0*s);
        M.decay_rates[s]            = 0.01 * s;
    }

    M.resize_space_uniform( 0.0, (double)NX, 0.0, (double)NY, 0.0, (double)NZ, 1.0 );
    set_default_microenvironment( &M );
    M.spatial_units = "micron";
    M.time_units    = "min";

    // initial field
    for( int v = 0; v < (int)M.mesh.voxels.size(); v++ )
        for( int s = 0; s < N_SUBSTRATES; s++ )
            M(v)[s] = 1.0 + 0.5*std::sin( 0.013*v + 0.7*s );

    // ── place agents (secreting + uptaking) ─────────────────────────────
    std::vector<Basic_Agent*> agents;
    agents.reserve( N_AGENTS );
    srand( 1 );
    for( int a = 0; a < N_AGENTS; a++ )
    {
        Basic_Agent* ag = create_basic_agent();
        ag->register_microenvironment( &M );
        ag->assign_position( (rand()%NX)+0.5, (rand()%NY)+0.5, (rand()%NZ)+0.5 );
        for( int s = 0; s < N_SUBSTRATES; s++ )
        {
            (*ag->secretion_rates)[s]      = 0.1*(s+1);
            (*ag->saturation_densities)[s] = 5.0;
            (*ag->uptake_rates)[s]         = 0.05*(s+1);
        }
        ag->set_total_volume( 1.0 );
        ag->set_internal_uptake_constants( DT );
        agents.push_back( ag );
    }

#ifdef BENCH_HAVE_GPU
    if( USE_GPU )
        M.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D_GPU;
#endif

    // ── warm-up (1 step; also triggers GPU setup/device alloc) ───────────
#ifdef BENCH_HAVE_GPU
    if( USE_GPU ) M.simulate_diffusion_and_secretion_gpu( DT );
    else
#endif
    { M.simulate_diffusion_decay( DT ); M.simulate_cell_sources_and_sinks( DT ); }

    // ── timed loop ──────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    for( int step = 0; step < N_STEPS; step++ )
    {
#ifdef BENCH_HAVE_GPU
        if( USE_GPU ) M.simulate_diffusion_and_secretion_gpu( DT );
        else
#endif
        { M.simulate_diffusion_decay( DT ); M.simulate_cell_sources_and_sinks( DT ); }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    // one host read (like writing output) — forces a single GPU download in GPU mode
    volatile double sink = 0;
    for( int v = 0; v < (int)M.mesh.voxels.size(); v += 101 ) sink += M(v)[0];
    (void)sink;

    double elapsed = std::chrono::duration<double>( t1 - t0 ).count();
    double per_step = elapsed / N_STEPS * 1e3; // ms

    const char* backend =
#ifdef BENCH_HAVE_GPU
        "GPU-hybrid (resident)";
#else
        BENCH_LABEL;
#endif

    std::printf( "\n========================================\n" );
    std::printf( "  BioFVM benchmark results [%s]\n", backend );
    std::printf( "  Domain : %dx%dx%d = %d voxels\n", NX, NY, NZ, NX*NY*NZ );
    std::printf( "  Substrates : %d\n", N_SUBSTRATES );
    std::printf( "  Agents : %d\n", N_AGENTS );
    std::printf( "  Steps : %d  (dt=%.4f min)\n", N_STEPS, DT );
    std::printf( "----------------------------------------\n" );
    std::printf( "  Total   : %.4f s\n", elapsed );
    std::printf( "  Per step: %.4f ms\n", per_step );
    std::printf( "========================================\n\n" );

    return 0;
}
