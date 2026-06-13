/*
###############################################################################
# bench_diffusion_cuda.cpp
#
# Benchmark: CPU two-call stepping vs GPU-resident combined stepping, on a 3D
# BioFVM microenvironment with secreting/uptaking cells.
#
#   CPU path : simulate_diffusion_decay(dt) + simulate_cell_sources_and_sinks(dt)
#   GPU path : simulate_diffusion_and_secretion_gpu(dt)   (field stays resident)
#
# Reports per-step wall time for each, the speedup, and the count of full-field
# host<->device transfers (which the residency optimization should drive to ~0
# per step in steady state). On a CPU-only build the "GPU" path runs the OpenMP
# fallback, so the speedup number reflects only the residency/secretion-batching
# bookkeeping; on a real GPU it reflects actual device acceleration. The transfer
# counts are meaningful either way.
#
# Build: make bench-cuda
###############################################################################
*/

#include "../BioFVM.h"
#include "../BioFVM_solvers.h"
#include "../BioFVM_diffusion_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace BioFVM;

// Problem size — pick something big enough to be solver-bound.
static int    NX = 80, NY = 80, NZ = 40;
static int    N_SUBSTRATES = 4;
static int    N_AGENTS = 5000;
static int    N_STEPS = 200;
static const double DT = 0.01;

static void config_densities( Microenvironment& M )
{
    M.set_density( 0, "s0", "dimensionless" );
    M.diffusion_coefficients[0] = 600.0; M.decay_rates[0] = 0.1;
    for( int s = 1; s < N_SUBSTRATES; s++ )
    {
        M.add_density( "s" + std::to_string(s), "dimensionless" );
        M.diffusion_coefficients[s] = (s == 2) ? 0.0 : (100.0 + 50.0*s);
        M.decay_rates[s] = 0.01 * s;
    }
}
static void init_field( Microenvironment& M )
{
    for( int v = 0; v < (int)M.mesh.voxels.size(); v++ )
        for( int s = 0; s < N_SUBSTRATES; s++ )
            M(v)[s] = 1.0 + 0.5*std::sin( 0.013*v + 0.7*s );
}
static void place_agents( Microenvironment& M )
{
    std::srand( 1 );
    for( int a=0;a<N_AGENTS;a++)
    {
        Basic_Agent* ag = create_basic_agent();
        ag->register_microenvironment( &M );
        ag->assign_position( (std::rand()%NX)+0.5, (std::rand()%NY)+0.5, (std::rand()%NZ)+0.5 );
        for( int s=0;s<N_SUBSTRATES;s++)
        {
            (*ag->secretion_rates)[s]      = 0.1*(s+1);
            (*ag->saturation_densities)[s] = 5.0;
            (*ag->uptake_rates)[s]         = 0.05*(s+1);
        }
        ag->set_total_volume( 1.0 );
        ag->set_internal_uptake_constants( DT );
    }
}

static double time_run( bool gpu, int n_steps )
{
    all_basic_agents.clear(); reset_max_basic_agent_ID();
    Microenvironment M; M.name="bench"; config_densities(M);
    M.resize_space_uniform( 0,(double)NX, 0,(double)NY, 0,(double)NZ, 1.0 );
    set_default_microenvironment( &M );
    init_field(M);
    place_agents(M);
    M.diffusion_decay_solver = gpu
        ? diffusion_decay_solver__constant_coefficients_LOD_3D_GPU
        : diffusion_decay_solver__constant_coefficients_LOD_3D;

    // warm-up (also triggers one-time setup / device alloc), not timed
    if( gpu ) M.simulate_diffusion_and_secretion_gpu( DT );
    else { M.simulate_diffusion_decay( DT ); M.simulate_cell_sources_and_sinks( DT ); }

    gpu_reset_transfer_counters();
    auto t0 = std::chrono::high_resolution_clock::now();
    for( int s=0;s<n_steps;s++)
    {
        if( gpu ) M.simulate_diffusion_and_secretion_gpu( DT );
        else { M.simulate_diffusion_decay( DT ); M.simulate_cell_sources_and_sinks( DT ); }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    // one host read at the end (like writing output) — triggers a single download in GPU mode
    volatile double sink=0; for( int v=0; v<(int)M.mesh.voxels.size(); v+=101 ) sink += M(v)[0]; (void)sink;
    return std::chrono::duration<double>( t1 - t0 ).count();
}

int main( int argc, char** argv )
{
    if( argc >= 4 ) { NX=atoi(argv[1]); NY=atoi(argv[2]); NZ=atoi(argv[3]); }
    if( argc >= 5 ) N_AGENTS = atoi(argv[4]);
    if( argc >= 6 ) N_STEPS  = atoi(argv[5]);

    int nthreads = 1;
#ifdef _OPENMP
    // Pin thread count for stable, comparable numbers (default scheduling oversubscribes
    // and produces wildly noisy timings). Honor OMP_NUM_THREADS if set.
    nthreads = omp_get_max_threads();
    omp_set_num_threads( nthreads );
#endif
    std::printf( "backend : %s, %d OpenMP thread(s)\n",
                 gpu_backend_is_cuda() ? "CUDA" : "CPU-fallback (OpenMP)", nthreads );
    std::printf( "grid    : %dx%dx%d = %d voxels, %d substrates, %d agents, %d steps\n\n",
                 NX,NY,NZ, NX*NY*NZ, N_SUBSTRATES, N_AGENTS, N_STEPS );

    // Best-of-N trials to suppress turbo/thermal/scheduler noise.
    const int TRIALS = 3;
    double t_cpu = 1e30, t_gpu = 1e30;
    unsigned long long up_g=0, dn_g=0; double bytes_g=0;
    for( int t=0; t<TRIALS; t++ )
    {
        double c = time_run( false, N_STEPS ); if( c < t_cpu ) t_cpu = c;
        double g = time_run( true,  N_STEPS ); if( g < t_gpu ) t_gpu = g;
        up_g = gpu_field_upload_count(); dn_g = gpu_field_download_count();
        bytes_g = (double)gpu_field_transfer_bytes();
    }
    unsigned long long up_c=0, dn_c=0;

    std::printf( "CPU two-call    : %8.3f ms/step  (best of %d)\n", t_cpu/N_STEPS*1e3, TRIALS );
    std::printf( "GPU resident    : %8.3f ms/step  (best of %d)\n", t_gpu/N_STEPS*1e3, TRIALS );
    std::printf( "speedup         : %8.2fx%s\n\n", t_cpu / t_gpu,
                 gpu_backend_is_cuda() ? "" : "   (CPU fallback: <1 expected; real GPU inverts this)" );

    std::printf( "field transfers over %d steps:\n", N_STEPS );
    std::printf( "  CPU path : uploads=%llu downloads=%llu (n/a — no device)\n", up_c, dn_c );
    std::printf( "  GPU path : uploads=%llu downloads=%llu  (%.2f MB total, %.4f transfers/step)\n",
                 up_g, dn_g, bytes_g/1e6, (double)(up_g+dn_g)/N_STEPS );
    return 0;
}
