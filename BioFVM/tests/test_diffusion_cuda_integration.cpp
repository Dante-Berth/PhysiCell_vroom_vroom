/*
###############################################################################
# test_diffusion_cuda_integration.cpp
#
# End-to-end integration tests for the GPU-resident BioFVM path:
#
#  Test 1 (diffusion only): run N steps of diffusion-decay two ways from
#    identical state — production CPU LOD_3D solver vs GPU LOD_3D_GPU solver —
#    and compare the fields.
#
#  Test 2 (diffusion + secretion, resident): place cells with secretion/uptake,
#    then compare:
#      (a) CPU: simulate_diffusion_decay + simulate_cell_sources_and_sinks
#      (b) GPU: simulate_diffusion_and_secretion_gpu  (field stays on device,
#          only the per-cell batch is transferred each step)
#    These must agree to tight tolerance.
#
# Exercises the full engine path (sync, dirichlet, agent packing) — not just the
# kernels. Build via Makefile target `test-cuda-integration`.
###############################################################################
*/

#include "../BioFVM.h"
#include "../BioFVM_solvers.h"
#include "../BioFVM_diffusion_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace BioFVM;

static constexpr int    NX = 21, NY = 17, NZ = 13;
static constexpr int    N_SUBSTRATES = 4;
static constexpr double DT = 0.01;
static constexpr int    N_STEPS = 40;
static constexpr int    N_AGENTS = 150;

static void config_densities( Microenvironment& M )
{
    M.set_density( 0, "s0", "dimensionless" );
    M.diffusion_coefficients[0] = 600.0; M.decay_rates[0] = 0.1;
    for( int s = 1; s < N_SUBSTRATES; s++ )
    {
        M.add_density( "s" + std::to_string(s), "dimensionless" );
        M.diffusion_coefficients[s] = (s == 2) ? 0.0 : (100.0 + 50.0*s); // s2 non-diffusing
        M.decay_rates[s] = 0.01 * s;
    }
}

static void init_field( Microenvironment& M )
{
    for( int v = 0; v < (int)M.mesh.voxels.size(); v++ )
        for( int s = 0; s < N_SUBSTRATES; s++ )
            M(v)[s] = 1.0 + 0.5*std::sin( 0.13*v + 0.7*s );
}

static void extract( Microenvironment& M, std::vector<double>& out )
{
    const int nv = (int)M.mesh.voxels.size();
    out.assign( (size_t)nv*N_SUBSTRATES, 0.0 );
    for( int v = 0; v < nv; v++ )
        for( int s = 0; s < N_SUBSTRATES; s++ )
            out[(size_t)s*nv + v] = M(v)[s];
}

static double compare( const std::vector<double>& a, const std::vector<double>& b,
                       double& max_rel )
{
    double max_abs = 0.0; max_rel = 0.0;
    for( size_t i = 0; i < a.size(); i++ )
    {
        double d = std::fabs( a[i]-b[i] );
        double r = d / (std::fabs(b[i]) + 1e-300);
        if( d > max_abs ) max_abs = d;
        if( r > max_rel ) max_rel = r;
    }
    return max_abs;
}

// ---- Test 1: diffusion only ------------------------------------------------
static int test_diffusion_only()
{
    auto run = [&]( void(*solver)(Microenvironment&,double), std::vector<double>& out ){
        Microenvironment M; M.name="t1"; config_densities(M);
        M.resize_space_uniform( 0,(double)NX, 0,(double)NY, 0,(double)NZ, 1.0 );
        init_field(M);
        M.diffusion_decay_solver = solver;
        for( int s=0;s<N_STEPS;s++) M.simulate_diffusion_decay(DT);
        extract(M,out);
    };
    std::vector<double> f_cpu, f_gpu;
    run( diffusion_decay_solver__constant_coefficients_LOD_3D,     f_cpu );
    run( diffusion_decay_solver__constant_coefficients_LOD_3D_GPU, f_gpu );
    double max_rel; double max_abs = compare( f_gpu, f_cpu, max_rel );
    std::printf( "[Test1 diffusion-only ] max abs %.3e  max rel %.3e  -> %s\n",
                 max_abs, max_rel, (max_abs<=1e-9)?"PASS":"FAIL" );
    return max_abs<=1e-9 ? 0 : 1;
}

// ---- Test 2: diffusion + secretion (resident) ------------------------------
static void place_agents( Microenvironment& M )
{
    std::srand( 7 );
    for( int a=0;a<N_AGENTS;a++)
    {
        Basic_Agent* ag = create_basic_agent();
        ag->register_microenvironment( &M );
        double x = (double)(std::rand()% NX) + 0.5;
        double y = (double)(std::rand()% NY) + 0.5;
        double z = (double)(std::rand()% NZ) + 0.5;
        ag->assign_position( x, y, z );
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

static int test_diffusion_and_secretion()
{
    // BioFVM's CPU secretion is an unordered omp-parallel-for over cells, so when two
    // cells share a voxel the result depends on thread scheduling — i.e. the CPU path
    // is NOT deterministic against itself (verified: ~3e-3 run-to-run with collisions).
    // For an apples-to-apples comparison we force single-threaded secretion on the CPU
    // reference so both sides apply cells in index order, exactly like the GPU kernel
    // (thread c == agent c). The numerics are then identical; the only difference in the
    // parallel case is collision-ordering, which is a property of BioFVM, not the GPU port.
#ifdef _OPENMP
    omp_set_num_threads( 1 );
#endif
    auto run = [&]( bool use_gpu_combined, std::vector<double>& out ){
        all_basic_agents.clear(); reset_max_basic_agent_ID();
        Microenvironment M; M.name="t2"; config_densities(M);
        M.resize_space_uniform( 0,(double)NX, 0,(double)NY, 0,(double)NZ, 1.0 );
        set_default_microenvironment( &M );
        init_field(M);
        place_agents(M);
        M.diffusion_decay_solver = use_gpu_combined
            ? diffusion_decay_solver__constant_coefficients_LOD_3D_GPU
            : diffusion_decay_solver__constant_coefficients_LOD_3D;
        for( int s=0;s<N_STEPS;s++)
        {
            if( use_gpu_combined )
                M.simulate_diffusion_and_secretion_gpu( DT );
            else
            {
                M.simulate_diffusion_decay( DT );
                M.simulate_cell_sources_and_sinks( DT );
            }
        }
        extract(M,out);
    };
    std::vector<double> f_cpu, f_gpu;
    run( false, f_cpu );
    run( true,  f_gpu );
    double max_rel; double max_abs = compare( f_gpu, f_cpu, max_rel );
    std::printf( "[Test2 diff+secretion ] max abs %.3e  max rel %.3e  -> %s\n",
                 max_abs, max_rel, (max_abs<=1e-9)?"PASS":"FAIL" );
    return max_abs<=1e-9 ? 0 : 1;
}

// ---- Test 3: interleaved host reads (exercises lazy device->host download) --
// The zero-transfer path keeps the field on the device and only downloads on a host
// read. This test reads the field mid-simulation (every few steps, like sensing) to
// prove the lazy download fires correctly and a read does not corrupt the resident
// state for subsequent steps.
static int test_interleaved_reads()
{
#ifdef _OPENMP
    omp_set_num_threads( 1 );
#endif
    auto run = [&]( bool use_gpu_combined, std::vector<double>& out ){
        all_basic_agents.clear(); reset_max_basic_agent_ID();
        Microenvironment M; M.name="t3"; config_densities(M);
        M.resize_space_uniform( 0,(double)NX, 0,(double)NY, 0,(double)NZ, 1.0 );
        set_default_microenvironment( &M );
        init_field(M);
        place_agents(M);
        M.diffusion_decay_solver = use_gpu_combined
            ? diffusion_decay_solver__constant_coefficients_LOD_3D_GPU
            : diffusion_decay_solver__constant_coefficients_LOD_3D;
        volatile double sink = 0.0;
        for( int s=0;s<N_STEPS;s++)
        {
            if( use_gpu_combined ) M.simulate_diffusion_and_secretion_gpu( DT );
            else { M.simulate_diffusion_decay( DT ); M.simulate_cell_sources_and_sinks( DT ); }
            // every 5th step, read the field (forces a device->host sync in GPU mode)
            if( s % 5 == 0 )
                for( int v = 0; v < (int)M.mesh.voxels.size(); v += 37 )
                    sink += M(v)[0];
        }
        (void)sink;
        extract(M,out);
    };
    std::vector<double> f_cpu, f_gpu;
    run( false, f_cpu );
    run( true,  f_gpu );
    double max_rel; double max_abs = compare( f_gpu, f_cpu, max_rel );
    std::printf( "[Test3 interleaved-read] max abs %.3e  max rel %.3e  -> %s\n",
                 max_abs, max_rel, (max_abs<=1e-9)?"PASS":"FAIL" );
    return max_abs<=1e-9 ? 0 : 1;
}

int main()
{
    std::printf( "backend            : %s\n", gpu_backend_is_cuda() ? "CUDA" : "CPU-fallback" );
    std::printf( "grid %dx%dx%d, %d substrates, %d steps, %d agents\n",
                 NX,NY,NZ,N_SUBSTRATES,N_STEPS,N_AGENTS );
    int rc = 0;
    rc |= test_diffusion_only();
    rc |= test_diffusion_and_secretion();
    rc |= test_interleaved_reads();
    std::printf( "RESULT: %s\n", rc==0 ? "PASS" : "FAIL" );
    return rc;
}
