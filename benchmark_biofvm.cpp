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

using namespace BioFVM;

// ── tuneable parameters ────────────────────────────────────────────────
static constexpr int    N_STEPS    = 500;      // diffusion timesteps
static constexpr double DT         = 0.01;     // minutes
static constexpr int    N_SUBSTRATES = 5;
static constexpr int    N_AGENTS   = 300;
static constexpr int    NX = 64, NY = 64, NZ = 1;
// ───────────────────────────────────────────────────────────────────────

int main( int argc, char** argv )
{
    // ── build microenvironment ──────────────────────────────────────────
    Microenvironment M;
    M.name = "benchmark_env";
    M.set_density( 0, "anti_tumoral_factor", "dimensionless" );
    M.diffusion_coefficients[0] = 600.0;
    M.decay_rates[0]            = 0.1;

    for( int s = 1; s < N_SUBSTRATES; s++ )
    {
        M.add_density( "substrate_" + std::to_string(s), "dimensionless" );
        M.diffusion_coefficients[s] = 100.0 + s * 50.0;
        M.decay_rates[s]            = 0.01 * s;
    }

    M.resize_space_uniform( 0.0, (double)NX, 0.0, (double)NY, -0.5, 0.5, 1.0 );
    M.spatial_units = "micron";
    M.time_units    = "min";

    // flat initial concentration of 1 everywhere
    for( int v = 0; v < (int)M.mesh.voxels.size(); v++ )
        for( int s = 0; s < N_SUBSTRATES; s++ )
            M(v)[s] = 1.0;

    M.build_diffusion_adjacency_list();
    M.set_substrate_dirichlet_to_default();
    M.display_information( std::cout );

    // ── place agents ────────────────────────────────────────────────────
    std::vector<Basic_Agent*> agents;
    agents.reserve( N_AGENTS );
    srand( 42 );
    for( int a = 0; a < N_AGENTS; a++ )
    {
        Basic_Agent* ag = create_basic_agent();
        double x = (double)(rand() % NX) + 0.5;
        double y = (double)(rand() % NY) + 0.5;
        double z = 0.0;
        ag->register_microenvironment( &M );
        ag->assign_position( x, y, z );
        ag->set_internal_uptake_constants( DT );
        agents.push_back( ag );
    }

    // ── warm-up (1 step) ────────────────────────────────────────────────
    M.simulate_diffusion_decay( DT );
    M.simulate_cell_sources_and_sinks( DT );

    // ── timed loop ──────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    for( int step = 0; step < N_STEPS; step++ )
    {
        M.simulate_diffusion_decay( DT );
        M.simulate_cell_sources_and_sinks( DT );
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>( t1 - t0 ).count();
    double per_step = elapsed / N_STEPS * 1e3; // ms

    std::printf( "\n========================================\n" );
    std::printf( "  BioFVM benchmark results\n" );
    std::printf( "  Domain : %dx%dx%d, dx=1 µm\n", NX, NY, NZ );
    std::printf( "  Substrates : %d\n", N_SUBSTRATES );
    std::printf( "  Agents : %d\n", N_AGENTS );
    std::printf( "  Steps : %d  (dt=%.4f min)\n", N_STEPS, DT );
    std::printf( "----------------------------------------\n" );
    std::printf( "  Total   : %.4f s\n", elapsed );
    std::printf( "  Per step: %.4f ms\n", per_step );
    std::printf( "========================================\n\n" );

    return 0;
}
