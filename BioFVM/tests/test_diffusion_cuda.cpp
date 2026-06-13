/*
###############################################################################
# test_diffusion_cuda.cpp
#
# Standalone correctness test for the dual-backend GPU-resident LOD solver.
#
# It builds Thomas coefficients exactly as BioFVM does for the
# constant-coefficient 3D LOD scheme, runs one diffusion-decay step through:
#   (a) a straightforward reference Thomas LOD solver written inline here, and
#   (b) the BioFVM::gpu_* backend (CPU fallback today, CUDA when built w/ nvcc),
# then asserts the two density fields agree to a tight tolerance.
#
# Build (CPU fallback, no GPU needed):
#   make test-cuda      (see Makefile target) OR:
#   g++ -O3 -fopenmp -std=c++11 -I.. \
#       ../BioFVM_diffusion_cuda.cu test_diffusion_cuda.cpp -o test_diffusion_cuda
#
# Build (GPU):
#   nvcc -O3 -D BioFVM_USE_CUDA -x cu -I.. -c ../BioFVM_diffusion_cuda.cu -o cuda.o
#   nvcc -O3 -D BioFVM_USE_CUDA -I.. cuda.o test_diffusion_cuda.cpp -o test_diffusion_cuda
###############################################################################
*/

#include "../BioFVM_diffusion_cuda.h"

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace BioFVM;

// Build the LOD Thomas coefficients for a regular mesh with given dt, D, lambda.
// Mirrors diffusion_decay_solver__constant_coefficients_LOD_3D setup.
static void build_params( gpu_solver_params& P,
                          unsigned int nx, unsigned int ny, unsigned int nz,
                          const std::vector<double>& D,
                          const std::vector<double>& lambda,
                          double dx, double dt )
{
    const unsigned int ns = D.size();
    P.nx=nx; P.ny=ny; P.nz=nz; P.ns=ns; P.nv=nx*ny*nz;
    P.c1.resize(ns);
    P.denom_x.resize(nx*ns); P.cx.resize(nx*ns);
    P.denom_y.resize(ny*ns); P.cy.resize(ny*ns);
    P.denom_z.resize(nz*ns); P.cz.resize(nz*ns);

    for( unsigned int s=0; s<ns; ++s )
    {
        const double c1  = dt*D[s]/(dx*dx);          // constant1
        const double c2  = dt*lambda[s]/3.0;          // constant2 (split 3 ways)
        const double c3  = 1.0 + 2.0*c1 + c2;          // interior diagonal
        const double c3a = 1.0 + c1 + c2;              // boundary diagonal
        P.c1[s] = c1;

        if( D[s] == 0.0 )
        {
            P.nodiff_subs.push_back(s);
            const double f = 1.0 + c2;
            P.nodiff_decay.push_back( 1.0/(f*f*f) );
            continue;
        }
        P.diff_subs.push_back(s);

        // one axis builder, reused for x/y/z (same n? no -> per axis)
        auto build_axis = [&]( unsigned int n, std::vector<double>& denom, std::vector<double>& c )
        {
            for( unsigned int t=0; t<n; ++t ){ denom[t*ns+s] = c3; c[t*ns+s] = -c1; }
            denom[0*ns+s] = c3a; denom[(n-1)*ns+s] = c3a;
            c[0*ns+s] /= denom[0*ns+s];
            for( unsigned int t=1; t<n; ++t )
            {
                denom[t*ns+s] += c1 * c[(t-1)*ns+s];
                c[t*ns+s] /= denom[t*ns+s];
            }
        };
        build_axis( nx, P.denom_x, P.cx );
        build_axis( ny, P.denom_y, P.cy );
        build_axis( nz, P.denom_z, P.cz );
    }
}

// Reference: independent, simple Thomas LOD over an AoS-ish flat array indexed
// [s*nv + v] with v=i+nx*j+nx*ny*k. Strided, unoptimized, easy to trust.
static void reference_solve( const gpu_solver_params& P, std::vector<double>& f )
{
    const unsigned int nx=P.nx,ny=P.ny,nz=P.nz,ns=P.ns,nv=P.nv;
    const int ij=1, jj=(int)nx, kj=(int)nx*(int)ny;
    auto idx=[&](unsigned int i,unsigned int j,unsigned int k){ return (size_t)i*ij + (size_t)j*jj + (size_t)k*kj; };

    // decay
    for( unsigned int si=0; si<P.nodiff_subs.size(); ++si ){ unsigned int s=P.nodiff_subs[si];
        for( unsigned int v=0; v<nv; ++v ) f[s*nv+v] *= P.nodiff_decay[si]; }

    for( unsigned int si=0; si<P.diff_subs.size(); ++si )
    {
        unsigned int s = P.diff_subs[si];
        const double c1 = P.c1[s];
        // x
        for( unsigned int k=0;k<nz;++k) for( unsigned int j=0;j<ny;++j){
            size_t n0=idx(0,j,k); f[s*nv+n0]/=P.denom_x[0*ns+s];
            for( unsigned int i=1;i<nx;++i){ size_t n=n0+(size_t)i*ij;
                f[s*nv+n]=(f[s*nv+n]+c1*f[s*nv+n-ij])/P.denom_x[i*ns+s]; }
            for( int i=(int)nx-2;i>=0;--i){ size_t n=n0+(size_t)i*ij;
                f[s*nv+n]-=P.cx[i*ns+s]*f[s*nv+n+ij]; } }
        // y
        for( unsigned int k=0;k<nz;++k) for( unsigned int i=0;i<nx;++i){
            size_t n0=idx(i,0,k); f[s*nv+n0]/=P.denom_y[0*ns+s];
            for( unsigned int j=1;j<ny;++j){ size_t n=n0+(size_t)j*jj;
                f[s*nv+n]=(f[s*nv+n]+c1*f[s*nv+n-jj])/P.denom_y[j*ns+s]; }
            for( int j=(int)ny-2;j>=0;--j){ size_t n=n0+(size_t)j*jj;
                f[s*nv+n]-=P.cy[j*ns+s]*f[s*nv+n+jj]; } }
        // z
        for( unsigned int j=0;j<ny;++j) for( unsigned int i=0;i<nx;++i){
            size_t n0=idx(i,j,0); f[s*nv+n0]/=P.denom_z[0*ns+s];
            for( unsigned int k=1;k<nz;++k){ size_t n=n0+(size_t)k*kj;
                f[s*nv+n]=(f[s*nv+n]+c1*f[s*nv+n-kj])/P.denom_z[k*ns+s]; }
            for( int k=(int)nz-2;k>=0;--k){ size_t n=n0+(size_t)k*kj;
                f[s*nv+n]-=P.cz[k*ns+s]*f[s*nv+n+kj]; } }
    }
}

int main()
{
    const unsigned int nx=23, ny=17, nz=19;       // deliberately non-cubic & odd
    std::vector<double> D     = { 1000.0, 0.0, 100.0 }; // substrate 1 is non-diffusing
    std::vector<double> lam   = {    0.1, 0.5,   0.0 };
    const double dx=20.0, dt=0.01;
    const unsigned int ns=D.size(), nv=nx*ny*nz;

    gpu_solver_params P;
    build_params( P, nx,ny,nz, D, lam, dx, dt );

    // random initial field
    std::vector<double> f0( (size_t)nv*ns );
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u(0.0, 10.0);
    for( auto& x : f0 ) x = u(rng);

    std::vector<double> f_ref = f0;
    reference_solve( P, f_ref );

    gpu_field* g = gpu_field_alloc( P );
    gpu_upload( g, f0.data() );
    gpu_solve_3D_LOD( g );
    std::vector<double> f_gpu( (size_t)nv*ns );
    gpu_download( g, f_gpu.data() );
    gpu_field_free( g );

    double max_abs=0.0, max_rel=0.0;
    for( size_t i=0;i<f_ref.size();++i )
    {
        double a=std::fabs(f_gpu[i]-f_ref[i]);
        double r=a/(std::fabs(f_ref[i])+1e-300);
        if(a>max_abs)max_abs=a; if(r>max_rel)max_rel=r;
    }
    std::printf( "backend            : %s\n", gpu_backend_is_cuda()?"CUDA":"CPU-fallback" );
    std::printf( "grid               : %ux%ux%u, %u substrates (%zu diffusing)\n",
                 nx,ny,nz,ns,P.diff_subs.size() );
    std::printf( "max abs difference : %.3e\n", max_abs );
    std::printf( "max rel difference : %.3e\n", max_rel );

    const double tol = 1e-10;
    if( max_abs > tol )
    {
        std::printf( "RESULT: FAIL (exceeds tol %.1e)\n", tol );
        return 1;
    }
    std::printf( "RESULT: PASS\n" );
    return 0;
}
