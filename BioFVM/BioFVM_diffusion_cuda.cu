/*
###############################################################################
# BioFVM_diffusion_cuda.cu
#
# Dual-backend implementation of the GPU-resident 3D LOD diffusion solver.
# See BioFVM_diffusion_cuda.h for the design rationale.
#
# Compile either way:
#   nvcc -x cu -D BioFVM_USE_CUDA -c BioFVM_diffusion_cuda.cu     (GPU build)
#   g++  -x c++ -fopenmp        -c BioFVM_diffusion_cuda.cu       (CPU fallback)
#
# The two paths implement the *same* numerical algorithm:
#   - per-substrate scalar decay for non-diffusing densities (D==0)
#   - three sequential 1D implicit Thomas sweeps (x, then y, then z) for the
#     diffusing densities, using precomputed forward/back coefficients.
# The CPU fallback strides through SoA memory exactly like the current solver in
# BioFVM_solvers.cpp, so results are identical and can be validated without a
# GPU. The CUDA path transposes the grid between sweeps so that every Thomas
# solve reads a contiguous run (coalesced access) on the device.
###############################################################################
*/

#include "BioFVM_diffusion_cuda.h"

#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>

// CUDA runtime must be included at global scope (not inside namespace BioFVM),
// otherwise <cstdio>/<cuda_runtime.h> names land under BioFVM::std::* and the
// CUDA_CHECK macro's std::fprintf/std::abort fail to resolve.
#if defined(__CUDACC__) && defined(BioFVM_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace BioFVM {

// --- shared transfer counters (both backends) -------------------------------
static unsigned long long g_upload_count = 0;
static unsigned long long g_download_count = 0;
static unsigned long long g_transfer_bytes = 0;
unsigned long long gpu_field_upload_count()   { return g_upload_count; }
unsigned long long gpu_field_download_count() { return g_download_count; }
unsigned long long gpu_field_transfer_bytes() { return g_transfer_bytes; }
void gpu_reset_transfer_counters()
{ g_upload_count = g_download_count = g_transfer_bytes = 0; }

// =============================================================================
//  CUDA BACKEND
// =============================================================================
#if defined(__CUDACC__) && defined(BioFVM_USE_CUDA)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                              \
        if( _e != cudaSuccess ) {                                             \
            std::fprintf( stderr, "[BioFVM-CUDA] %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(_e) );                \
            std::abort();                                                     \
        }                                                                      \
    } while(0)

struct gpu_field
{
    gpu_solver_params p;

    // Authoritative density buffer on device, SoA: d_soa[s*nv + v].
    double* d_soa = nullptr;
    // Scratch buffer of equal size, used as the destination of transposes.
    double* d_tmp = nullptr;

    // Coefficient arrays on device (flattened [coord*ns + s]).
    double* d_c1 = nullptr;
    double* d_denom_x = nullptr; double* d_cx = nullptr;
    double* d_denom_y = nullptr; double* d_cy = nullptr;
    double* d_denom_z = nullptr; double* d_cz = nullptr;

    unsigned int* d_diff_subs = nullptr;   unsigned int nd = 0;
    unsigned int* d_nodiff_subs = nullptr; unsigned int nn = 0;
    double*       d_nodiff_decay = nullptr;

    char* d_dirichlet = nullptr; // length nv, or null if none

    // Secretion batch scratch (grown on demand; capacity in cells).
    int*    d_sec_voxel = nullptr;
    double* d_sec_t1 = nullptr;
    double* d_sec_t2 = nullptr;
    double* d_sec_ex = nullptr;
    unsigned int sec_capacity = 0;
};

static double* dev_alloc_copy( const std::vector<double>& v )
{
    if( v.empty() ) return nullptr;
    double* d = nullptr;
    CUDA_CHECK( cudaMalloc( &d, v.size()*sizeof(double) ) );
    CUDA_CHECK( cudaMemcpy( d, v.data(), v.size()*sizeof(double), cudaMemcpyHostToDevice ) );
    return d;
}
static unsigned int* dev_alloc_copy_u( const std::vector<unsigned int>& v )
{
    if( v.empty() ) return nullptr;
    unsigned int* d = nullptr;
    CUDA_CHECK( cudaMalloc( &d, v.size()*sizeof(unsigned int) ) );
    CUDA_CHECK( cudaMemcpy( d, v.data(), v.size()*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
    return d;
}

gpu_field* gpu_field_alloc( const gpu_solver_params& params )
{
    gpu_field* g = new gpu_field();
    g->p = params;
    const size_t bytes = (size_t)params.nv * params.ns * sizeof(double);
    CUDA_CHECK( cudaMalloc( &g->d_soa, bytes ) );
    CUDA_CHECK( cudaMalloc( &g->d_tmp, bytes ) );

    g->d_c1      = dev_alloc_copy( params.c1 );
    g->d_denom_x = dev_alloc_copy( params.denom_x ); g->d_cx = dev_alloc_copy( params.cx );
    g->d_denom_y = dev_alloc_copy( params.denom_y ); g->d_cy = dev_alloc_copy( params.cy );
    g->d_denom_z = dev_alloc_copy( params.denom_z ); g->d_cz = dev_alloc_copy( params.cz );

    g->d_diff_subs   = dev_alloc_copy_u( params.diff_subs );   g->nd = params.diff_subs.size();
    g->d_nodiff_subs = dev_alloc_copy_u( params.nodiff_subs ); g->nn = params.nodiff_subs.size();
    g->d_nodiff_decay = dev_alloc_copy( params.nodiff_decay );
    return g;
}

void gpu_field_free( gpu_field* g )
{
    if( !g ) return;
    cudaFree( g->d_soa );  cudaFree( g->d_tmp );
    cudaFree( g->d_c1 );
    cudaFree( g->d_denom_x ); cudaFree( g->d_cx );
    cudaFree( g->d_denom_y ); cudaFree( g->d_cy );
    cudaFree( g->d_denom_z ); cudaFree( g->d_cz );
    cudaFree( g->d_diff_subs ); cudaFree( g->d_nodiff_subs ); cudaFree( g->d_nodiff_decay );
    cudaFree( g->d_dirichlet );
    cudaFree( g->d_sec_voxel ); cudaFree( g->d_sec_t1 ); cudaFree( g->d_sec_t2 ); cudaFree( g->d_sec_ex );
    delete g;
}

void gpu_upload( gpu_field* g, const double* host_soa )
{
    const size_t bytes = (size_t)g->p.nv * g->p.ns * sizeof(double);
    CUDA_CHECK( cudaMemcpy( g->d_soa, host_soa, bytes, cudaMemcpyHostToDevice ) );
    g_upload_count++; g_transfer_bytes += bytes;
}
void gpu_download( gpu_field* g, double* host_soa )
{
    const size_t bytes = (size_t)g->p.nv * g->p.ns * sizeof(double);
    CUDA_CHECK( cudaMemcpy( host_soa, g->d_soa, bytes, cudaMemcpyDeviceToHost ) );
    g_download_count++; g_transfer_bytes += bytes;
}
void gpu_set_dirichlet( gpu_field* g, const char* host_mask )
{
    if( !host_mask ) { if(g->d_dirichlet){cudaFree(g->d_dirichlet); g->d_dirichlet=nullptr;} return; }
    if( !g->d_dirichlet ) CUDA_CHECK( cudaMalloc( &g->d_dirichlet, g->p.nv ) );
    CUDA_CHECK( cudaMemcpy( g->d_dirichlet, host_mask, g->p.nv, cudaMemcpyHostToDevice ) );
}

// --- secretion / uptake -----------------------------------------------------

// One thread per cell. Applies rho=(rho+t1)/t2+ex at the cell's voxel for every
// substrate, directly on the resident SoA field. Mirrors the CPU agent loop,
// which is itself an unordered omp-parallel-for over cells (so cells sharing a
// voxel race identically here — matching existing BioFVM semantics).
__global__ void k_secretion( double* soa, unsigned int nv, unsigned int ns,
                             const int* voxel, const double* t1,
                             const double* t2, const double* ex,
                             unsigned int n_cells )
{
    unsigned int c = blockIdx.x*blockDim.x + threadIdx.x;
    if( c >= n_cells ) return;
    int v = voxel[c];
    if( v < 0 ) return;                 // inactive / out-of-domain cell
    double* base = soa + (unsigned int)v;
    const double* c1 = t1 + (size_t)c*ns;
    const double* c2 = t2 + (size_t)c*ns;
    const double* ce = ex + (size_t)c*ns;
    for( unsigned int s=0; s<ns; ++s )
        base[(size_t)s*nv] = ( base[(size_t)s*nv] + c1[s] ) / c2[s] + ce[s];
}

static void ensure_sec_capacity( gpu_field* g, unsigned int n, unsigned int ns )
{
    if( n <= g->sec_capacity ) return;
    cudaFree( g->d_sec_voxel ); cudaFree( g->d_sec_t1 );
    cudaFree( g->d_sec_t2 );    cudaFree( g->d_sec_ex );
    CUDA_CHECK( cudaMalloc( &g->d_sec_voxel, (size_t)n*sizeof(int) ) );
    CUDA_CHECK( cudaMalloc( &g->d_sec_t1, (size_t)n*ns*sizeof(double) ) );
    CUDA_CHECK( cudaMalloc( &g->d_sec_t2, (size_t)n*ns*sizeof(double) ) );
    CUDA_CHECK( cudaMalloc( &g->d_sec_ex, (size_t)n*ns*sizeof(double) ) );
    g->sec_capacity = n;
}

void gpu_apply_secretion( gpu_field* g, const gpu_secretion_batch& b )
{
    const unsigned int n = b.n_cells, ns = g->p.ns;
    if( n == 0 ) return;
    ensure_sec_capacity( g, n, ns );
    // Async H2D on the default stream: the kernel below and any later download all
    // run on stream 0, so they serialize after these copies without a host-side wait.
    const size_t dn = (size_t)n*ns*sizeof(double);
    CUDA_CHECK( cudaMemcpyAsync( g->d_sec_voxel, b.voxel.data(), (size_t)n*sizeof(int), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpyAsync( g->d_sec_t1, b.temp1.data(), dn, cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpyAsync( g->d_sec_t2, b.temp2.data(), dn, cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpyAsync( g->d_sec_ex, b.temp_export2.data(), dn, cudaMemcpyHostToDevice ) );
    const unsigned int B = 128;
    k_secretion<<< (n+B-1)/B, B >>>( g->d_soa, g->p.nv, ns,
        g->d_sec_voxel, g->d_sec_t1, g->d_sec_t2, g->d_sec_ex, n );
    CUDA_CHECK( cudaGetLastError() );
#ifdef BIOFVM_SEC_SYNC
    CUDA_CHECK( cudaDeviceSynchronize() );
#endif
    // No cudaDeviceSynchronize here (unless BIOFVM_SEC_SYNC, for A/B): secretion is the
    // last device op of the resident step and its result is consumed only by the next
    // same-stream solve or by a syncing gpu_download before a host read. Returning lets
    // the host pack the next batch while the kernel runs.
}

// --- kernels ----------------------------------------------------------------

// Scalar decay for non-diffusing substrates: one thread per (voxel).
__global__ void k_decay( double* soa, unsigned int nv,
                         const unsigned int* nodiff, unsigned int nn,
                         const double* decay )
{
    unsigned int v = blockIdx.x*blockDim.x + threadIdx.x;
    if( v >= nv ) return;
    for( unsigned int si=0; si<nn; ++si )
        soa[ nodiff[si]*nv + v ] *= decay[si];
}

// Generic 1D Thomas sweep along the fastest (contiguous) axis of a layout where
// each "line" of length `n` is contiguous. One thread handles one (line,substrate)
// pair. `lines` = number of independent solves per substrate. Buffer layout is
// soa[s*nv + line*n + t], so a thread walks t=0..n-1 with stride 1.
// denom/c are flattened [t*ns + s].
__global__ void k_thomas_axis( double* soa, unsigned int nv, unsigned int ns,
                               unsigned int n, unsigned int lines,
                               const double* denom, const double* c,
                               const double* c1,
                               const unsigned int* diff, unsigned int nd )
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int total = lines * nd;
    if( tid >= total ) return;
    unsigned int line = tid / nd;
    unsigned int s    = diff[ tid % nd ];
    double* col = soa + (size_t)s*nv + (size_t)line*n;
    const double cs = c1[s];

    // forward elimination
    col[0] /= denom[ 0*ns + s ];
    for( unsigned int t=1; t<n; ++t )
        col[t] = ( col[t] + cs*col[t-1] ) / denom[ t*ns + s ];
    // back substitution
    for( int t=(int)n-2; t>=0; --t )
        col[t] -= c[ t*ns + s ] * col[(size_t)t+1];
}

// General axis-permuting transpose. The source buffer is stored so that the
// three logical axes (x,y,z) appear in slot order (slot0 = contiguous/fastest,
// then slot1, then slot2). `e0,e1,e2` are the EXTENTS of slots 0,1,2 and
// `ax0,ax1,ax2` say which logical axis (0=x,1=y,2=z) each slot is. The
// destination is always written in a layout chosen by the *destination* slot
// order given via the same (ax/e) describing dst — but to keep call sites
// unambiguous we always write CANONICAL output here: dst is laid out with
// dst-slot0..2 extents (de0,de1,de2) for logical axes (dax0,dax1,dax2).
// One thread per source voxel; gathers all substrates.
__global__ void k_permute( const double* src, double* dst,
                           unsigned int nv, unsigned int ns,
                           unsigned int e0, unsigned int e1, unsigned int e2,
                           int ax0, int ax1, int ax2,
                           unsigned int de0, unsigned int de1, unsigned int de2,
                           int dax0, int dax1, int dax2 )
{
    unsigned int v = blockIdx.x*blockDim.x + threadIdx.x;
    if( v >= nv ) return;
    // decode source slot indices
    unsigned int i0 = v % e0;
    unsigned int i1 = (v / e0) % e1;
    unsigned int i2 = v / ((size_t)e0*e1);
    // map to logical (x,y,z)
    unsigned int lx=0, ly=0, lz=0;
    unsigned int si[3] = { i0, i1, i2 };
    int          sa[3] = { ax0, ax1, ax2 };
    for( int k=0;k<3;++k ){ if(sa[k]==0) lx=si[k]; else if(sa[k]==1) ly=si[k]; else lz=si[k]; }
    // encode into destination slot order
    unsigned int dl[3]; // logical index per dst slot
    int          da[3] = { dax0, dax1, dax2 };
    for( int k=0;k<3;++k ){ dl[k] = (da[k]==0)?lx : (da[k]==1)?ly : lz; }
    unsigned int out = dl[0] + de0*( dl[1] + (size_t)de1*dl[2] );
    (void)de2;
    for( unsigned int s=0; s<ns; ++s )
        dst[ (size_t)s*nv + out ] = src[ (size_t)s*nv + v ];
}

__global__ void k_apply_dirichlet( double* /*soa*/, const char* /*mask*/, unsigned int /*nv*/ )
{
    // Dirichlet values are re-imposed on host between solves in this PoC; the
    // device hook is present for the GPU-resident extension. Intentionally a
    // no-op placeholder so the call site is stable.
}

static inline unsigned int grid_for( unsigned int total, unsigned int block )
{ return (total + block - 1) / block; }

void gpu_solve_3D_LOD( gpu_field* g )
{
    const unsigned int nx=g->p.nx, ny=g->p.ny, nz=g->p.nz, ns=g->p.ns, nv=g->p.nv;
    const unsigned int B = 128;

    // 1) scalar decay for non-diffusing substrates
    if( g->nn > 0 )
        k_decay<<< grid_for(nv,B), B >>>( g->d_soa, nv, g->d_nodiff_subs, g->nn, g->d_nodiff_decay );

    if( g->nd > 0 )
    {
        // Layout notation: slots are (contiguous, mid, slow); each tagged with its
        // logical axis 0=x,1=y,2=z. Natural L0 = slots(x,y,z) extents(nx,ny,nz).

        // --- x-sweep in L0: x contiguous. lines = ny*nz, length nx ---
        k_thomas_axis<<< grid_for(ny*nz*g->nd, B), B >>>(
            g->d_soa, nv, ns, nx, ny*nz, g->d_denom_x, g->d_cx, g->d_c1, g->d_diff_subs, g->nd );

        // L0 (x,y,z) -> Ly (y,x,z): make y contiguous
        k_permute<<< grid_for(nv,B), B >>>( g->d_soa, g->d_tmp, nv, ns,
            nx,ny,nz, 0,1,2,   ny,nx,nz, 1,0,2 );
        // --- y-sweep on d_tmp: y contiguous. lines = nx*nz, length ny ---
        k_thomas_axis<<< grid_for(nx*nz*g->nd, B), B >>>(
            g->d_tmp, nv, ns, ny, nx*nz, g->d_denom_y, g->d_cy, g->d_c1, g->d_diff_subs, g->nd );
        // Ly (y,x,z) -> L0 (x,y,z)
        k_permute<<< grid_for(nv,B), B >>>( g->d_tmp, g->d_soa, nv, ns,
            ny,nx,nz, 1,0,2,   nx,ny,nz, 0,1,2 );

        // L0 (x,y,z) -> Lz (z,x,y): make z contiguous
        k_permute<<< grid_for(nv,B), B >>>( g->d_soa, g->d_tmp, nv, ns,
            nx,ny,nz, 0,1,2,   nz,nx,ny, 2,0,1 );
        // --- z-sweep on d_tmp: z contiguous. lines = nx*ny, length nz ---
        k_thomas_axis<<< grid_for(nx*ny*g->nd, B), B >>>(
            g->d_tmp, nv, ns, nz, nx*ny, g->d_denom_z, g->d_cz, g->d_c1, g->d_diff_subs, g->nd );
        // Lz (z,x,y) -> L0 (x,y,z)
        k_permute<<< grid_for(nv,B), B >>>( g->d_tmp, g->d_soa, nv, ns,
            nz,nx,ny, 2,0,1,   nx,ny,nz, 0,1,2 );
    }
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );
}

bool gpu_backend_is_cuda() { return true; }

// =============================================================================
//  CPU FALLBACK  (no CUDA toolkit required)
// =============================================================================
#else

#ifdef _OPENMP
#include <omp.h>
#endif

// On the CPU there is no separate device memory: the "device" buffer is host
// memory, and the solve strides through SoA exactly like the production solver
// in BioFVM_solvers.cpp so that results are bit-for-bit identical. No transpose
// is performed (CPU caches make strided access acceptable); the transpose is a
// pure GPU-coalescing optimization and does not change the math.
struct gpu_field
{
    gpu_solver_params p;
    std::vector<double> soa;   // length nv*ns, SoA [s*nv+v]
    std::vector<char>   dirichlet; // length nv (or empty)
};

gpu_field* gpu_field_alloc( const gpu_solver_params& params )
{
    gpu_field* g = new gpu_field();
    g->p = params;
    g->soa.assign( (size_t)params.nv * params.ns, 0.0 );
    return g;
}
void gpu_field_free( gpu_field* g ) { delete g; }

void gpu_upload( gpu_field* g, const double* host_soa )
{
    const size_t bytes = g->soa.size()*sizeof(double);
    std::memcpy( g->soa.data(), host_soa, bytes );
    g_upload_count++; g_transfer_bytes += bytes;
}
void gpu_download( gpu_field* g, double* host_soa )
{
    const size_t bytes = g->soa.size()*sizeof(double);
    std::memcpy( host_soa, g->soa.data(), bytes );
    g_download_count++; g_transfer_bytes += bytes;
}
void gpu_set_dirichlet( gpu_field* g, const char* host_mask )
{
    if( !host_mask ) { g->dirichlet.clear(); return; }
    g->dirichlet.assign( host_mask, host_mask + g->p.nv );
}

void gpu_apply_secretion( gpu_field* g, const gpu_secretion_batch& b )
{
    const unsigned int n = b.n_cells, ns = g->p.ns, nv = g->p.nv;
    if( n == 0 ) return;
    double* __restrict__ soa = g->soa.data();
    const int*    __restrict__ vox = b.voxel.data();
    const double* __restrict__ t1  = b.temp1.data();
    const double* __restrict__ t2  = b.temp2.data();
    const double* __restrict__ ex  = b.temp_export2.data();
    // Same unordered parallelism over cells as the production loop.
    #pragma omp parallel for
    for( unsigned int c=0; c<n; ++c )
    {
        int v = vox[c];
        if( v < 0 ) continue;
        double* base = soa + (unsigned int)v;
        const double* c1 = t1 + (size_t)c*ns;
        const double* c2 = t2 + (size_t)c*ns;
        const double* ce = ex + (size_t)c*ns;
        for( unsigned int s=0; s<ns; ++s )
            base[(size_t)s*nv] = ( base[(size_t)s*nv] + c1[s] ) / c2[s] + ce[s];
    }
}

void gpu_solve_3D_LOD( gpu_field* g )
{
    const gpu_solver_params& P = g->p;
    const unsigned int nx=P.nx, ny=P.ny, nz=P.nz, ns=P.ns, nv=P.nv;
    const int ijump = 1;
    const int jjump = (int)nx;
    const int kjump = (int)nx*(int)ny;
    double* __restrict__ soa = g->soa.data();

    const double* __restrict__ c1 = P.c1.data();
    const double* __restrict__ dx = P.denom_x.data();
    const double* __restrict__ cxf= P.cx.data();
    const double* __restrict__ dy = P.denom_y.data();
    const double* __restrict__ cyf= P.cy.data();
    const double* __restrict__ dz = P.denom_z.data();
    const double* __restrict__ czf= P.cz.data();
    const unsigned int nd = P.diff_subs.size();
    const unsigned int nn = P.nodiff_subs.size();
    const unsigned int* diff = P.diff_subs.data();

    // scalar decay for non-diffusing substrates
    if( nn > 0 )
    {
        #pragma omp parallel for
        for( unsigned int v=0; v<nv; ++v )
            for( unsigned int si=0; si<nn; ++si )
                soa[ P.nodiff_subs[si]*nv + v ] *= P.nodiff_decay[si];
    }

    if( nd == 0 ) return;

    // x-sweep
    #pragma omp parallel for collapse(2)
    for( unsigned int k=0;k<nz;++k) for( unsigned int j=0;j<ny;++j)
    {
        const int n0 = (int)(0 + j*jjump + k*kjump);
        for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si]; soa[s*nv+n0] /= dx[s]; }
        for( unsigned int i=1;i<nx;++i){ const int n=n0+(int)i*ijump,nm=n-ijump;
            for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si];
                soa[s*nv+n]+=c1[s]*soa[s*nv+nm]; soa[s*nv+n]/=dx[i*ns+s]; } }
        for( int i=(int)nx-2;i>=0;--i){ const int n=n0+i*ijump,np=n+ijump;
            for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si];
                soa[s*nv+n]-=cxf[i*ns+s]*soa[s*nv+np]; } }
    }
    // y-sweep
    #pragma omp parallel for collapse(2)
    for( unsigned int k=0;k<nz;++k) for( unsigned int i=0;i<nx;++i)
    {
        const int n0 = (int)(i + 0*jjump + k*kjump);
        for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si]; soa[s*nv+n0] /= dy[s]; }
        for( unsigned int j=1;j<ny;++j){ const int n=n0+(int)j*jjump,nm=n-jjump;
            for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si];
                soa[s*nv+n]+=c1[s]*soa[s*nv+nm]; soa[s*nv+n]/=dy[j*ns+s]; } }
        for( int j=(int)ny-2;j>=0;--j){ const int n=n0+j*jjump,np=n+jjump;
            for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si];
                soa[s*nv+n]-=cyf[j*ns+s]*soa[s*nv+np]; } }
    }
    // z-sweep
    #pragma omp parallel for collapse(2)
    for( unsigned int j=0;j<ny;++j) for( unsigned int i=0;i<nx;++i)
    {
        const int n0 = (int)(i + j*jjump + 0*kjump);
        for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si]; soa[s*nv+n0] /= dz[s]; }
        for( unsigned int kk=1;kk<nz;++kk){ const int n=n0+(int)kk*kjump,nm=n-kjump;
            for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si];
                soa[s*nv+n]+=c1[s]*soa[s*nv+nm]; soa[s*nv+n]/=dz[kk*ns+s]; } }
        for( int kk=(int)nz-2;kk>=0;--kk){ const int n=n0+kk*kjump,np=n+kjump;
            for( unsigned int si=0; si<nd; ++si ){ unsigned int s=diff[si];
                soa[s*nv+n]-=czf[kk*ns+s]*soa[s*nv+np]; } }
    }
}

bool gpu_backend_is_cuda() { return false; }

#endif // backend select

}; // namespace BioFVM
