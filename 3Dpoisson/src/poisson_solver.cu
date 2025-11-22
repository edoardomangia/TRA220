// poisson_solver.cu
#include <cuda_runtime.h>
#include <utility>   
#include <iostream>  
#include "poisson_solver.hpp"
#include "poisson_system.cuh"
#include "poisson_init.cuh"
#include "grid3d_device.cuh"
#include "idx3d.cuh"
#include "cuda_utils.hpp"

template<typename Real>
__global__
void PoissonKernel(
    const Real* __restrict__ phi_old,
    Real* __restrict__ phi_new,
    const Real* __restrict__ aw,
    const Real* __restrict__ ae,
    const Real* __restrict__ as_,
    const Real* __restrict__ an,
    const Real* __restrict__ al,
    const Real* __restrict__ ah,
    const Real* __restrict__ su,
    const Real* __restrict__ ap,
    int ni, int nj, int nk
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= ni || j >= nj || k >= nk) return;

    const int sx = 1;
    const int sy = ni;
    const int sz = ni * nj;

    int idx = i + ni * (j + nj * k);

    // Neumann BC 
    Real phiC = phi_old[idx];

    Real phiE = (i + 1 < ni) ? phi_old[idx + sx] : phiC;
    Real phiW = (i - 1 >= 0) ? phi_old[idx - sx] : phiC;
    Real phiN = (j + 1 < nj) ? phi_old[idx + sy] : phiC;
    Real phiS = (j - 1 >= 0) ? phi_old[idx - sy] : phiC;
    Real phiH = (k + 1 < nk) ? phi_old[idx + sz] : phiC;
    Real phiL = (k - 1 >= 0) ? phi_old[idx - sz] : phiC;

    Real numerator =
        ae[idx] * phiE + aw[idx] * phiW +
        an[idx] * phiN + as_[idx] * phiS +
        ah[idx] * phiH + al[idx] * phiL +
        su[idx];

    // For Dirichlet cells set ap = 1 and all neighbor coeffs = 0
    // so this reduces to phi_new[idx] = su[idx]
    phi_new[idx] = numerator / ap[idx];
}

template<typename Real>
void solvePoissonGPU_impl(const Grid3DDevice &g,
                          Real *h_phi,
                          int nIter)
{
    const int ni = g.ni;
    const int nj = g.nj;
    const int nk = g.nk;

    const std::size_t N = static_cast<std::size_t>(ni) * nj * nk;
    const std::size_t bytes = N * sizeof(Real);

    // Allocate and initialize Poisson system 
    PoissonSystemDevice<Real> sys;
    allocatePoissonSystemDevice<Real>(ni, nj, nk, sys);
    initPoissonSystemDevice<Real>(g, sys);

    // Set up phi buffers
    // sys.phi contains the initial guess from initPoissonSystemDevice
    Real *d_phi_old = sys.phi;
    Real *d_phi_new = nullptr;

    cudaMalloc(&d_phi_new, bytes);
    cudaMemcpy(d_phi_new, d_phi_old, bytes, cudaMemcpyDeviceToDevice);

    // Launch
    dim3 block(8, 8, 8);
    dim3 gridDim(
        (ni + block.x - 1) / block.x,
        (nj + block.y - 1) / block.y,
        (nk + block.z - 1) / block.z
    );

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Iterations
    for (int it = 0; it < nIter; ++it) {
        PoissonKernel<Real><<<gridDim, block>>>(
            d_phi_old, d_phi_new,
            sys.aw, sys.ae, sys.as_, sys.an,
            sys.al, sys.ah,
            sys.su, sys.ap,
            ni, nj, nk
        );
        cudaGetLastError();

        std::swap(d_phi_old, d_phi_new);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "GPU Poisson solve: " << ms
              << " ms for " << nIter << " iterations\n";

    // Copy result back to host
    cudaMemcpy(h_phi, d_phi_old, bytes, cudaMemcpyDeviceToHost);

    // Clean 
    cudaFree(d_phi_new);
    freePoissonSystemDevice<Real>(sys);
}

// Explicit instantiation of the internal templated solver
template void solvePoissonGPU_impl<float>(const Grid3DDevice&, float*, int);
template void solvePoissonGPU_impl<double>(const Grid3DDevice&, double*, int);

// Public non templated wrappers
namespace poisson3d {

    void solvePoissonGPU_float(const Grid3DDevice &g,
                               float *h_phi,
                               int nIter)
    {
        solvePoissonGPU_impl<float>(g, h_phi, nIter);
    }

    void solvePoissonGPU_double(const Grid3DDevice &g,
                                double *h_phi,
                                int nIter)
    {
        solvePoissonGPU_impl<double>(g, h_phi, nIter);
    }

} 

