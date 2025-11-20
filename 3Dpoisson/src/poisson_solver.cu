#include <cuda_runtime.h>
#include <iostream>
#include "poisson_solver.cuh"
#include "stencils.cuh"

template<typename Real, typename Stencil>
__global__
void poissonKernel(
    const Real* __restrict__ phi_old,
    Real* __restrict__ phi_new,
    const Real* __restrict__ su,
    const Real* __restrict__ ap,
    int ni, int nj, int nk,
    Stencil stencil
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Skip threads outside the domain... halos on stencils?
    if (i >= ni || j >= nj || k >= nk) return;
    
    // 1D flat array index 
    int idx = i + ni * (j + nj * k);

    // Update
    Real neighborSum = stencil(idx, i, j, k, phi_old);
    Real numerator = su[idx] + stencil(idx, i, j, k, phi_old);
    phi_new[idx] = numerator / ap[idx];
}


template<typename Real, typename Stencil>
void solvePoissonGPU(
    int ni, int nj, int nk,
    const Real* h_aw,
    const Real* h_ae,
    const Real* h_as,
    const Real* h_an,
    const Real* h_al,
    const Real* h_ah,
    const Real* h_su,
    const Real* h_ap,
    Real* h_phi,
    int nIter
) {
    
    // Number of cells and memory needed
    size_t N = (size_t)ni * nj * nk;
    size_t bytes = N * sizeof(Real);
    
    // Device memory allocation 
    Real *d_aw, *d_ae, 
         *d_as, *d_an, 
         *d_al, *d_ah, 
         *d_su, *d_ap;

    Real *d_phi_old, *d_phi_new;

    cudaMalloc(&d_aw, bytes);
    cudaMalloc(&d_ae, bytes);
    cudaMalloc(&d_as, bytes);
    cudaMalloc(&d_an, bytes);
    cudaMalloc(&d_al, bytes);
    cudaMalloc(&d_ah, bytes);
    cudaMalloc(&d_su, bytes);
    cudaMalloc(&d_ap, bytes);
    cudaMalloc(&d_phi_old, bytes);
    cudaMalloc(&d_phi_new, bytes);

    cudaMemcpy(d_aw,  h_aw,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ae,  h_ae,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_as,  h_as,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_an,  h_an,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_al,  h_al,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ah,  h_ah,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_su,  h_su,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ap,  h_ap,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_old, h_phi, bytes, cudaMemcpyHostToDevice);
    
    Stencil stencil {
        d_aw, d_ae, d_as, d_an, d_al, d_ah,
        ni, nj, nk
    };
    
    // Dimensions
    dim3 block(8, 8, 8);
    dim3 grid(
        (ni + block.x - 1) / block.x,
        (nj + block.y - 1) / block.y,
        (nk + block.z - 1) / block.z
    );

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Iter
    for (int it = 0; it < nIter; ++it) {
        poissonKernel<Real, Stencil><<<grid, block>>>(
            d_phi_old, d_phi_new,
            d_su, d_ap,
            ni, nj, nk,
            stencil
        );
        // cudaDeviceSynchronize(); 
        std::swap(d_phi_old, d_phi_new);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GPU solve time: " << ms << " ms for " << nIter << " iterations\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Copy solution back
    cudaMemcpy(h_phi, d_phi_old, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_aw); cudaFree(d_ae); 
    cudaFree(d_as); cudaFree(d_an);
    cudaFree(d_al); cudaFree(d_ah); 
    cudaFree(d_su); cudaFree(d_ap);
    cudaFree(d_phi_old); cudaFree(d_phi_new);
}

template<typename Real>
void solvePoissonGPU7(
    int ni, int nj, int nk,
    const Real* h_aw,
    const Real* h_ae,
    const Real* h_as,
    const Real* h_an,
    const Real* h_al,
    const Real* h_ah,
    const Real* h_su,
    const Real* h_ap,
    Real* h_phi,
    int nIter
) {
    solvePoissonGPU<Real, Stencil7<Real>>(
        ni, nj, nk,
        h_aw, h_ae, h_as, h_an,
        h_al, h_ah,
        h_su, h_ap,
        h_phi,
        nIter
    );
}

template<typename Real>
void solvePoissonGPU27(
    int ni, int nj, int nk,
    const Real* const h_coeffs[26],
    const Real* h_su,
    const Real* h_ap,
    Real* h_phi,
    int nIter
) {
    solvePoissonGPU_impl<Real, Stencil27<Real>>(
        ni, nj, nk,
    );
}
template void solvePoissonGPU<int>(
    int, int, int,
    const int*, const int*, 
    const int*, const int*,
    const int*, const int*,
    const int*, const int*,
    int*, int
);
template void solvePoissonGPU<float>(
    int, int, int,
    const float*, const float*, 
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    float*, int
);

template void solvePoissonGPU<double>(
    int, int, int,
    const double*, const double*, 
    const double*, const double*,
    const double*, const double*,
    const double*, const double*,
    double*, int
);

