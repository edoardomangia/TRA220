#include <cuda_runtime.h>
#include <iostream>
#include "poisson_solver.hpp"

__global__
void poissonKernel(
    const double* __restrict__ phi_old,
    double* __restrict__ phi_new,
    const double* __restrict__ aw,
    const double* __restrict__ ae,
    const double* __restrict__ as_,
    const double* __restrict__ an,
    const double* __restrict__ al,
    const double* __restrict__ ah,
    const double* __restrict__ su,
    const double* __restrict__ ap,
    int ni, int nj, int nk
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Skip threads outside the domain... halos on stencils?
    if (i >= ni || j >= nj || k >= nk) return;
    
    // 1D flat array index 
    int sx = 1;
    int sy = ni;
    int sz = ni * nj;
    int idx = i + ni * (j + nj * k);

    // Read neighbors values with reflection at boundaries
    double phiE = (i+1 < ni) ? phi_old[idx + sx] : phi_old[idx];
    double phiW = (i-1 >= 0) ? phi_old[idx - sx] : phi_old[idx];
    double phiN = (j+1 < nj) ? phi_old[idx + sy] : phi_old[idx];
    double phiS = (j-1 >= 0) ? phi_old[idx - sy] : phi_old[idx];
    double phiH = (k+1 < nk) ? phi_old[idx + sz] : phi_old[idx];
    double phiL = (k-1 >= 0) ? phi_old[idx - sz] : phi_old[idx];

    // Update
    double numerator =
        ae[idx] * phiE + aw[idx] * phiW +
        an[idx] * phiN + as_[idx] * phiS +
        ah[idx] * phiH + al[idx] * phiL +
        su[idx];
    
    phi_new[idx] = numerator / ap[idx];
}

void solvePoissonGPU(
    int ni, int nj, int nk,
    const double* h_aw,
    const double* h_ae,
    const double* h_as,
    const double* h_an,
    const double* h_al,
    const double* h_ah,
    const double* h_su,
    const double* h_ap,
    double* h_phi,
    int nIter
) {
    
    // Number of cells and memory needed
    size_t N = (size_t)ni * nj * nk;
    size_t bytes = N * sizeof(double);
    
    // Device memory allocation 
    double *d_aw, *d_ae, 
           *d_as, *d_an, 
           *d_al, *d_ah, 
           *d_su, 
           *d_ap;

    double *d_phi_old, *d_phi_new;

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
        poissonKernel<<<grid, block>>>(
            d_phi_old, d_phi_new,
            d_aw, d_ae, d_as, d_an, d_al, d_ah,
            d_su, d_ap,
            ni, nj, nk
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
    cudaFree(d_su); 
    cudaFree(d_ap);
    cudaFree(d_phi_old); cudaFree(d_phi_new);
}

