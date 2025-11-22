// poisson_system.cu
#include <cuda_runtime.h>
#include <cstddef>
#include "poisson_system.cuh"
// #include "cuda_utils.hpp"

template<typename Real>
void allocatePoissonSystemDevice(int ni, int nj, int nk,
                                 PoissonSystemDevice<Real> &sys) {
    size_t N = static_cast<size_t>(ni) * nj * nk;
    size_t bytes = N * sizeof(Real);
    
    sys.aw = sys.ae = sys.as_ = sys.an = sys.al = sys.ah = nullptr;
    sys.ap = sys.su = sys.phi = nullptr;
    
    cudaMalloc(&sys.aw,  bytes);
    cudaMalloc(&sys.ae,  bytes);
    cudaMalloc(&sys.as_, bytes);
    cudaMalloc(&sys.an,  bytes);
    cudaMalloc(&sys.al,  bytes);
    cudaMalloc(&sys.ah,  bytes);
    cudaMalloc(&sys.ap,  bytes);
    cudaMalloc(&sys.su,  bytes);
    cudaMalloc(&sys.phi, bytes);

    cudaMemset(sys.aw,  0, bytes);
    cudaMemset(sys.ae,  0, bytes);
    cudaMemset(sys.as_, 0, bytes);
    cudaMemset(sys.an,  0, bytes);
    cudaMemset(sys.al,  0, bytes);
    cudaMemset(sys.ah,  0, bytes);
    cudaMemset(sys.ap,  0, bytes);
    cudaMemset(sys.su,  0, bytes);
    cudaMemset(sys.phi, 0, bytes);
}

template<typename Real>
void freePoissonSystemDevice(PoissonSystemDevice<Real>& sys) {
    cudaFree(sys.aw);
    cudaFree(sys.ae);
    cudaFree(sys.as_);
    cudaFree(sys.an);
    cudaFree(sys.al);
    cudaFree(sys.ah);
    cudaFree(sys.ap);
    cudaFree(sys.su);
    cudaFree(sys.phi);

    sys.aw = sys.ae = sys.as_ = sys.an = sys.al = sys.ah = nullptr;
    sys.ap = sys.su = sys.phi = nullptr;
}

template void allocatePoissonSystemDevice<float>(int, int, int,
                                                 PoissonSystemDevice<float>&);

template void allocatePoissonSystemDevice<double>(int, int, int,
                                                  PoissonSystemDevice<double>&);

template void freePoissonSystemDevice<float>(PoissonSystemDevice<float>&);

template void freePoissonSystemDevice<double>(PoissonSystemDevice<double>&);

