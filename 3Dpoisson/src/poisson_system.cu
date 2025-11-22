// poisson_system.cu
#include "poisson_system.cuh"
#include <stdexcept>

template<typename Real>
PoissonSystemDevice<Real> allocatePoissonSystemDevice(int ni, int nj, int nk) {
    PoissonSystemDevice<Real> sys{};
    size_t N = (size_t)ni * nj * nk;
    size_t bytes = N * sizeof(Real);

    auto check = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " +
                                     cudaGetErrorString(err));
        }
    };

    check(cudaMalloc(&sys.aw,  bytes), "cudaMalloc aw");
    check(cudaMalloc(&sys.ae,  bytes), "cudaMalloc ae");
    check(cudaMalloc(&sys.as_, bytes), "cudaMalloc as");
    check(cudaMalloc(&sys.an,  bytes), "cudaMalloc an");
    check(cudaMalloc(&sys.al,  bytes), "cudaMalloc al");
    check(cudaMalloc(&sys.ah,  bytes), "cudaMalloc ah");
    check(cudaMalloc(&sys.ap,  bytes), "cudaMalloc ap");
    check(cudaMalloc(&sys.su,  bytes), "cudaMalloc su");
    check(cudaMalloc(&sys.phi, bytes), "cudaMalloc phi");

    cudaMemset(sys.aw,  0, bytes);
    cudaMemset(sys.ae,  0, bytes);
    cudaMemset(sys.as_, 0, bytes);
    cudaMemset(sys.an,  0, bytes);
    cudaMemset(sys.al,  0, bytes);
    cudaMemset(sys.ah,  0, bytes);
    cudaMemset(sys.ap,  0, bytes);
    cudaMemset(sys.su,  0, bytes);
    cudaMemset(sys.phi, 0, bytes);

    return sys;
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

template PoissonSystemDevice<float>  allocatePoissonSystemDevice<float>(int,int,int);
template PoissonSystemDevice<double> allocatePoissonSystemDevice<double>(int,int,int);
template void freePoissonSystemDevice<float>(PoissonSystemDevice<float>&);
template void freePoissonSystemDevice<double>(PoissonSystemDevice<double>&);

