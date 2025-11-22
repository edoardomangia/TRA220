// grid3d.cu
#include "grid3d.cuh"
#include <stdexcept>

__global__
void init_axis(double *axis, int n, double d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        axis[i] = i * d;
    }
}

Grid3DDevice make_grid_device(int ni, int nj, int nk,
                              double xmax, double ymax, double zmax) {
    Grid3DDevice g{};
    g.ni = ni;
    g.nj = nj;
    g.nk = nk;

    g.xmax = xmax;
    g.ymax = ymax;
    g.zmax = zmax;

    g.dx = xmax / ni;
    g.dy = ymax / nj;
    g.dz = zmax / nk;
    
    // Allocate device arrays
    // TODO Use scripts to check maybe 
    cudaError_t err;

    err = cudaMalloc(&g.x, ni * sizeof(double));
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for g.x");
    }

    err = cudaMalloc(&g.y, nj * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(g.x);
        throw std::runtime_error("cudaMalloc failed for g.y");
    }

    err = cudaMalloc(&g.z, nk * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(g.x);
        cudaFree(g.y);
        throw std::runtime_error("cudaMalloc failed for g.z");
    }

    // Launch kernels to fill x, y, z on device
    dim3 block(256);
    
    dim3 grid_x((ni + block.x - 1) / block.x);
    init_axis<<<grid_x, block>>>(g.x, ni, g.dx);

    dim3 grid_y((nj + block.x - 1) / block.x);
    init_axis<<<grid_y, block>>>(g.y, nj, g.dy);

    dim3 grid_z((nk + block.x - 1) / block.x);
    init_axis<<<grid_z, block>>>(g.z, nk, g.dz);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(g.x);
        cudaFree(g.y);
        cudaFree(g.z);
        throw std::runtime_error(std::string("Kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }

    return g;
}

void free_grid_device(Grid3DDevice& g) {
    if (g.x) cudaFree(g.x);
    if (g.y) cudaFree(g.y);
    if (g.z) cudaFree(g.z);
    g.x = g.y = g.z = nullptr;
}
