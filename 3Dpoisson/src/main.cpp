// main.cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "types.hpp"
#include "grid3d.cuh"
#include "poisson_solver.hpp"
#include "grid3d.hpp"
#include "gen_vti.hpp"

using poisson3d::Real;

int main() {
    
    // Info
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Using: " << device << ": " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout<< "Global mem: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    
    // Grid: cells and domain
    int ni = 64; 
    int nj = 64; 
    int nk = 64;
    
    double xmax = 1.0; 
    double ymax = 1.0; 
    double zmax = 1.0;
    
    Grid3DDevice grid_d = make_grid_device(ni, nj, nk, xmax, ymax, zmax);
    
    std::size_t N = static_cast<std::size_t>(ni) * nj * nk;
    std::vector<Real> phi(N, Real(0));

    int nIter = 2000;

    std::cout << "Before phi[0] = " << phi[0] << "\n";
    
    // poisson3d::solvePoissonGPU_float(grid_d, phi.data(), nIter);
    poisson3d::solvePoissonGPU_double(grid_d, phi.data(), nIter);
    
    std::cout << "After phi[0] = " << phi[0] << "\n";

    // TODO: write phi + grid to .vti using a CPU-only writer later

    
 
    // Checks phi 
    // Real minphi = std::numeric_limits<Real>::max();
    // Real maxphi = -std::numeric_limits<Real>::max();
    // for (Real v : phi) {
    //     if (v < minphi) minphi = v;
    //     if (v > maxphi) maxphi = v;
    // } 
    // std::cout << "phi range: min = " << minphi
    //           << ", max = " << maxphi << "\n";
    
    Grid3D grid_h = make_grid_from_device(grid_d);

    // Plotting 
    write_vti("phi.vti", grid_h, phi);
    
    free_grid_device(grid_d);
    return 0;
}

