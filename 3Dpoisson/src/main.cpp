/* 
 * main.cpp
 */

#include "grid3d.cuh"
#include "poisson_solver.hpp"
#include "grid3d.hpp"
#include "gen_vti.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <string>

using Real = double;

int main() {
    
    // Info
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Using " << device << ": " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout<< "Global mem: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    
    // Grid: cells and domain
    int ni = 10; 
    int nj = 10; 
    int nk = 10;
    
    double xmax = 1.0; 
    double ymax = 1.0; 
    double zmax = 1.0;
    
    Grid3DDevice grid_d = make_grid_device(ni, nj, nk, xmax, ymax, zmax);
    
    std::size_t N = static_cast<std::size_t>(ni) * nj * nk;
    std::vector<Real> phi(N, Real(0));

    int nIter = 10000;

    std::cout << "Before phi[0] = " << phi[0] << "\n";
    
    solvePoissonGPU<Real>(grid_d, phi.data(), nIter);
    
    std::cout << "After phi[0] = " << phi[0] << "\n";
 
    Grid3D grid_h = make_grid_from_device(grid_d);

    // Output directory
    // Resolve to project-level /output (parent of build/)
    const std::filesystem::path outDir = std::filesystem::path("..") / "output";
    std::filesystem::create_directories(outDir);

    // Write raw binary for bitwise comparison with Python output
    {
        std::ofstream out(outDir / "phi_cpp.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(phi.data()),
                  phi.size() * sizeof(Real));
    }

    // Plotting 
    write_vti((outDir / "phi.vti").string(), grid_h, phi);
    
    return 0;
}
