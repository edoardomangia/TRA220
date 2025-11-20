#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <limits>
#include "grid3d.hpp"
#include "poisson_coeffs.hpp"
#include "poisson_solver.cuh"
#include "gen_vti.hpp"

int main() {
    
    // using Real = int;    // not now
    // using Real = float;
    using Real = double;

    // Info
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Using: " << device << ": " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount
              << ", global mem: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    
    // Cartesian grid
    int ni = 64, 
        nj = 64, 
        nk = 64;
    double xmax = 1.0, 
           ymax = 1.0, 
           zmax = 1.0;

    Grid3D grid = make_grid(ni, nj, nk, xmax, ymax, zmax);

    PoissonSystem<Real> sys = make_poisson_system<Real>(grid);
    
    // ni = grid.ni, nj = grid.nj, nk = grid.nk;
    int idx0 = 0;
    int ni2 = ni / 2, nj2 = nj / 2, nk2 = nk / 2;
    int idx_mid = ni2 + ni * (nj2 + nj * nk2);
     
    std::cout << "ap[0] = " << sys.ap[idx0]
              << ", su[0] = " << sys.su[idx0] << "\n";
    std::cout << "ap[mid] = " << sys.ap[idx_mid]
              << ", su[mid] = " << sys.su[idx_mid] << "\n";
    
    // Checks su
    Real minsu = std::numeric_limits<Real>::max();
    Real maxsu = -std::numeric_limits<Real>::max();
    for (Real v : sys.su) {
        if (v < minsu) minsu = v;
        if (v > maxsu) maxsu = v;
    }
    std::cout << "su range: min = " << minsu
              << ", max = " << maxsu << "\n";

    int nIter = 2000;  
    
    std::cout << "Before phi[0] = " << sys.phi[0] << "\n";
 
    solvePoissonGPU<Real>(
        ni, nj, nk,
        sys.aw.data(), sys.ae.data(),
        sys.as_.data(), sys.an.data(),
        sys.al.data(), sys.ah.data(),
        sys.su.data(), sys.ap.data(),
        sys.phi.data(),
        nIter
    );
    
    std::cout << "After phi[0] = " << sys.phi[0] << "\n";
    
    // Checks phi 
    Real minphi = std::numeric_limits<Real>::max();
    Real maxphi = -std::numeric_limits<Real>::max();
    for (Real v : sys.phi) {
        if (v < minphi) minphi = v;
        if (v > maxphi) maxphi = v;
    } 
    std::cout << "phi range: min = " << minphi
              << ", max = " << maxphi << "\n";

    // Plotting
    // std::ofstream fout("phi_midplane.csv");
    // int k_mid = nk / 2;
    // for (int j = 0; j < nj; ++j) {
    //     for (int i = 0; i < ni; ++i) {
    //         int id = i + ni * (j + nj * k_mid);
    //         fout << grid.x[i] << "," << grid.y[j] << "," << sys.phi[id] << "\n";
    //     }
    //     fout << "\n";
    // }
    
    // Plotting 
    gen_vti("phi.vti", grid, sys.phi);

    return 0;
}

