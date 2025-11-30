/*
 * main.cpp
 */

#include "gen_vti.hpp"
#include "grid3d.cuh"
#include "grid3d.hpp"
#include "poisson_solver.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

template <typename T> inline double to_double(T v) {
  if constexpr (std::is_same_v<T, __half>) {
    return static_cast<double>(__half2float(v));
  } else {
    return static_cast<double>(v);
  }
}

template <typename Real> int run(int ni, int nj, int nk, int nIter) {

  // Info
  // int device = 0;
  // cudaSetDevice(device);
  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, device);
  // std::cout << "Using " << device << ": " << prop.name << "\n";
  // std::cout << "SMs: " << prop.multiProcessorCount << "\n";
  // std::cout << "Global mem: " << (prop.totalGlobalMem / (1024 * 1024))
  //           << " MB\n";

  Real xmax = 1.0;
  Real ymax = 1.0;
  Real zmax = 1.0;

  // New grid
  Grid3DDevice<Real> grid_d =
      make_grid_device<Real>(ni, nj, nk, xmax, ymax, zmax);

  // Set initial phis
  std::size_t N = static_cast<std::size_t>(ni) * nj * nk;
  std::vector<Real> phi(N, Real(0));

  // std::cout << "Before phi[0] = " << to_double(phi[0]) << "\n";

  iterPoissonSolver<Real>(grid_d, phi.data(), nIter);

  // std::cout << "After phi[0] = " << to_double(phi[0]) << "\n";

  Grid3D<Real> grid_h = make_grid_from_device(grid_d);

  // Output directory (keep alongside Python/CuPy outputs)
  const std::filesystem::path outDir = std::filesystem::path("output");
  std::filesystem::create_directories(outDir);

  // Write raw binary for checking after
  std::ofstream out(outDir / "phi_cpp.bin", std::ios::binary);
  out.write(reinterpret_cast<const char *>(phi.data()),
            phi.size() * sizeof(Real));

  // Plotting
  write_vti((outDir / "phi.vti").string(), grid_h, phi);

  return 0;
}

int main(int argc, char **argv) {
  // Defaults
  int ni = 10, nj = 10, nk = 10;
  int nIter = 10000;
  std::string type = "half";

  if (argc >= 4) {
    ni = std::stoi(argv[1]);
    nj = std::stoi(argv[2]);
    nk = std::stoi(argv[3]);
  }
  if (argc >= 5) {
    nIter = std::stoi(argv[4]);
  }
  if (argc >= 6) {
    type = argv[5];
  }

  if (type == "float" || type == "float32") {
    return run<float>(ni, nj, nk, nIter);
  } else if (type == "double") {
    return run<double>(ni, nj, nk, nIter);
  } else {
    return run<__half>(ni, nj, nk, nIter);
  }
}
