/*
 * poisson_system.cuh
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// Linear system
template <typename Real> struct PoissonSystemDevice {
  Real *aw, *ae, *as_, *an, *al, *ah;
  Real *ap, *su;
  Real *phi;
};

// Allocate system
template <typename Real>
void allocatePoissonSystemDevice(int ni, int nj, int nk,
                                 PoissonSystemDevice<Real> &sys) {
  std::size_t N = static_cast<std::size_t>(ni) * nj * nk;
  std::size_t bytes = N * sizeof(Real);

  sys.aw = sys.ae = sys.as_ = sys.an = sys.al = sys.ah = nullptr;
  sys.ap = sys.su = sys.phi = nullptr;

  cudaMalloc(&sys.aw, bytes);
  cudaMalloc(&sys.ae, bytes);
  cudaMalloc(&sys.as_, bytes);
  cudaMalloc(&sys.an, bytes);
  cudaMalloc(&sys.al, bytes);
  cudaMalloc(&sys.ah, bytes);
  cudaMalloc(&sys.ap, bytes);
  cudaMalloc(&sys.su, bytes);
  cudaMalloc(&sys.phi, bytes);
}

// Free system
template <typename Real>
void freePoissonSystemDevice(PoissonSystemDevice<Real> &sys) {
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
