// grid3d.cuh
#pragma once
#include <cuda_runtime.h>

// Grid description (device-side metadata)
template <typename Real> struct Grid3DDevice {
  int ni, nj, nk;        // Number of cells
  Real xmax, ymax, zmax; // Domain
  Real dx, dy, dz;       // Cell spacing
};

// Allocate device grid metadata (no device allocations yet)
template <typename Real>
Grid3DDevice<Real> make_grid_device(int ni, int nj, int nk, Real xmax,
                                    Real ymax, Real zmax) {
  Grid3DDevice<Real> g{};
  g.ni = ni;
  g.nj = nj;
  g.nk = nk;

  g.xmax = xmax;
  g.ymax = ymax;
  g.zmax = zmax;

  // Compute spacing in higher precision for low-precision Real
  g.dx = static_cast<Real>(static_cast<double>(xmax) / static_cast<double>(ni));
  g.dy = static_cast<Real>(static_cast<double>(ymax) / static_cast<double>(nj));
  g.dz = static_cast<Real>(static_cast<double>(zmax) / static_cast<double>(nk));

  return g;
}

template <typename Real> inline void free_grid_device(Grid3DDevice<Real> &) {
  // Grid3DDevice currently holds only metadata, nothing to free
}
