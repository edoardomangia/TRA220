// poisson_solver.hpp
#pragma once
#include "grid3d.cuh"

template<typename Real>
void solvePoissonGPU(const Grid3DDevice& g, Real* h_phi, int nIter);
