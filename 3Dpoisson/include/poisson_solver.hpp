// poisson_solver.hpp
#pragma once
#include "grid3d.cuh"

namespace poisson3d {
    void solvePoissonGPU_float(
            const Grid3DDevice&, float *h_phi, int nIter);
    void solvePoissonGPU_double(
            const Grid3DDevice&, double *h_phi, int nIter);
}

