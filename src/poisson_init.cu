/*
 * poisson_init.cu
 * Explicit instantiations for common types.
 */ 

#include "poisson_init.cuh"

template void initPoissonSystemDevice<float>(
        const Grid3DDevice<float> &,
        PoissonSystemDevice<float> &);

template void initPoissonSystemDevice<double>(
        const Grid3DDevice<double> &,
        PoissonSystemDevice<double> &);
