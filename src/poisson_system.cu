/* 
 * poisson_system.cu
 * Explicit instantiations for common types.
 */

#include "poisson_system.cuh"

template void allocatePoissonSystemDevice<float>(
        int, int, int, PoissonSystemDevice<float>&);

template void allocatePoissonSystemDevice<double>(
        int, int, int, PoissonSystemDevice<double>&);

template void freePoissonSystemDevice<float>(
        PoissonSystemDevice<float>&);

template void freePoissonSystemDevice<double>(
        PoissonSystemDevice<double>&);
