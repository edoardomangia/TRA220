/* 
 * poisson_solver.cu
 * Explicit instantiations for common types.
 */

#include "poisson_solver.cuh"

template void iterPoissonSolver<float>(const Grid3DDevice<float>&, float*, int);
template void iterPoissonSolver<double>(const Grid3DDevice<double>&, double*, int);
