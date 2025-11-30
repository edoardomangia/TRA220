// grid3d.cu
// Explicit instantiations for common types.
#include "grid3d.cuh"

template Grid3DDevice<float> make_grid_device<float>(int, int, int, float, float, float);
template Grid3DDevice<double> make_grid_device<double>(int, int, int, double, double, double);

template void free_grid_device<float>(Grid3DDevice<float>&);
template void free_grid_device<double>(Grid3DDevice<double>&);
