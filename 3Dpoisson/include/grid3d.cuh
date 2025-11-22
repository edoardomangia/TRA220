// grid3d.cuh
#pragma once
#include <cuda_runtime.h>

// Grid description
struct Grid3DDevice {
    int ni, nj, nk;             // Number of cells
    double xmax, ymax, zmax;    // Domain
    double dx, dy, dz;          // Cells spacing
    double *x, *y, *z;          // Device pointers
};

// Allocate device arrays, fills them with coords
Grid3DDevice make_grid_device(int ni, int nj, int nk,
                              double xmax, double ymax, double zmax);

void free_grid_device(Grid3DDevice& g);

