#pragma once 
#include <vector>

struct Grid3D {
    int ni, nj, nk;
    double xmax, ymax, zmax;
    double dx, dy, dz;
    std::vector<double> x, y, z;
};

Grid3D make_grid(int ni, int nj, int nk,
                 double xmax, double ymax, double zmax);
