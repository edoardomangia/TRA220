// include/grid3d.hpp
#pragma once

#include "grid3d.cuh"  

// host-side grid, for plotting output
struct Grid3D {
    int ni, nj, nk;
    double dx, dy, dz;
};

// Build a host Grid3D from a device Grid3DDevice (just copies metadata)
inline Grid3D make_grid_from_device(const Grid3DDevice &gd)
{
    Grid3D gh;
    gh.ni = gd.ni;
    gh.nj = gd.nj;
    gh.nk = gd.nk;
    gh.dx = gd.dx;
    gh.dy = gd.dy;
    gh.dz = gd.dz;
    return gh;
}

