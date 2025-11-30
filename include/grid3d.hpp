// include/grid3d.hpp
#pragma once

#include "grid3d.cuh"  

// host-side grid, for plotting output
template<typename Real>
struct Grid3D {
    int ni, nj, nk;
    Real dx, dy, dz;
};

// Build a host Grid3D from a device Grid3DDevice (just copies metadata)
template<typename Real>
inline Grid3D<Real> make_grid_from_device(const Grid3DDevice<Real> &gd)
{
    Grid3D<Real> gh;
    gh.ni = gd.ni;
    gh.nj = gd.nj;
    gh.nk = gd.nk;
    gh.dx = gd.dx;
    gh.dy = gd.dy;
    gh.dz = gd.dz;
    return gh;
}
