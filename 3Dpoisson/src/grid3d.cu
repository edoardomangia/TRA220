// grid3d.cu
#include "grid3d.cuh"

Grid3DDevice make_grid_device(int ni, int nj, int nk,
                              double xmax, double ymax, double zmax) {
    Grid3DDevice g{};
    g.ni = ni;
    g.nj = nj;
    g.nk = nk;

    g.xmax = xmax;
    g.ymax = ymax;
    g.zmax = zmax;

    g.dx = xmax / ni;
    g.dy = ymax / nj;
    g.dz = zmax / nk;
    
    return g;
}

void free_grid_device(Grid3DDevice&) {
    // Grid3DDevice currently holds only metadata; nothing to free.
}
