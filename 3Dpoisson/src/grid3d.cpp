#include <grid3d.hpp>

Grid3D make_grid(int ni, int nj, int nk,
                 double xmax, double ymax, double zmax) {
    Grid3D g;
    g.ni = ni;
    g.nj = nj;
    g.nk = nk;

    g.xmax = xmax;
    g.ymax = ymax;
    g.zmax = zmax;
    
    // Spacing
    g.dx = xmax / ni;
    g.dy = ymax / nj;
    g.dz = zmax / nk;
    
    g.x.resize(ni);
    g.y.resize(nj);
    g.z.resize(nk);

    for (int i = 0; i < ni; ++i) g.x[i] = i * g.dx;
    for (int j = 0; j < nj; ++j) g.y[j] = j * g.dy;
    for (int k = 0; k < nk; ++k) g.z[k] = k * g.dz;

    return g;
}

