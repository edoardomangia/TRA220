#include "poisson_coeffs.hpp"

inline int idx3D(int i, int j, int k, 
                 int ni, int nj, int nk) {
    return i + ni * (j + nj * k);
}

PoissonSystem make_poisson_system(const Grid3D& g) {
    int ni = g.ni, nj = g.nj, nk = g.nk;
    size_t N = (size_t)ni * nj * nk;

    PoissonSystem sys;
    sys.aw.assign(N, 0.0);
    sys.ae.assign(N, 0.0);
    sys.as_.assign(N, 0.0);
    sys.an.assign(N, 0.0);
    sys.al.assign(N, 0.0);
    sys.ah.assign(N, 0.0);
    sys.ap.assign(N, 0.0);
    sys.su.assign(N, 0.0);
    sys.phi.assign(N, 0.0);

    double cx = g.dy * g.dz / g.dx;
    double cy = g.dx * g.dz / g.dy;
    double cz = g.dx * g.dy / g.dz;

    // Interior default coefficients
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int id = idx3D(i,j,k,ni,nj,nk);
                sys.aw[id] = cx;
                sys.ae[id] = cx;
                sys.as_[id] = cy;
                sys.an[id] = cy;
                sys.al[id] = cz;
                sys.ah[id] = cz;
            }
        }
    }

    // Dirichlet west boundary p=2, Neumann elsewhere
    double p_west = 2.0;

    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            int i = 0;
            int id = idx3D(i,j,k,ni,nj,nk);

            // Dirichlet: phi = p_west
            // Enforce by setting ap=1, su=p_west, zero neighbors
            sys.aw[id] = 0.0;
            sys.ae[id] = 0.0;
            sys.as_[id] = 0.0;
            sys.an[id] = 0.0;
            sys.al[id] = 0.0;
            sys.ah[id] = 0.0;
            sys.ap[id] = 1.0;
            sys.su[id] = p_west;
        }
    }

    // Neumann on other boundaries (zero flux)
    // Zeroing the outward coefficient, ap will be recomputed
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            int j0 = 0;
            int j1 = nj-1;
            sys.as_[idx3D(i,j0,k,ni,nj,nk)] = 0.0;
            sys.an[idx3D(i,j1,k,ni,nj,nk)] = 0.0;
        }
    }

    // ap = sum of neighbor coeffs - sp (here sp=0)
    for (size_t id = 0; id < N; ++id) {
        if (sys.ap[id] == 1.0 && sys.su[id] == p_west) {
            continue;
        }
        sys.ap[id] = sys.aw[id] + sys.ae[id]
                   + sys.as_[id] + sys.an[id]
                   + sys.al[id] + sys.ah[id];
    }

    // Point source in the middle
    int ni2 = ni / 2;
    int nj2 = nj / 2;
    int nk2 = nk / 2;
    int id_mid = idx3D(ni2, nj2, nk2, ni, nj, nk);
    sys.su[id_mid] += 100.0 * g.dx * g.dy * g.dz;

    return sys;
}

