#pragma once
#include <vector>
#include "grid3d.hpp"

template<typename Real>
struct PoissonSystem {
    std::vector<Real> aw, ae, as_, an, al, ah;
    std::vector<Real> ap, su;
    std::vector<Real> phi;
};

inline int idx3D(int i, int j, int k, 
                 int ni, int nj, int nk) {
    return i + ni * (j + nj * k);
}

template<typename Real>
PoissonSystem<Real> make_poisson_system(const Grid3D& g) {
    int ni = g.ni, nj = g.nj, nk = g.nk;
    size_t N = (size_t)ni * nj * nk;

    PoissonSystem<Real> sys;
    sys.aw.assign(N, Real(0));
    sys.ae.assign(N, Real(0));
    sys.as_.assign(N, Real(0));
    sys.an.assign(N, Real(0));
    sys.al.assign(N, Real(0));
    sys.ah.assign(N, Real(0));
    sys.ap.assign(N, Real(1));
    sys.su.assign(N, Real(0));
    sys.phi.assign(N, Real(0));

    Real cx = Real(g.dy * g.dz / g.dx);
    Real cy = Real(g.dx * g.dz / g.dy);
    Real cz = Real(g.dx * g.dy / g.dz);

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

    // Dirichlet on west boundary p=2
    Real p_west = Real(2.0);
    
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            int i = 0; // Cells on the west boundary
            int id = idx3D(i,j,k,ni,nj,nk);

            // Enforce by setting ap=1, su=p_west, zero neighbors
            sys.aw[id] = Real(0);
            sys.ae[id] = Real(0);
            sys.as_[id] = Real(0);
            sys.an[id] = Real(0);
            sys.al[id] = Real(0);
            sys.ah[id] = Real(0);
            sys.ap[id] = Real(1);
            sys.su[id] = p_west;
        }
    }

    // ap = sum of neighbor coeffs - sp (here sp=0)
    for (size_t id = 0; id < N; ++id) {
        if (sys.ap[id] == Real(1.0) && sys.su[id] == p_west) {
            continue; // Skip Dirichlet cells
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
    sys.su[id_mid] += Real(100.0 * g.dx * g.dy * g.dz);

    return sys;
}

