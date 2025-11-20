#pragma once

// 7-points stencil, 6 neighbors 

template<typename Real>
struct Stencil7 {
    const Real *aw, *ae, *as_, *an, *al, *ah;
    int ni, nj, nk;

    __host__ __device__
    Stencil7() : aw(nullptr), ae(nullptr), as_(nullptr), an(nullptr),
                 al(nullptr), ah(nullptr),
                 ni(0), nj(0), nk(0) {};

    __host__ __device__
    Stencil7(const Real* aw_, const Real* ae_,
             const Real* as__, const Real* an_,
             const Real* al_, const Real* ah_,
             int ni_, int nj_, int nk_)
        : aw(aw_), ae(ae_), as_(as__), an(an_),
          al(al_), ah(ah_),
          ni(ni_), nj(nj_), nk(nk_) {}

    // Computes only the neighbor sum a_N * phi_N
    // su and ap handled in the kernel
    __device__ Real operator()(int idx, int i, int j, int k,
                               const Real* __restrict__ phi_old) const {
        int sx = 1;
        int sy = ni;
        int sz = ni * nj;
        
        // Neumann... if boundary cell, kill flux to the outside
        Real phiE = (i+1 < ni) ? phi_old[idx + sx] : phi_old[idx];
        Real phiW = (i-1 >= 0) ? phi_old[idx - sx] : phi_old[idx];
        Real phiN = (j+1 < nj) ? phi_old[idx + sy] : phi_old[idx];
        Real phiS = (j-1 >= 0) ? phi_old[idx - sy] : phi_old[idx];
        Real phiH = (k+1 < nk) ? phi_old[idx + sz] : phi_old[idx];
        Real phiL = (k-1 >= 0) ? phi_old[idx - sz] : phi_old[idx];

        return ae[idx] * phiE + aw[idx] * phiW +
               an[idx] * phiN + as_[idx] * phiS +
               ah[idx] * phiH + al[idx] * phiL;
    }
};

// TODO Stencil 27?
// template<typename Real>
// struct Stencil27 {
//     const Real* a[26];
//     int ni, nj, nk;
// 
//     __host__ __device__
//     Stencil27(const Real* a_[26], int ni_, int nj_, int nk_)
//         ni(ni_), nj(nj_), nk(nk_) {
//         for (int n = 0; n < 26; ++n) a[n] = a_[n];
//     };
// 
//     __device__ Real operator()(int idx, int i, int j, int k,
//                                const Real* __restrict__ phi_old) const {
//         return Real(0);
//     }
// };

