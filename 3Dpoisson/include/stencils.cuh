#pragma once

// 7-point stencil (6 neighbors) for the 3D Laplacian
// Uses SoA layout: one coefficient array per direction.
// Coeffs + domain size are provided by the host.

template<typename Real>
struct Stencil7 {
    const Real *aw, *ae, *as_, *an, *al, *ah;
    int ni, nj, nk;

    __host__ __device__
    Stencil7() : aw(nullptr), ae(nullptr), as_(nullptr), an(nullptr),
                 al(nullptr), ah(nullptr),
                 ni(0), nj(0), nk(0) {}

    __host__ __device__
    Stencil7(const Real* aw_, const Real* ae_,
             const Real* as__, const Real* an_,
             const Real* al_, const Real* ah_,
             int ni_, int nj_, int nk_)
        : aw(aw_), ae(ae_), as_(as__), an(an_),
          al(al_), ah(ah_),
          ni(ni_), nj(nj_), nk(nk_) {}

    // Computes only the *neighbor* sum a_N * phi_N.
    // su and ap are handled in the kernel.
    __device__ Real operator()(int idx, int i, int j, int k,
                               const Real* __restrict__ phi_old) const {
        int sx = 1;
        int sy = ni;
        int sz = ni * nj;

        Real phiC = phi_old[idx];

        Real phiE = (i+1 < ni) ? phi_old[idx + sx] : phiC;
        Real phiW = (i-1 >= 0) ? phi_old[idx - sx] : phiC;
        Real phiN = (j+1 < nj) ? phi_old[idx + sy] : phiC;
        Real phiS = (j-1 >= 0) ? phi_old[idx - sy] : phiC;
        Real phiH = (k+1 < nk) ? phi_old[idx + sz] : phiC;
        Real phiL = (k-1 >= 0) ? phi_old[idx - sz] : phiC;

        return ae[idx] * phiE + aw[idx] * phiW +
               an[idx] * phiN + as_[idx] * phiS +
               ah[idx] * phiH + al[idx] * phiL;
    }
};


// Skeleton for a 27-point stencil.
// This assumes a coefficient array per neighbor direction (26 neighbors).
// You will need to define how you store and fill these in your PoissonSystem.

template<typename Real>
struct Stencil27 {
    // 26 neighbor coefficient arrays (e.g. a[0] for direction (-1, 0, 0) etc.)
    const Real* a[26];
    int ni, nj, nk;

    __host__ __device__
    Stencil27() : a{nullptr}, ni(0), nj(0), nk(0) {}

    __host__ __device__
    Stencil27(const Real* a_[26], int ni_, int nj_, int nk_)
        : ni(ni_), nj(nj_), nk(nk_) {
        for (int n = 0; n < 26; ++n) a[n] = a_[n];
    }

    __device__ Real operator()(int idx, int i, int j, int k,
                               const Real* __restrict__ phi_old) const {
        int sx = 1;
        int sy = ni;
        int sz = ni * nj;

        // Example offset lists – YOU need to fill these consistently
        // with however you order the 26 neighbors in 'a'.
        // For brevity, I’m not filling all 26 entries here.
        __shared__ int dx[26], dy[26], dz[26]; // or make them static constexpr in a .cu

        Real phiC = phi_old[idx];
        Real sum = Real(0);

        #pragma unroll
        for (int n = 0; n < 26; ++n) {
            int ii = i + dx[n];
            int jj = j + dy[n];
            int kk = k + dz[n];

            int idxN;
            if (ii < 0 || ii >= ni ||
                jj < 0 || jj >= nj ||
                kk < 0 || kk >= nk) {
                // Reflecting BC as in the 7-point version
                idxN = idx;
            } else {
                idxN = ii + ni * (jj + nj * kk);
            }

            sum += a[n][idx] * phi_old[idxN];
        }

        return sum;
    }
};

