#pragma once

template<typename Real>
void solvePoissonGPU7(
    int ni, int nj, int nk,
    const Real* h_aw,
    const Real* h_ae,
    const Real* h_as,
    const Real* h_an,
    const Real* h_al,
    const Real* h_ah,
    const Real* h_su,
    const Real* h_ap,
    Real* h_phi,
    int nIter
);

template<typename Real>
void solvePoissonGPU27(
    int ni, int nj, int nk,
    const Real* const h_coeffs[26],
    const Real* h_su,
    const Real* h_ap,
    Real* h_phi,
    int nIter
);
