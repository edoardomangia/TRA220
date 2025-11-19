#pragma once

template<typename Real>
void solvePoissonGPU(
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

