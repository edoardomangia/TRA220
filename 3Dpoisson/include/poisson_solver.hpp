#pragma once

void solvePoissonGPU(
    int ni, int nj, int nk,
    const double* h_aw,
    const double* h_ae,
    const double* h_as,
    const double* h_an,
    const double* h_al,
    const double* h_ah,
    const double* h_su,
    const double* h_ap,
    double* h_phi,
    int nIter
);

