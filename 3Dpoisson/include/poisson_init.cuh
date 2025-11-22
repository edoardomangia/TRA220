// poisson_init.cuh
#pragma once
#include "grid3d.cuh"
#include "poisson_system.cuh"

template<typename Real>
void allocatePoissonSystemDevice(int ni, int nj, int nk,
                                 PoissonSystemDevice<Real> &sys);

template<typename Real>
void freePoissonSystemDevice(PoissonSystemDevice<Real> &sys);

template<typename Real>
void initPoissonSystemDevice(const Grid3DDevice &g,
                             PoissonSystemDevice<Real> &sys);

