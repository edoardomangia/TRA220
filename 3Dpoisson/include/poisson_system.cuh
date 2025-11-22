// poisson_system.cuh
#pragma once

// Linear system 
template<typename Real>
struct PoissonSystemDevice {
    Real *aw, *ae, *as_, *an, *al, *ah;
    Real *ap, *su;
    Real *phi;
};

// Allocate and free 
template<typename Real>
void allocatePoissonSystemDevice(int ni, int nj, int nk,
                                 PoissonSystemDevice<Real> &sys);

template<typename Real>
void freePoissonSystemDevice(PoissonSystemDevice<Real> &sys);
