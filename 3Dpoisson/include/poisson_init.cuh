// include/poisson_init.cuh
#pragma once
#include "grid3d.cuh"    
#include "poisson_system.cuh"

template<typename Real>
void initPoissonSystemDevice(const Grid3DDevice &g,
                             PoissonSystemDevice<Real> &sys);
