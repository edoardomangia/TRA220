#pragma once
#include <vector>
#include "grid3d.hpp"

struct PoissonSystem {
    std::vector<double> aw, ae, as_, an, al, ah;
    std::vector<double> ap, su;
    std::vector<double> phi;  // unknown
};

PoissonSystem make_poisson_system(const Grid3D& g);

