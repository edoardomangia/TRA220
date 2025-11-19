#pragma once

#include <string>
#include <vector> 
#include "grid3d.hpp"

void gen_vti(const std::string& filename,
             const Grid3D& g,
             const std::vector<double>& phi);
