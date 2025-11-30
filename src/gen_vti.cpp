/*
 * gen_vti.hpp 
 */

#include "gen_vti.hpp"

#include "idx3d.cuh"

template void write_vti<float>(const std::string&, const Grid3D&, const std::vector<float>&);

template void write_vti<double>(const std::string&, const Grid3D&, const std::vector<double>&);
