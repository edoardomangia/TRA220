/*
#pragma once

#include <string>
#include <vector> 
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <type_traits>
#include "grid3d.hpp"


template<typename Real>
void gen_vti(const std::string& filename,
             const Grid3D& g,
             const std::vector<Real>& phi);

template<typename Real>
void gen_vti(const std::string& filename,
             const Grid3D& g, 
             const std::vector<Real>& phi) {
    int ni = g.ni, nj = g.nj, nk = g.nk;

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open " + filename);
    }

    out << std::setprecision(16);

    out << R"(<?xml version="1.0"?>)" << "\n";
    out << R"(<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">)" << "\n";

    out << "  <ImageData WholeExtent=\"0 " << (ni-1)
        << " 0 " << (nj-1)
        << " 0 " << (nk-1)
        << "\" Origin=\"0 0 0\" Spacing=\""
        << g.dx << " " << g.dy << " " << g.dz << "\">\n";

    out << "    <Piece Extent=\"0 " << (ni-1)
        << " 0 " << (nj-1)
        << " 0 " << (nk-1) << "\">\n";

    out << "      <PointData Scalars=\"phi\">\n";
    
    // TODO int types?
    const char* vtkType =
        // std::is_same_v<Real, int> ? "..." :
        std::is_same_v<Real, double> ? "Float64" :
        std::is_same_v<Real, float>  ? "Float32" :
                                       "Float64"; 

    out << "        <DataArray type=\"" << vtkType << "\" Name=\"phi\" format=\"ascii\">\n";

    // VTK expects values in x-fastest order   
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int id = idx3D(i,j,k,ni,nj,nk);
                out << " " << phi[id];
            }
            out << "\n";
        }
    }

    out << "        </DataArray>\n";
    out << "      </PointData>\n";
    out << "      <CellData></CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";
}

*/
