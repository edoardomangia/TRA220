/*
 * 
 */

#pragma once

#include "grid3d.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

template<typename Real>
void write_vti(const std::string &filename,
               const Grid3D &g,
               const std::vector<Real> &phi)
{
    int ni = g.ni;
    int nj = g.nj;
    int nk = g.nk;

    std::ofstream f(filename);
    if (!f) throw std::runtime_error("Cannot open VTI file");

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <ImageData WholeExtent=\"0 " << ni-1
      << " 0 " << nj-1
      << " 0 " << nk-1
      << "\" Origin=\"0 0 0\" "
      << "Spacing=\"" << g.dx << " " << g.dy << " " << g.dz << "\">\n";
    f << "    <Piece Extent=\"0 " << ni-1
      << " 0 " << nj-1
      << " 0 " << nk-1 << "\">\n";

    f << "      <PointData Scalars=\"phi\">\n";
    f << "        <DataArray type=\"Float64\" Name=\"phi\" format=\"ascii\">\n";

    // Write scalars in i-fastest ordering (X fastest)
    for (int k = 0; k < nk; ++k)
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < ni; ++i)
                f << phi[i + ni*(j + nj*k)] << " ";

    f << "\n        </DataArray>\n";
    f << "      </PointData>\n";
    f << "      <CellData></CellData>\n";
    f << "    </Piece>\n";
    f << "  </ImageData>\n";
    f << "</VTKFile>\n";
}
