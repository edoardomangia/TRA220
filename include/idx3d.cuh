/*
 * idx3d.cuh
 */

#pragma once

__host__ __device__
inline int idx3D(int i, int j, int k, 
                 int ni, int nj, int nk) {
    // NumPy C-order flattening, k fastest
    return (i * nj + j) * nk + k;
    
    // return i * nj + (j * nk + k);
}

