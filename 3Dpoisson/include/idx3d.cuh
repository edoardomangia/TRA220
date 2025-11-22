// idx3d.cuh
#pragma once

__host__ __device__
inline int idx3D(int i, int j, int k, int ni, int nj, int /*nk*/) {
    return i + ni * (j + nj * k);
}


