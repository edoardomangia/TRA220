// poisson_init.cu
#include <cuda_runtime.h>
#include "idx3d.cuh"
#include "poisson_init.cuh"

template<typename Real>
__global__
void initPoissonSystemKernel(PoissonSystemDevice<Real> sys,
                             int ni, int nj, int nk,
                             Real cx, Real cy, Real cz,
                             Real p_west,
                             int id_mid,
                             Real source_strength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= ni || j >= nj || k >= nk) return;

    int id = idx3D(i, j, k, ni, nj, nk);

    // Default coefficients
    Real aw  = cx;
    Real ae  = cx;
    Real as_ = cy;
    Real an  = cy;
    Real al  = cz;
    Real ah  = cz;
    Real su  = Real(0);
    Real phi = Real(0);

    // Neumann BC: kill flux by zeroing those coefficients
    if (i == 0)      aw  = Real(0);
    if (i == ni - 1) ae  = Real(0);
    if (j == 0)      as_ = Real(0);
    if (j == nj - 1) an  = Real(0);
    if (k == 0)      al  = Real(0);
    if (k == nk - 1) ah  = Real(0);
   
    // TODO What if other Dir BCs? 
    // Dirichlet BC: at west boundary i = 0, p = p_west
    if (i == 0) {
        aw  = Real(0);
        ae  = Real(0);
        as_ = Real(0);
        an  = Real(0);
        al  = Real(0);
        ah  = Real(0);

        su  = p_west;
        phi = p_west;

        sys.ap[id] = Real(1);    
    } else {
        sys.ap[id] = aw + ae + as_ + an + al + ah;
    }

    sys.aw[id]  = aw;
    sys.ae[id]  = ae;
    sys.as_[id] = as_;
    sys.an[id]  = an;
    sys.al[id]  = al;
    sys.ah[id]  = ah;
    sys.su[id]  = su;
    sys.phi[id] = phi;

    // Point source in the middle: add to su
    if (id == id_mid) {
        sys.su[id] += source_strength;
    }
}
    
    // TODO Other BCs? 

template<typename Real>
void initPoissonSystemDevice(const Grid3DDevice &g,
                             PoissonSystemDevice<Real> &sys)
{
    int ni = g.ni;
    int nj = g.nj;
    int nk = g.nk;

    Real dx = static_cast<Real>(g.dx);
    Real dy = static_cast<Real>(g.dy);
    Real dz = static_cast<Real>(g.dz);

    // Coefficients from finite volume discretization
    Real cx = dy * dz / dx;
    Real cy = dx * dz / dy;
    Real cz = dx * dy / dz;

    // Dirichlet value at west boundary
    Real p_west = Real(2.0);

    // Middle cell index (for point source location)
    int ni2 = ni / 2;
    int nj2 = nj / 2;
    int nk2 = nk / 2;

    int id_mid = idx3D(ni2, nj2, nk2, ni, nj, nk);

    // Point source strength 
    Real source_strength = Real(1000000.0) * dx * dy * dz;

    dim3 block(8, 8, 8);
    dim3 gridDim(
        (ni + block.x - 1) / block.x,
        (nj + block.y - 1) / block.y,
        (nk + block.z - 1) / block.z
    );

    initPoissonSystemKernel<<<gridDim, block>>>(
        sys,
        ni, nj, nk,
        cx, cy, cz,
        p_west,
        id_mid,
        source_strength
    );
}

// cudaGetLastError();

template void initPoissonSystemDevice<float>(
        const Grid3DDevice &,
        PoissonSystemDevice<float> &);

template void initPoissonSystemDevice<double>(
        const Grid3DDevice &,
        PoissonSystemDevice<double> &);

