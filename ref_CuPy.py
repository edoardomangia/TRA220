#!/usr/bin/env python

"""
GPU version of sample.py using CuPy.
"""

import socket
import time
from pathlib import Path
import argparse

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np


# Fused update kernel (Jacobi-style) to avoid many tiny kernel launches.
_poisson_update = cp.RawKernel(
    r"""
extern "C" __global__
void poisson_update(
    const double* __restrict__ phi_old,
    double* __restrict__ phi_new,
    const double* __restrict__ aw,
    const double* __restrict__ ae,
    const double* __restrict__ as_,
    const double* __restrict__ an,
    const double* __restrict__ al,
    const double* __restrict__ ah,
    const double* __restrict__ su,
    const double* __restrict__ ap,
    int ni, int nj, int nk)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int N = ni * nj * nk;
    if (idx >= N) return;

    int i = idx / (nj * nk);
    int j = (idx / nk) % nj;
    int k = idx % nk;

    // Strides (NumPy/C-order): k fastest, then j, then i
    int sx = nj * nk;
    int sy = nk;

    double phiC = phi_old[idx];
    double phiE = (i + 1 < ni) ? phi_old[idx + sx] : phiC;
    double phiW = (i - 1 >= 0) ? phi_old[idx - sx] : phiC;
    double phiN = (j + 1 < nj) ? phi_old[idx + sy] : phiC;
    double phiS = (j - 1 >= 0) ? phi_old[idx - sy] : phiC;
    double phiH = (k + 1 < nk) ? phi_old[idx + 1]  : phiC;
    double phiL = (k - 1 >= 0) ? phi_old[idx - 1]  : phiC;

    double numerator =
        ae[idx] * phiE + aw[idx] * phiW +
        an[idx] * phiN + as_[idx] * phiS +
        ah[idx] * phiH + al[idx] * phiL +
        su[idx];

    phi_new[idx] = numerator / ap[idx];
}
""",
    "poisson_update",
)


def solve_gs_gpu(phi3d, aw3d, ae3d, as3d, an3d, al3d, ah3d, su3d, ap3d, tol_conv, nmax):
    """
    Jacobi-like sweep implemented with a single fused kernel per iteration.
    Reduces launch overhead versus many cp.roll kernels.
    """
    acrank_conv = 1.0
    print("solve_gs_gpu called, nmax=", nmax)
    resid = cp.inf

    # Work on flattened views to match kernel expectations
    phi_old = phi3d.ravel()
    phi_new = cp.empty_like(phi_old)
    aw = aw3d.ravel()
    ae = ae3d.ravel()
    as_ = as3d.ravel()
    an = an3d.ravel()
    al = al3d.ravel()
    ah = ah3d.ravel()
    su = su3d.ravel()
    ap = ap3d.ravel()

    N = phi_old.size
    block = 256
    grid = (N + block - 1) // block

    for n in range(nmax):
        _poisson_update((grid,), (block,),
                        (phi_old, phi_new, aw, ae, as_, an, al, ah, su, ap,
                         phi3d.shape[0], phi3d.shape[1], phi3d.shape[2]))
        if (n + 1) % 10 == 0 or n == nmax - 1:
            # Quick residual check (same fused math) without extra kernels
            # Here we approximate convergence by L1 of (phi_new - phi_old)
            resid = cp.sum(cp.abs(phi_new - phi_old))
            if resid < tol_conv:
                print(f"GS converged at iter {n+1} with residual {float(resid):.3e}")
                phi_old, phi_new = phi_new, phi_old
                break
        # Swap buffers
        phi_old, phi_new = phi_new, phi_old

    # Reshape back
    phi3d = phi_old.reshape(phi3d.shape)
    return resid, phi3d


def poisson_gpu(solver, niter, convergence_limit, ni=10, nj=10, nk=10, xmax=1.0, ymax=1.0, zmax=1.0):
    print("\nhostname: ", socket.gethostname())
    print("\nsolver, convergence_limit, niter", solver, convergence_limit, niter)

    # grid setup
    dx = xmax / ni
    dy = ymax / nj
    dz = zmax / nk

    x = np.linspace(0, xmax, ni)
    y = np.linspace(0, ymax, nj)
    z = np.linspace(0, zmax, nk)

    # initial coefficients
    shape = (ni, nj, nk)
    aw3d = cp.ones(shape) * 1e-20
    ae3d = cp.ones(shape) * 1e-20
    as3d = cp.ones(shape) * 1e-20
    an3d = cp.ones(shape) * 1e-20
    al3d = cp.ones(shape) * 1e-20
    ah3d = cp.ones(shape) * 1e-20
    ap3d = cp.ones(shape) * 1e-20
    su3d = cp.ones(shape) * 1e-20
    sp3d = cp.ones(shape) * 1e-20

    # initial solution
    p3d = cp.ones(shape) * 1e-20

    # coefficients (finite volume)
    aw3d = cp.ones(shape) * dy * dz / dx
    ae3d = cp.ones(shape) * dy * dz / dx
    as3d = cp.ones(shape) * dx * dz / dy
    an3d = cp.ones(shape) * dx * dz / dy
    al3d = cp.ones(shape) * dx * dy / dz
    ah3d = cp.ones(shape) * dx * dy / dz

    # Dirichlet west boundary p=2
    p_west = 2.0
    sp3d[0, :, :] = sp3d[0, :, :] - aw3d[0, :, :]
    su3d[0, :, :] = su3d[0, :, :] + aw3d[0, :, :] * p_west

    # Neumann elsewhere via zero flux
    aw3d[0, :, :] = 0
    ae3d[-1, :, :] = 0
    as3d[:, 0, :] = 0
    an3d[:, -1, :] = 0
    al3d[:, :, 0] = 0
    ah3d[:, :, -1] = 0

    ap3d = aw3d + ae3d + as3d + an3d + al3d + ah3d - sp3d

    # point source in the middle
    ni2 = int(ni / 2)
    nj2 = int(nj / 2)
    nk2 = int(nk / 2)
    su3d[ni2, nj2, nk2] = 100 * dx * dy * dz

    t0 = time.time()
    if solver == "gs":
        residual, p3d = solve_gs_gpu(
            p3d, aw3d, ae3d, as3d, an3d, al3d, ah3d, su3d, ap3d, convergence_limit, niter
        )
    else:
        raise ValueError("Only 'gs' solver is implemented for the CuPy version.")
    t1 = time.time()

    print(f"Elapsed time: {t1 - t0:.3f} s, residual: {float(residual):.3e}")
    return p3d, residual, (x, y, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="gs", choices=["gs"])
    ap.add_argument("--niter", type=int, default=10000)
    ap.add_argument("--convergence_limit", type=float, default=0.0)
    ap.add_argument("--ni", type=int, default=10)
    ap.add_argument("--nj", type=int, default=10)
    ap.add_argument("--nk", type=int, default=10)
    ap.add_argument("--xmax", type=float, default=1.0)
    ap.add_argument("--ymax", type=float, default=1.0)
    ap.add_argument("--zmax", type=float, default=1.0)
    args = ap.parse_args()

    p3d_gpu, residual, (x, y, z) = poisson_gpu(
        solver=args.solver,
        niter=args.niter,
        convergence_limit=args.convergence_limit,
        ni=args.ni,
        nj=args.nj,
        nk=args.nk,
        xmax=args.xmax,
        ymax=args.ymax,
        zmax=args.zmax,
    )

    # outputs (host-side)
    p3d = cp.asnumpy(p3d_gpu)
    outdir = Path(__file__).resolve().parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    np.ravel(p3d, order="C").astype(np.float64).tofile(outdir / "phi_cupy.bin")

r"""
    plt.close("all")
    plt.interactive(True)
    plt.rcParams.update({"font.size": 22})

    fig1, ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20, bottom=0.20)
    nk2 = int(p3d.shape[2] / 2)
    plt.contourf(y, x, np.transpose(p3d[:, :, nk2]), 20, cmap="RdGy")
    plt.ylabel("$y$")
    plt.xlabel("$x$")
    plt.title(r"$\phi$ in plane $z=z_{max}/2$")
    plt.colorbar()
    plt.savefig(outdir / "poisson-p3d-cupy.png", bbox_inches="tight")

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
        c=p3d.ravel(),
        s=30,
        alpha=0.8,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(sc, ax=ax, label="Temperature")

    plt.tight_layout()
    plt.savefig(outdir / "poisson_3d_scatter_cupy.png", bbox_inches="tight")
    plt.show()
"""

if __name__ == "__main__":
    main()
