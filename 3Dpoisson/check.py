#!/usr/bin/env python

# Compare phi_py.bin and phi_cpp.bin

import argparse, sys
from pathlib import Path
import numpy as np
base_out = Path(__file__).resolve().parent / "output"

ap = argparse.ArgumentParser()
ap.add_argument("--py", default=str(base_out / "phi_py.bin"))
ap.add_argument("--cpp", default=str(base_out / "phi_cpp.bin"))
ap.add_argument("--ni", type=int)
ap.add_argument("--nj", type=int)
ap.add_argument("--nk", type=int)
args = ap.parse_args()

shape = (args.ni, args.nj, args.nk) if all([args.ni, args.nj, args.nk]) else None



def load(path):
    arr = np.fromfile(path, dtype=np.float64)
    if shape:
        if arr.size != np.prod(shape):
            sys.exit(f"Size mismatch ({path})")
        return arr.reshape(shape)
    r = round(arr.size ** (1/3))
    if r**3 != arr.size:
        sys.exit(f"Provide --ni/--nj/--nk")
    return arr.reshape((r, r, r))

py = load(Path(args.py))
cpp = load(Path(args.cpp))

if py.shape != cpp.shape:
    sys.exit(f"Shape mismatch: py {py.shape}, cpp {cpp.shape}")



def build_coeffs(ni, nj, nk, xmax=1.0, ymax=1.0, zmax=1.0):
    # Usual coefficients
    dx, dy, dz = xmax / ni, ymax / nj, zmax / nk
    aw = np.ones((ni, nj, nk)) * dy * dz / dx
    ae = aw.copy()
    as_ = np.ones((ni, nj, nk)) * dx * dz / dy
    an = as_.copy()
    al = np.ones((ni, nj, nk)) * dx * dy / dz
    ah = al.copy()
    sp = np.ones((ni, nj, nk)) * 1e-20
    su = np.ones((ni, nj, nk)) * 1e-20
    
    # Dirichlet BC
    p_west = 2.0
    sp[0, :, :] -= aw[0, :, :]
    su[0, :, :] += aw[0, :, :] * p_west
    
    # Neumann BC elsewhere
    aw[0, :, :] = 0
    ae[-1, :, :] = 0
    as_[:, 0, :] = 0
    an[:, -1, :] = 0
    al[:, :, 0] = 0
    ah[:, :, -1] = 0

    # Compute
    ap = aw + ae + as_ + an + al + ah - sp

    # Point source
    ni2, nj2, nk2 = ni // 2, nj // 2, nk // 2
    su[ni2, nj2, nk2] = 100 * dx * dy * dz

    return aw, ae, as_, an, al, ah, ap, su



def residual(phi, coeffs):
    aw, ae, as_, an, al, ah, ap, su = coeffs

    # A * phi -su
    return ap * phi - (
        ae * np.roll(phi, -1, axis=0) + aw * np.roll(phi, 1, axis=0) +
        an * np.roll(phi, -1, axis=1) + as_ * np.roll(phi, 1, axis=1) +
        ah * np.roll(phi, -1, axis=2) + al * np.roll(phi, 1, axis=2)
    ) - su



diff = py - cpp
idx = np.unravel_index(np.argmax(np.abs(diff)), py.shape)

# Euclidean norm
l2 = np.linalg.norm(diff)

print(f"Shape: {py.shape} - Elements: {py.size}\n")
print(f"Max abs diff: {np.abs(diff).max():.6e} at {idx}")
print(f"   (py={py[idx]:.6e}, cpp={cpp[idx]:.6e})\n")
print(f"L2 diff: {l2:.6e}\n")
print(f"Rel L2: {l2/(np.linalg.norm(py)+1e-30):.6e}")

coeffs = build_coeffs(*py.shape)
for name, arr in (("py", py), ("cpp", cpp)):
    r = residual(arr, coeffs)
    # Sum of absolute residuals 
    l1 = np.sum(np.abs(r))
    # ...for residuals 
    l2r = np.linalg.norm(r)
    ridx = np.unravel_index(np.argmax(np.abs(r)), r.shape)

    print(f"\n{name} Residual:")
    print(f"   L1 {l1:.3e}")
    print(f"   L2 {l2r:.3e}")
    print(f"   max {r[ridx]:.3e} at {ridx}\n")

# Mid-plane sanity check
mid_k = py.shape[2] // 2
mid_diff = (cpp[:, :, mid_k] - py[:, :, mid_k])
print(f"mid-plane k={mid_k} max diff {np.abs(mid_diff).max():.3e}")

# Accuracy threshold
tol_field = 1e-10
tol_resid = 1e-8
if np.linalg.norm(diff) > tol_field or np.linalg.norm(residual(cpp, coeffs)) > tol_resid:
    sys.exit("Do better...")
