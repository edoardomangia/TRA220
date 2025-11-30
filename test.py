#!/usr/bin/env python

"""
Benchmark/compare CPU, CuPy, and cpp Poisson solvers.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Helpers
def run_cmd(cmd: str, label: str):
    """Run a shell command, capture time and output."""
    t0 = time.perf_counter()
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        sys.exit(f"{label} failed: {cmd}")
    return {"runtime_s": dt, "stdout": res.stdout.strip(), "stderr": res.stderr.strip()}


def load_array(path: Path, dtype: np.dtype, shape):
    arr = np.fromfile(path, dtype=dtype)
    if shape:
        expected = np.prod(shape)
        if arr.size != expected:
            sys.exit(f"Size mismatch for {path}: found {arr.size}, expected {expected}")
        return arr.reshape(shape)
    r = round(arr.size ** (1 / 3))
    if r ** 3 != arr.size:
        sys.exit("Provide --ni/--nj/--nk when data is not cubic.")
    return arr.reshape((r, r, r))


def build_coeffs(ni, nj, nk, xmax=1.0, ymax=1.0, zmax=1.0):
    dx, dy, dz = xmax / ni, ymax / nj, zmax / nk
    aw = np.ones((ni, nj, nk)) * dy * dz / dx
    ae = aw.copy()
    as_ = np.ones((ni, nj, nk)) * dx * dz / dy
    an = as_.copy()
    al = np.ones((ni, nj, nk)) * dx * dy / dz
    ah = al.copy()
    sp = np.ones((ni, nj, nk)) * 1e-20
    su = np.ones((ni, nj, nk)) * 1e-20

    p_west = 2.0
    sp[0, :, :] -= aw[0, :, :]
    su[0, :, :] += aw[0, :, :] * p_west

    aw[0, :, :] = 0
    ae[-1, :, :] = 0
    as_[:, 0, :] = 0
    an[:, -1, :] = 0
    al[:, :, 0] = 0
    ah[:, :, -1] = 0

    ap = aw + ae + as_ + an + al + ah - sp

    ni2, nj2, nk2 = ni // 2, nj // 2, nk // 2
    su[ni2, nj2, nk2] = 100 * dx * dy * dz

    return aw, ae, as_, an, al, ah, ap, su


def residual(phi, coeffs):
    aw, ae, as_, an, al, ah, ap, su = coeffs
    return ap * phi - (
        ae * np.roll(phi, -1, axis=0) + aw * np.roll(phi, 1, axis=0) +
        an * np.roll(phi, -1, axis=1) + as_ * np.roll(phi, 1, axis=1) +
        ah * np.roll(phi, -1, axis=2) + al * np.roll(phi, 1, axis=2)
    ) - su


def summarize_diff(ref, other):
    diff = ref - other
    idx = np.unravel_index(np.argmax(np.abs(diff)), ref.shape)
    l2 = np.linalg.norm(diff)
    rel_l2 = l2 / (np.linalg.norm(ref) + 1e-30)
    return {
        "max_abs": float(np.abs(diff).max()),
        "max_idx": tuple(int(i) for i in idx),
        "ref_val": float(ref[idx]),
        "other_val": float(other[idx]),
        "l2": float(l2),
        "rel_l2": float(rel_l2),
    }


def summarize_residual(phi, coeffs):
    r = residual(phi, coeffs)
    ridx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    return {
        "l1": float(np.sum(np.abs(r))),
        "l2": float(np.linalg.norm(r)),
        "max": float(r[ridx]),
        "max_idx": tuple(int(i) for i in ridx),
    }


def parse_args():
    base_out = Path(__file__).resolve().parent / "output"
    ap = argparse.ArgumentParser()

    ap.add_argument("--py", default=str(base_out / "phi_py.bin"))
    ap.add_argument("--cupy", default=str(base_out / "phi_cupy.bin"))
    ap.add_argument("--cpp", default=str(base_out / "phi_cpp.bin"))
    
    ap.add_argument("--cpp-type", default="half", choices=["half", "float16", "float", "float32", "double"])
    ap.add_argument("--dtype-cpp", default=None)

    ap.add_argument("--nijk", type=int, help="set ni=nj=nk to this value")
    ap.add_argument("--ni", type=int)
    ap.add_argument("--nj", type=int)
    ap.add_argument("--nk", type=int)
    
    ap.add_argument("--run-py", default="python sample.py")
    ap.add_argument("--run-cupy", default="python sample_cuPy.py")
    ap.add_argument("--run-cpp", default="build/run")
    
    ap.add_argument("--tag", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--niter", type=int)
    ap.add_argument("--write-log", action="store_true")
    ap.add_argument("--log-path", default=str(base_out / "benchmarks.jsonl"))
    return ap.parse_args()


def main():
    args = parse_args()
    # Collapse nijk into ni/nj/nk if provided
    if args.nijk:
        args.ni = args.nj = args.nk = args.nijk

    shape = (args.ni, args.nj, args.nk) if all([args.ni, args.nj, args.nk]) else None
    # Map cpp type to dtype and C++ token
    type_to_dtype = {
        "half": "float16",
        "float16": "float16",
        "float": "float32",
        "float32": "float32",
        "double": "float64",
    }
    type_to_token = {
        "half": "half",
        "float16": "half",
        "float": "float",
        "float32": "float",
        "double": "double",
    }
    cpp_type_token = type_to_token[args.cpp_type]
    dtype_cpp = np.dtype(args.dtype_cpp) if args.dtype_cpp else np.dtype(type_to_dtype[args.cpp_type])

    # Print shape info up front if known
    if shape:
        print()
        print(f"Shape: {shape} - Elements: {np.prod(shape)}")
    else:
        print()
        print("Shape: (inferred from files)")
    
    print()
    print(" --- Timings --- ")
    print()

    # Build command strings with optional grid/niter flags
    grid_args = ""
    if args.ni and args.nj and args.nk:
        grid_args = f" --ni {args.ni} --nj {args.nj} --nk {args.nk}"
    niter_arg = f" --niter {args.niter}" if args.niter is not None else ""

    cmd_py = args.run_py + grid_args + niter_arg
    cmd_cupy = args.run_cupy + grid_args + niter_arg
    # C++ binary expects positional args: ni nj nk [nIter]
    cmd_cpp = args.run_cpp
    if args.ni and args.nj and args.nk:
        cmd_cpp += f" {args.ni} {args.nj} {args.nk}"
    # If type is provided but niter omitted, pass the default C++ nIter (10000) to keep argv positions valid
    if args.niter is not None:
        cmd_cpp += f" {args.niter}"
    elif args.cpp_type:
        cmd_cpp += f" 10000"
    if args.cpp_type:
        cmd_cpp += f" {cpp_type_token}"

    # Run 
    runs = {}
    for name, cmd in (("Python", cmd_py), ("CuPy", cmd_cupy), ("cpp", cmd_cpp)):
        runs[name.lower()] = run_cmd(cmd, name)
        print(f"{name} run time: {runs[name.lower()]['runtime_s']:.3f}s")
        #if runs[name.lower()]["stdout"]:
        #    print(f"{name} stdout:\n{runs[name.lower()]['stdout']}")
        if runs[name.lower()]["stderr"]:
            print(f"{name} stderr:\n{runs[name.lower()]['stderr']}")

    # Load outputs 
    py = load_array(Path(args.py), dtype=np.float64, shape=shape)
    cupy = load_array(Path(args.cupy), dtype=np.float64, shape=shape)
    cpp_raw = load_array(Path(args.cpp), dtype=dtype_cpp, shape=shape)

    cpp = cpp_raw.astype(np.float64, copy=False)
    if py.shape != cpp.shape or py.shape != cupy.shape:
        sys.exit(f"Shape mismatch among outputs: py {py.shape}, cpp {cpp.shape}, cupy {cupy.shape}")

    print()
    print(" --- Accuracy --- ")
    print()

    # vs Python reference
    cpp_stats = summarize_diff(py, cpp)
    print("[ py vs cpp ]")
    print(f"Max abs diff: {cpp_stats['max_abs']:.6e} at {cpp_stats['max_idx']}")
    print(f"   (py={cpp_stats['ref_val']:.6e}, cpp={cpp_stats['other_val']:.6e})")
    print(f"L2 diff: {cpp_stats['l2']:.6e}")
    print(f"Rel L2: {cpp_stats['rel_l2']:.6e}")
    print()

    cupy_stats = summarize_diff(py, cupy)
    print("[ py vs cuPy ]")
    print(f"Max abs diff: {cupy_stats['max_abs']:.6e} at {cupy_stats['max_idx']}")
    print(f"   (py={cupy_stats['ref_val']:.6e}, cupy={cupy_stats['other_val']:.6e})")
    print(f"L2 diff: {cupy_stats['l2']:.6e}")
    print(f"Rel L2: {cupy_stats['rel_l2']:.6e}")
    print()

    # Residuals
    coeffs = build_coeffs(*py.shape)
    resid_stats = {}
    for name, arr in (("py", py), ("cupy", cupy), ("cpp", cpp)):
        stats = summarize_residual(arr, coeffs)
        resid_stats[name] = stats
        print(f"{name} Residual:")
        print(f"   L1 {stats['l1']:.3e}")
        print(f"   L2 {stats['l2']:.3e}")
        print(f"   max {stats['max']:.3e} at {stats['max_idx']}\n")
    
    '''
    # Tolerance
    tol_field = 1e-10
    tol_resid = 1e-8
    if cpp_stats["l2"] > tol_field or np.linalg.norm(residual(cpp, coeffs)) > tol_resid:
        sys.exit("Do better...\n")
    '''

    # Logging
    if args.write_log:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": args.tag,
            "notes": args.notes,
            "grid": {"ni": py.shape[0], "nj": py.shape[1], "nk": py.shape[2]},
            "niter": args.niter,
            "cpp_dtype": args.dtype_cpp,
            "runs": runs,
            "accuracy": {
                "cpp_vs_py": cpp_stats,
                "cupy_vs_py": cupy_stats,
            },
            "residuals": resid_stats,
            "paths": {
                "py": str(Path(args.py)),
                "cupy": str(Path(args.cupy)),
                "cpp": str(Path(args.cpp)),
            },
        }
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Appended benchmark entry to {log_path}")


if __name__ == "__main__":
    main()
