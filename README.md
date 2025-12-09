_...WIP_

# Poisson Solver (Python / CuPy / C++ CUDA)

The idea is to benchmark and compare a 3D Poisson solver implemented in:
- Python with NumPy: the reference implementation, CPU only.
- Python with CuPy: a port of the above, same Jacobi-like sweep with a fused kernel in CuPy for GPU workload.
- CUDA C++: the actual solver that's worked on the most, with the logic ported from the reference NumPy code.

The solver tackles a finite-volume discretization of the Poisson equation on a unit cube with:
- Dirichlet boundary on the west face (ϕ = 2).
- Neumann boundaries elsewhere.
- A point source at the domain center.

Outputs from all implementations are written to the `output/` directory for cross-checks and plotting.

## Build and run the CUDA C++ solver

From the repo root:
```bash
cmake -S . -B build
cmake --build build

# run: build/run ni nj nk [nIter] [type]
# type: half (default), float/float32, or double
build/run 32 32 32 200 float
```

Outputs:
- `output/phi_cpp.bin` (raw field, dtype depends on `type`)
- `output/phi.vti` (VTK image for visualization)

Run from the repo root so outputs land in `output/`. 

## Python and CuPy solvers

```bash
python ref_NumPy.py    # writes output/phi_py.bin
python ref_CuPy.py     # writes output/phi_cupy.bin
```

Both accept `--ni/--nj/--nk` and `--niter` flags (see scripts for full args).

## Batch benchmarks (`test.py`)

`test.py` launches batch runs of all three solvers and compares outputs.

Defaults:
- Grid sizes: 16^3, 32^3, 64^3
- Iterations: 100, 200, 300
- C++ types: half, float, double

Run and log a batch:
```bash
python test.py --write-log --tag sample-batch --notes "The sun is up"
```

Behavior:
- Runs Python (`ref_NumPy.py`), CuPy (`ref_CuPy.py`), and C++ (`build/run`) for each case.
- Measures wall-clock runtime.
- Computes accuracy vs Python reference (max abs, L2, relative L2) and residual norms (L1, L2, max).
- Appends one JSONL record to `output/benchmarks.jsonl` if `--write-log` is set (includes host/CPU/GPU info and all batch cases).

Key CLI flags:
- `--run-py`, `--run-cupy`, `--run-cpp`: command strings to execute.
- `--cpp-type`, `--niter`, `--nijk`, `--ni/--nj/--nk`: present but overridden by batch defaults; adjust `GRID_SIZES`, `NITER_VALUES`, `CPP_TYPES` in `test.py` to change batches.
- `--write-log`, `--tag`, `--notes`, `--log-path`: logging controls.

## Plotting benchmarks

`plot.py` reads `output/benchmarks.jsonl` and emits two plots per log entry:
- `*_runtime_vs_niter.png`
- `*_accuracy_vs_niter.png`

```bash
python plot.py --log-path output/benchmarks.jsonl --outdir output/plots
```

Each log entry produces its own pair of plots (tag is used in filenames).
