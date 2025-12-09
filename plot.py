#!/usr/bin/env python
"""
Plot benchmark results from output/benchmarks.jsonl.

Generates runtime and accuracy plots across grid sizes, iterations, and C++ types.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

SOLVER_COLORS = {
    "python": "#f2c14f",  # yellow
    "cupy": "#d7263d",    # red
}

CPP_TYPE_COLORS = {
    "half": "#9be564",
    "float16": "#9be564",
    "float": "#66b447",
    "float32": "#66b447",
    "double": "#2e8b57",
}

GRID_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load_entries(path: Path):
    raw = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return raw


def grid_label(grid):
    return f"{grid['ni']}x{grid['nj']}x{grid['nk']}"


def host_label(host):
    if not host:
        return "host:unknown"
    parts = []
    if host.get("hostname"):
        parts.append(host["hostname"])
    if host.get("cpu"):
        parts.append(host["cpu"])
    if host.get("gpus"):
        parts.append(f"GPU:{'|'.join(host['gpus'])}")
    return "\n".join(parts)


def plot_record(record, outdir: Path, runtime_max=None):
    host = host_label(record.get("host"))
    tag = record.get("tag") or "benchmark"
    cases = record.get("cases", [])

    # Map grid to marker
    grids = sorted({grid_label(c["grid"]) for c in cases})
    grid_marker_map = {g: GRID_MARKERS[i % len(GRID_MARKERS)] for i, g in enumerate(grids)}

    # Runtime plot
    plt.figure(figsize=(10, 6))
    series = defaultdict(list)
    for case in cases:
        grid = grid_label(case["grid"])
        niter = case.get("niter")
        runs = case.get("runs", {})
        cpp_type = case.get("cpp_type") or case.get("cpp_dtype")
        for solver in ("python", "cupy", "cpp"):
            if solver not in runs:
                continue
            rt = runs[solver].get("runtime_s")
            if rt is None or niter is None:
                continue
            if runtime_max is not None and rt > runtime_max:
                rt = runtime_max
            key = (solver, grid, cpp_type if solver == "cpp" else None)
            series[key].append((niter, rt))

    for (solver, grid, cpp_type), pts in series.items():
        pts.sort(key=lambda t: t[0])
        xs, ys = zip(*pts)
        marker = grid_marker_map[grid]
        if solver == "cpp":
            color = CPP_TYPE_COLORS.get(cpp_type, "#2e8b57")
            label = f"cpp ({cpp_type}) {grid}"
        else:
            color = SOLVER_COLORS[solver]
            label = f"{solver} {grid}"
        plt.plot(xs, ys, marker=marker, color=color, label=label, linestyle="-")

    plt.xlabel("Iterations (niter)")
    plt.ylabel("Runtime (s)")
    plt.title(f"Runtime vs iterations\n{host}")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Deduplicate legend
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=2)
    out_path_rt = outdir / f"{tag}_runtime_vs_niter.png"
    plt.tight_layout()
    plt.savefig(out_path_rt, dpi=200)
    plt.close()

    # Accuracy plot (relative L2)
    plt.figure(figsize=(10, 6))
    series_acc = defaultdict(list)
    for case in cases:
        grid = grid_label(case["grid"])
        niter = case.get("niter")
        acc = case.get("accuracy", {})
        cpp_type = case.get("cpp_type") or case.get("cpp_dtype")

        if "cupy_vs_py" in acc:
            rel = acc["cupy_vs_py"].get("rel_l2")
            if rel is not None and niter is not None:
                key = ("cupy", grid, None)
                series_acc[key].append((niter, rel))
        if "cpp_vs_py" in acc:
            rel = acc["cpp_vs_py"].get("rel_l2")
            if rel is not None and niter is not None:
                key = ("cpp", grid, cpp_type)
                series_acc[key].append((niter, rel))

    for (solver, grid, cpp_type), pts in series_acc.items():
        pts.sort(key=lambda t: t[0])
        xs, ys = zip(*pts)
        marker = grid_marker_map[grid]
        if solver == "cpp":
            color = CPP_TYPE_COLORS.get(cpp_type, "#2e8b57")
            label = f"cpp ({cpp_type}) {grid}"
        else:
            color = SOLVER_COLORS[solver]
            label = f"{solver} {grid}"
        plt.semilogy(xs, ys, marker=marker, color=color, linestyle="-", label=label)

    plt.xlabel("Iterations (niter)")
    plt.ylabel("Relative L2 error vs py")
    plt.title(f"Accuracy vs iterations\n{host}")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=2)
    out_path_acc = outdir / f"{tag}_accuracy_vs_niter.png"
    plt.tight_layout()
    plt.savefig(out_path_acc, dpi=200)
    plt.close()

    return [out_path_rt, out_path_acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-path", default="output/benchmarks.jsonl", help="Path to benchmarks jsonl")
    ap.add_argument("--outdir", default="output/plots", help="Where to write plots")
    ap.add_argument(
        "--runtime-max",
        type=float,
        default=None,
        help="If set, cap runtime values at this threshold for plotting",
    )
    args = ap.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    records = load_entries(log_path)
    if not records:
        raise SystemExit("No entries found in log.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    for rec in records:
        # Support older format that used "batch"
        if "batch" in rec and "cases" not in rec:
            rec["cases"] = rec.pop("batch")
        if not rec.get("cases"):
            continue
        all_paths.extend(plot_record(rec, outdir, runtime_max=args.runtime_max))

    print("Plots written:")
    for p in all_paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
