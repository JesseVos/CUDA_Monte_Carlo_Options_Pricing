#!/usr/bin/env python3
"""
Aggregate N outer benchmark runs into a single per-hardware CSV.

Usage:
    python3 scripts/aggregate_benchmarks.py <rundir> <GPU_SHORT>

Reads:  <rundir>/run_1.csv, run_2.csv, ... (all run_*.csv files, auto-detected)
Writes: <rundir>/aggregated.csv
        benchmark_results_<GPU_SHORT>.csv  (repo root)
Prints: summary table (median ± cross-run std, GPU speedup at 1M paths)
"""

import sys
import csv
import glob
import statistics
from pathlib import Path


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def main():
    if len(sys.argv) < 3:
        print("Usage: aggregate_benchmarks.py <rundir> <GPU_SHORT>", file=sys.stderr)
        sys.exit(1)

    rundir    = Path(sys.argv[1])
    gpu_short = sys.argv[2]

    run_files = sorted(glob.glob(str(rundir / "run_*.csv")))
    if not run_files:
        print(f"Error: no run_*.csv files in {rundir}", file=sys.stderr)
        sys.exit(1)

    n_runs = len(run_files)
    print(f"Aggregating {n_runs} run(s) from {rundir}")

    # key → list[row dict] across all runs
    rows_by_key = {}
    last_row    = {}
    for run_path in run_files:
        with open(run_path) as f:
            for row in csv.DictReader(f):
                key = (row["model"], row["option_type"], row["engine"], row["n_paths"])
                rows_by_key.setdefault(key, []).append(row)
                last_row[key] = row

    # Output columns (drop-in for plot_benchmarks.py which reads median/min/max/std)
    out_cols = [
        "gpu", "model", "option_type", "engine", "n_paths",
        "time_ms_median", "time_ms_mean", "time_ms_min", "time_ms_max", "time_ms_std",
        "n_outer_runs", "price", "se",
    ]

    out_rows = []
    # Sort: CPU before GPU, then ascending n_paths, then model, option_type
    sorted_keys = sorted(
        rows_by_key.keys(),
        key=lambda k: (k[2] != "CPU", int(k[3]), k[0], k[1]),
    )
    for key in sorted_keys:
        run_rows = rows_by_key[key]
        model, option_type, engine, n_paths = key
        medians = [float(r["time_ms_median"]) for r in run_rows]
        last    = last_row[key]
        out_rows.append({
            "gpu":            gpu_short,
            "model":          model,
            "option_type":    option_type,
            "engine":         engine,
            "n_paths":        n_paths,
            "time_ms_median": f"{_median(medians):.3f}",
            "time_ms_mean":   f"{_mean(medians):.3f}",
            "time_ms_min":    f"{min(medians):.3f}",
            "time_ms_max":    f"{max(medians):.3f}",
            "time_ms_std":    f"{_std(medians):.3f}",
            "n_outer_runs":   str(n_runs),
            "price":          last["price"],
            "se":             last["se"],
        })

    # Write aggregated.csv inside the run directory
    agg_path = rundir / "aggregated.csv"
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Written: {agg_path}")

    # Copy to benchmarks/benchmark_results_<GPU>.csv
    repo_root = Path(__file__).parent.parent
    out_csv   = repo_root / "benchmarks" / f"benchmark_results_{gpu_short}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Written: {out_csv}")

    # ── Print summary table ────────────────────────────────────────────────
    # Build lookup for speedup computation
    lookup = {
        (r["model"], r["option_type"], r["engine"], int(r["n_paths"])): r
        for r in out_rows
    }

    print()
    print("=" * 74)
    print(f"  Performance Summary — {gpu_short}  ({n_runs} outer run(s))")
    print("=" * 74)
    print(f"  {'Option':<24} {'Engine':<5} {'N paths':>10}  "
          f"{'Median (ms)':>11}  {'±Std':>7}  {'Speedup':>8}")
    print("  " + "-" * 70)

    cpu_med = {}  # (model, opt) → median ms at 1M
    gpu_med = {}
    for r in out_rows:
        if int(r["n_paths"]) == 1_000_000:
            key2 = (r["model"], r["option_type"])
            t    = float(r["time_ms_median"])
            if r["engine"] == "CPU":
                cpu_med[key2] = t
            else:
                gpu_med[key2] = t

    for r in out_rows:
        n = int(r["n_paths"])
        if n not in {1_000_000, 10_000_000}:
            continue
        key2 = (r["model"], r["option_type"])
        t    = float(r["time_ms_median"])
        s    = float(r["time_ms_std"])
        lbl  = f"{r['model']} {r['option_type']}"

        if r["engine"] == "GPU" and n == 1_000_000 and key2 in cpu_med:
            spd = f"{cpu_med[key2] / t:.1f}×"
        else:
            spd = ""

        print(f"  {lbl:<24} {r['engine']:<5} {n:>10,}  "
              f"{t:>11.1f}  {s:>7.3f}  {spd:>8}")

    print("=" * 74)


if __name__ == "__main__":
    main()
