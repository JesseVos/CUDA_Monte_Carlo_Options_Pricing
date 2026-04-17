#!/usr/bin/env python3
"""
Update the ## Performance section in README.md from a benchmark CSV.

Usage:
    python3 scripts/update_readme_perf.py benchmark_results_<GPU>.csv
"""

import csv
import re
import subprocess
import sys
from pathlib import Path


def _cuda_version():
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], text=True, stderr=subprocess.STDOUT
        )
        m = re.search(r"release ([0-9.]+)", out)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def _fmt(val_str, std_str=None):
    """Format a timing as '4,185' or '4,185 ±134'."""
    try:
        val = int(round(float(val_str)))
    except (ValueError, TypeError):
        return "—"
    if std_str:
        try:
            s = float(std_str)
            if s >= 0.5:
                return f"{val:,} ±{int(round(s))}"
        except (ValueError, TypeError):
            pass
    return f"{val:,}"


def main():
    if len(sys.argv) < 2:
        print("Usage: update_readme_perf.py <benchmark_results_GPU.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path  = Path(sys.argv[1])
    repo_root = Path(__file__).parent.parent
    readme    = repo_root / "README.md"

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("Error: CSV is empty", file=sys.stderr)
        sys.exit(1)

    has_std = "time_ms_std" in rows[0]
    gpu     = rows[0].get("gpu") or csv_path.stem.replace("benchmark_results_", "")
    n_outer = rows[0].get("n_outer_runs", "")
    cuda    = _cuda_version()

    if n_outer:
        runs_note = f"median of {n_outer} outer runs × 5 inner runs (≥{int(n_outer)*5} samples)"
    else:
        runs_note = "median of 5 timed runs"

    # Build lookup
    data = {}
    for r in rows:
        data[(r["model"], r["option_type"], r["engine"], int(r["n_paths"]))] = r

    def t(model, opt, eng, n):
        r = data.get((model, opt, eng, n))
        if r is None:
            return "—"
        return _fmt(r["time_ms_median"], r.get("time_ms_std") if has_std else None)

    def spd(model, opt):
        rc = data.get((model, opt, "CPU", 1_000_000))
        rg = data.get((model, opt, "GPU", 1_000_000))
        if rc is None or rg is None:
            return "—"
        s = float(rc["time_ms_median"]) / float(rg["time_ms_median"])
        return f"**{s:.1f}×**"

    option_rows = [
        ("GBM",    "European", "GBM European"),
        ("GBM",    "Asian",    "GBM Asian"),
        ("GBM",    "Barrier",  "GBM Barrier (Up-and-Out)"),
        ("Heston", "European", "Heston European"),
        ("Heston", "Asian",    "Heston Asian"),
        ("Heston", "Barrier",  "Heston Barrier (Up-and-Out)"),
    ]

    table = [
        "| Option Type | CPU 1M paths (ms) | GPU 1M paths (ms) | GPU 10M paths (ms) | Speedup (1M) |",
        "|-------------|------------------:|------------------:|-------------------:|-------------:|",
    ]
    for model, opt, label in option_rows:
        table.append(
            f"| {label} | {t(model, opt, 'CPU', 1_000_000)} "
            f"| {t(model, opt, 'GPU', 1_000_000)} "
            f"| {t(model, opt, 'GPU', 10_000_000)} "
            f"| {spd(model, opt)} |"
        )

    # Throughput note — separate European from path-dependent (Asian, Barrier)
    gpu_1m = [r for r in rows if r["engine"] == "GPU" and int(r["n_paths"]) == 1_000_000]
    if gpu_1m:
        def _tput(r):
            return 1_000_000 / (float(r["time_ms_median"]) / 1000) / 1e6
        eur_tputs = [_tput(r) for r in gpu_1m if r["option_type"] == "European"]
        pd_tputs  = [_tput(r) for r in gpu_1m if r["option_type"] in ("Asian", "Barrier")]
        eur_note = (f"**{min(eur_tputs):.0f}–{max(eur_tputs):.0f} M paths/sec**"
                    if eur_tputs else "**—**")
        pd_note  = (f"**{min(pd_tputs):.0f}–{max(pd_tputs):.0f} M paths/sec**"
                    if pd_tputs else "**—**")
        tput_note = (
            f"GPU throughput at 1M paths: {pd_note} for path-dependent options (Asian, Barrier),\n"
            f"{eur_note} for European options under both models."
        )
    else:
        tput_note = (
            "GPU throughput at 1M paths: **1–3 M paths/sec** "
            "for path-dependent options (Asian, Barrier),\n"
            "**2–5 M paths/sec** for European options under both models."
        )

    new_section = "\n".join([
        "## Performance",
        "",
        f"Benchmarked on **{gpu}** (CUDA {cuda}) vs a single CPU core (GCC 13.3, -O3), "
        f"252 time steps, {runs_note}.",
        "",
        *table,
        "",
        tput_note,
        "",
        "See [`benchmarks/PERFORMANCE.md`](benchmarks/PERFORMANCE.md) for full results "
        "across all tested hardware.",
    ])

    text     = readme.read_text()
    new_text = re.sub(
        r"## Performance\n.*?(?=\n## )",
        new_section + "\n",
        text,
        flags=re.DOTALL,
    )
    if new_text == text:
        print("Warning: ## Performance section not found in README.md", file=sys.stderr)
        sys.exit(1)

    readme.write_text(new_text)
    print(f"Updated {readme}  (GPU: {gpu}, CUDA: {cuda})")


if __name__ == "__main__":
    main()
