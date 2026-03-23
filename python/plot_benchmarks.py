#!/usr/bin/env python3
"""
Plot benchmark results from benchmark_all_options CSV output.

Usage:
    ./benchmark_all_options > benchmark_results.csv
    python plot_benchmarks.py benchmark_results.csv

Produces three figures:
  1. CPU vs GPU speedup by option type (log-log) with uncertainty band
  2. GPU throughput in paths/sec for all combinations
  3. CPU vs GPU wall-time comparison with min/max error bars
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
df = pd.read_csv(csv_path)

# Support both old single-column format and new multi-column format.
if "time_ms_median" not in df.columns and "time_ms" in df.columns:
    df["time_ms_median"] = df["time_ms"]
    df["time_ms_min"]    = df["time_ms"]
    df["time_ms_max"]    = df["time_ms"]
    df["time_ms_std"]    = 0.0

# Derived columns
df["throughput"] = df["n_paths"] / (df["time_ms_median"] / 1000.0)   # paths/sec
df["label"] = df["model"] + " " + df["option_type"]

cpu = df[df["engine"] == "CPU"].copy()
gpu = df[df["engine"] == "GPU"].copy()

# ---------------------------------------------------------------------------
# Figure 1 — Speedup vs path count (log-log), one line per option type
#             with shaded uncertainty band from min/max timings
# ---------------------------------------------------------------------------
common = pd.merge(
    cpu[["model", "option_type", "n_paths",
         "time_ms_median", "time_ms_min", "time_ms_max"]],
    gpu[["model", "option_type", "n_paths",
         "time_ms_median", "time_ms_min", "time_ms_max"]],
    on=["model", "option_type", "n_paths"],
    suffixes=("_cpu", "_gpu"),
)
# Median speedup = CPU_median / GPU_median
common["speedup"]     = common["time_ms_median_cpu"] / common["time_ms_median_gpu"]
# Conservative bound: fastest CPU / slowest GPU  →  lower speedup bound
common["speedup_lo"]  = common["time_ms_min_cpu"]   / common["time_ms_max_gpu"]
# Optimistic bound: slowest CPU / fastest GPU  →  upper speedup bound
common["speedup_hi"]  = common["time_ms_max_cpu"]   / common["time_ms_min_gpu"]
common["label"] = common["model"] + " " + common["option_type"]

fig1, ax1 = plt.subplots(figsize=(9, 5))
colors = plt.cm.tab10.colors
for i, (label, grp) in enumerate(common.groupby("label")):
    grp = grp.sort_values("n_paths")
    c = colors[i % len(colors)]
    ax1.plot(grp["n_paths"], grp["speedup"],
             marker="o", label=label, color=c)
    ax1.fill_between(grp["n_paths"], grp["speedup_lo"], grp["speedup_hi"],
                     alpha=0.12, color=c)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Number of Paths")
ax1.set_ylabel("GPU Speedup (×)")
ax1.set_title("CPU vs GPU Speedup by Option Type\n"
              "(shaded band = min/max timing uncertainty)")
ax1.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}×"))
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, which="both", alpha=0.3)
fig1.tight_layout()
fig1.savefig("speedup_by_option_type.png", dpi=150)
print("Saved speedup_by_option_type.png")

# ---------------------------------------------------------------------------
# Figure 2 — GPU throughput (paths/sec) bar chart at 1M paths
# ---------------------------------------------------------------------------
gpu_1m = gpu[gpu["n_paths"] == 1_000_000].copy()
gpu_1m["Mpaths_per_sec"] = gpu_1m["throughput"] / 1e6

fig2, ax2 = plt.subplots(figsize=(9, 5))
bar_labels = gpu_1m["label"].tolist()
bar_vals   = gpu_1m["Mpaths_per_sec"].tolist()
x2 = np.arange(len(bar_labels))
bars = ax2.bar(x2, bar_vals,
               color=[colors[i % len(colors)] for i in range(len(bar_labels))])
ax2.bar_label(bars, fmt="%.1f M", padding=3, fontsize=8)
ax2.set_ylabel("GPU Throughput (M paths / sec)")
ax2.set_title("GPU Throughput at 1M Paths per Option Type")
ax2.set_xticks(x2)
ax2.set_xticklabels(bar_labels, rotation=20, ha="right")
ax2.set_ylim(0, max(bar_vals) * 1.25)
ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout()
fig2.savefig("gpu_throughput.png", dpi=150)
print("Saved gpu_throughput.png")

# ---------------------------------------------------------------------------
# Figure 3 — CPU vs GPU time (ms) at 1M paths, grouped bar with error bars
# ---------------------------------------------------------------------------
def _1m(df_side):
    return df_side[df_side["n_paths"] == 1_000_000][
        ["label", "time_ms_median", "time_ms_min", "time_ms_max"]
    ].rename(columns={
        "time_ms_median": "median",
        "time_ms_min":    "mn",
        "time_ms_max":    "mx",
    })

cpu_1m = _1m(cpu)
gpu_1m_t = _1m(gpu)
timing = pd.merge(cpu_1m, gpu_1m_t, on="label", suffixes=("_cpu", "_gpu"))

x = np.arange(len(timing))
w = 0.35
fig3, ax3 = plt.subplots(figsize=(9, 5))

# CPU bars + error bars
b1 = ax3.bar(x - w/2, timing["median_cpu"], w, label="CPU", color="#4C72B0")
cpu_err = np.array([
    timing["median_cpu"] - timing["mn_cpu"],
    timing["mx_cpu"]    - timing["median_cpu"],
])
ax3.errorbar(x - w/2, timing["median_cpu"], yerr=cpu_err,
             fmt="none", color="black", capsize=4, linewidth=1)

# GPU bars + error bars
b2 = ax3.bar(x + w/2, timing["median_gpu"], w, label="GPU", color="#DD8452")
gpu_err = np.array([
    timing["median_gpu"] - timing["mn_gpu"],
    timing["mx_gpu"]    - timing["median_gpu"],
])
ax3.errorbar(x + w/2, timing["median_gpu"], yerr=gpu_err,
             fmt="none", color="black", capsize=4, linewidth=1)

ax3.set_xticks(x)
ax3.set_xticklabels(timing["label"], rotation=20, ha="right")
ax3.set_ylabel("Wall Time — median (ms)")
ax3.set_title("CPU vs GPU Wall Time at 1M Paths\n"
              "(error bars = min/max over repeated runs)")
ax3.set_yscale("log")
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
ax3.legend()
ax3.grid(axis="y", which="both", alpha=0.3)
fig3.tight_layout()
fig3.savefig("cpu_vs_gpu_timing.png", dpi=150)
print("Saved cpu_vs_gpu_timing.png")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
summary = common[common["n_paths"] == 1_000_000][
    ["label", "time_ms_median_cpu", "time_ms_median_gpu", "speedup"]
].copy()
# Attach std columns for both engines
cpu_std = cpu[cpu["n_paths"] == 1_000_000][["label", "time_ms_std"]].rename(
    columns={"time_ms_std": "std_cpu"})
gpu_std = gpu[gpu["n_paths"] == 1_000_000][["label", "time_ms_std"]].rename(
    columns={"time_ms_std": "std_gpu"})
summary = summary.merge(cpu_std, on="label").merge(gpu_std, on="label")
summary = summary.sort_values("speedup", ascending=False)

print("\n=== 1M paths — CPU vs GPU summary (median ± std, ms) ===")
header = f"{'Option':<22} {'CPU ms':>10} {'±':>2} {'GPU ms':>8} {'±':>2} {'Speedup':>8}"
print(header)
print("-" * len(header))
for _, row in summary.iterrows():
    print(f"{row['label']:<22} {row['time_ms_median_cpu']:>10.1f} "
          f"± {row['std_cpu']:<6.1f} {row['time_ms_median_gpu']:>8.1f} "
          f"± {row['std_gpu']:<6.1f} {row['speedup']:>7.1f}×")
