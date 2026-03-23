"""Plot RMSE vs number of paths for each variance reduction method.

Usage:
    ./benchmark_convergence > convergence_results.csv
    python benchmarks/plot_convergence.py convergence_results.csv

Saves 'convergence_plot.png' in the current directory.
"""

import sys
import csv
import math
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
input_file = sys.argv[1] if len(sys.argv) > 1 else "convergence_results.csv"

# data[method][n_paths] = rmse
data = collections.defaultdict(dict)

with open(input_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        method  = row["method"]
        n       = int(row["n_paths"])
        rmse    = float(row["rmse"])
        data[method][n] = rmse

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
STYLES = {
    "plain":      dict(color="#4c72b0", linestyle="-",  marker="o", label="Plain MC"),
    "antithetic": dict(color="#dd8452", linestyle="--", marker="s", label="Antithetic"),
    "cv":         dict(color="#55a868", linestyle="-.", marker="^", label="Control Variate"),
    "both":       dict(color="#c44e52", linestyle=":",  marker="D", label="Antithetic + CV"),
}

fig, ax = plt.subplots(figsize=(8, 5))

for method, style in STYLES.items():
    if method not in data:
        continue
    ns   = sorted(data[method].keys())
    rmse = [data[method][n] for n in ns]
    ax.loglog(ns, rmse, **style, linewidth=1.8, markersize=6)

# Reference 1/sqrt(N) line anchored to the plain MC curve at max N.
plain_data = data.get("plain", {})
if plain_data:
    ns_plain = sorted(plain_data.keys())
    n_ref    = ns_plain[-1]
    rmse_ref = plain_data[n_ref]
    ns_ref   = [ns_plain[0], n_ref]
    ref_vals = [rmse_ref * math.sqrt(n_ref / n) for n in ns_ref]
    ax.loglog(ns_ref, ref_vals, color="grey", linestyle="--", linewidth=1,
              label=r"$1/\sqrt{N}$ reference")

ax.set_xlabel("Number of paths  $N$", fontsize=12)
ax.set_ylabel("RMSE  (option price)", fontsize=12)
ax.set_title("Convergence of MC estimators: RMSE vs path count", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("convergence_plot.png", dpi=150)
print("Saved convergence_plot.png")
