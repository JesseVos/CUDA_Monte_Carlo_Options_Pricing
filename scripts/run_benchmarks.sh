#!/usr/bin/env bash
# run_benchmarks.sh — automated GPU Monte Carlo benchmark harness.
#
# Usage:
#   ./scripts/run_benchmarks.sh [options]
#
# Options:
#   -n, --runs N       Outer repetitions (default: 3, max: 5).
#                      Each outer run invokes benchmark_all_options which itself
#                      does 3 (CPU) / 5 (GPU) inner timed runs with IQR filtering.
#                      Total samples at 1M GPU: N × 5.
#   --skip-cpu         Intent flag: signals that CPU rows are less important.
#                      NOTE: benchmark_all_options has no --skip-cpu flag, so the
#                      binary still runs CPU paths. The flag is a no-op today and
#                      is reserved for a future C++ change.
#   --build            Rebuild via cmake before benchmarking.
#   --update-readme    Patch README.md performance table after aggregation.
#
# Examples:
#   ./scripts/run_benchmarks.sh --runs 1 --skip-cpu       # quick smoke test
#   ./scripts/run_benchmarks.sh --runs 3 --update-readme  # standard run + README
#   ./scripts/run_benchmarks.sh --runs 5 --build          # full run with rebuild

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
RUNS=3
SKIP_CPU=0
DO_BUILD=0
UPDATE_README=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--runs)
            RUNS="$2"
            shift 2
            ;;
        --skip-cpu)
            SKIP_CPU=1
            shift
            ;;
        --build)
            DO_BUILD=1
            shift
            ;;
        --update-readme)
            UPDATE_README=1
            shift
            ;;
        -h|--help)
            sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if ! [[ "$RUNS" =~ ^[1-5]$ ]]; then
    echo "Error: --runs must be an integer between 1 and 5 (got: $RUNS)" >&2
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
RESULTS_BASE="$REPO_ROOT/benchmarks/results"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

# ── GPU detection ─────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: nvidia-smi not found. CUDA toolkit required." >&2
    exit 1
fi

GPU_FULL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
# Extract a short identifier matching common GPU families
GPU_SHORT=$(echo "$GPU_FULL" \
    | grep -oE '(A100|A10G?|H100|H200|GH200|L[0-9]+|RTX[[:space:]]*[0-9A-Z]+|Titan[[:space:]]*[A-Za-z0-9]+|V100|T[0-9]+)' \
    | head -1 \
    | tr -d '[:space:]')
if [[ -z "$GPU_SHORT" ]]; then
    # Fallback: sanitize the full name into a filename-safe string
    GPU_SHORT=$(echo "$GPU_FULL" | sed 's/NVIDIA[[:space:]]*//' | tr ' /' '__' | sed 's/__*/_/g' | sed 's/_$//')
fi

NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "unknown")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CPU_MODEL=$(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")

echo "========================================================================"
echo "  GPU Monte Carlo Benchmark Suite"
echo "========================================================================"
printf "  GPU    : %s  (%s)\n" "$GPU_FULL" "$GPU_SHORT"
printf "  CUDA   : %s  (driver %s)\n" "$NVCC_VER" "$DRIVER_VER"
printf "  CPU    : %s\n" "$CPU_MODEL"
printf "  Date   : %s\n" "$(date)"
printf "  Runs   : %d outer repetition(s)  ×  5 inner GPU runs  =  %d GPU samples\n" \
    "$RUNS" "$((RUNS * 5))"
if (( SKIP_CPU )); then
    echo "  Note   : --skip-cpu passed (flag reserved; CPU paths still run)"
fi
echo "========================================================================"
echo ""

# ── Optional rebuild ──────────────────────────────────────────────────────────
if (( DO_BUILD )); then
    echo "[benchmark] Building..."
    cmake --build "$BUILD_DIR" --parallel
    echo ""
fi

# ── Check binaries ────────────────────────────────────────────────────────────
for bin in benchmark_all_options benchmark_convergence benchmark_heston_smile; do
    if [[ ! -f "$BUILD_DIR/$bin" ]]; then
        echo "Error: $BUILD_DIR/$bin not found. Run with --build or build manually first." >&2
        exit 1
    fi
done

# ── Create run directory ──────────────────────────────────────────────────────
RUN_DIR="$RESULTS_BASE/$GPU_SHORT/$TIMESTAMP"
mkdir -p "$RUN_DIR"
echo "[benchmark] Results dir : $RUN_DIR"
echo ""

# ── Outer loop ────────────────────────────────────────────────────────────────
echo "[benchmark] benchmark_all_options  ($RUNS outer run(s))"
for i in $(seq 1 "$RUNS"); do
    printf "[benchmark]   run %d/%d ... " "$i" "$RUNS"
    "$BUILD_DIR/benchmark_all_options" > "$RUN_DIR/run_${i}.csv"
    # Print a quick progress indicator: GPU 1M GBM European median
    T=$(awk -F, '$1=="GBM" && $2=="European" && $3=="GPU" && $4==1000000 {print $5}' \
        "$RUN_DIR/run_${i}.csv")
    printf "done  (GBM European 1M GPU: %s ms)\n" "$T"
done
echo ""

# ── Aggregate across outer runs ───────────────────────────────────────────────
echo "[benchmark] Aggregating..."
python3 "$SCRIPT_DIR/aggregate_benchmarks.py" "$RUN_DIR" "$GPU_SHORT"
echo ""

RESULTS_CSV="$REPO_ROOT/benchmarks/benchmark_results_${GPU_SHORT}.csv"

# ── Supplementary benchmarks (once each, not repeated in outer loop) ──────────
echo "[benchmark] benchmark_convergence..."
"$BUILD_DIR/benchmark_convergence" > "$RUN_DIR/convergence.csv"

echo "[benchmark] benchmark_heston_smile..."
"$BUILD_DIR/benchmark_heston_smile" > "$RUN_DIR/heston_smile.csv"
echo ""

# ── Plots ─────────────────────────────────────────────────────────────────────
echo "[benchmark] Generating plots  (output → repo root)"
cd "$REPO_ROOT"
python3 python/plot_benchmarks.py "$RESULTS_CSV" \
    || echo "[benchmark] Warning: plot_benchmarks.py failed (check pandas/matplotlib)"
python3 benchmarks/plot_convergence.py "$RUN_DIR/convergence.csv" \
    || echo "[benchmark] Warning: plot_convergence.py failed"
python3 benchmarks/plot_heston_smile.py "$RUN_DIR/heston_smile.csv" \
    || echo "[benchmark] Warning: plot_heston_smile.py failed"
echo ""

# ── Generate PERFORMANCE.md ───────────────────────────────────────────────────
echo "[benchmark] Generating benchmarks/PERFORMANCE.md..."
python3 "$SCRIPT_DIR/generate_perf_doc.py"
echo ""

# ── Optional README update ────────────────────────────────────────────────────
if (( UPDATE_README )); then
    echo "[benchmark] Updating README.md performance table..."
    python3 "$SCRIPT_DIR/update_readme_perf.py" "$RESULTS_CSV"
    echo ""
fi

echo "========================================================================"
echo "  Benchmark complete."
echo "  Per-hardware CSV : benchmarks/benchmark_results_${GPU_SHORT}.csv"
echo "  Run directory    : $RUN_DIR"
echo "  PERFORMANCE.md   : benchmarks/PERFORMANCE.md"
if (( UPDATE_README )); then
    echo "  README.md        : performance table updated"
fi
echo "========================================================================"
