/// Comprehensive benchmark: CPU vs GPU timing for every option type.
///
/// Each measurement is repeated N_RUNS_CPU (3) or N_RUNS_GPU (5) times.
/// An IQR-based outlier filter is applied to the collected timings before
/// computing median, mean, min, max, and std_dev.
///
/// Output: CSV to stdout with columns:
///   model,option_type,engine,n_paths,
///   time_ms_median,time_ms_mean,time_ms_min,time_ms_max,time_ms_std,
///   price,se
///
/// Usage:
///   ./benchmark_all_options > benchmark_results.csv

#include "engine/cpu_engine.h"
#include "engine/gpu_engine.h"
#include "models/gbm.h"
#include "models/heston.h"
#include "pricing/barrier.h"
#include "pricing/european.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Market parameters
// ---------------------------------------------------------------------------
static constexpr double SPOT       = 100.0;
static constexpr double STRIKE     = 105.0;
static constexpr double RATE       = 0.05;
static constexpr double VOL        = 0.20;
static constexpr double MATURITY   = 1.0;
static constexpr double BARRIER    = 115.0;          // up-and-out
static constexpr std::size_t STEPS = 252;
static constexpr unsigned long long SEED = 42ULL;

// Heston params
static constexpr double H_V0    = 0.04;
static constexpr double H_KAPPA = 2.0;
static constexpr double H_THETA = 0.04;
static constexpr double H_XI    = 0.30;
static constexpr double H_RHO   = -0.70;

// Run counts
static constexpr int N_RUNS_CPU = 3;
static constexpr int N_RUNS_GPU = 5;

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Statistics: IQR-based outlier removal, then median/mean/min/max/std
// ---------------------------------------------------------------------------
struct Stats {
    double median;
    double mean;
    double mn;      // min
    double mx;      // max
    double std_dev;
};

static Stats compute_stats(std::vector<double> times) {
    assert(!times.empty());
    std::sort(times.begin(), times.end());
    const std::size_t n = times.size();

    // Compute IQR (use linear interpolation for Q1/Q3)
    auto quantile = [&](double p) -> double {
        double idx = p * (n - 1);
        std::size_t lo = static_cast<std::size_t>(idx);
        std::size_t hi = lo + 1 < n ? lo + 1 : lo;
        double frac = idx - lo;
        return times[lo] * (1.0 - frac) + times[hi] * frac;
    };
    double q1  = quantile(0.25);
    double q3  = quantile(0.75);
    double iqr = q3 - q1;

    // Filter: keep values in [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    double lo_fence = q1 - 1.5 * iqr;
    double hi_fence = q3 + 1.5 * iqr;
    std::vector<double> filtered;
    filtered.reserve(n);
    for (double t : times) {
        if (t >= lo_fence && t <= hi_fence) filtered.push_back(t);
    }
    // Fallback: if all samples were filtered (degenerate IQR=0), use all
    if (filtered.empty()) filtered = times;

    const std::size_t m = filtered.size();
    // Median
    double med = (m % 2 == 1) ? filtered[m / 2]
                               : 0.5 * (filtered[m / 2 - 1] + filtered[m / 2]);
    // Mean
    double sum = std::accumulate(filtered.begin(), filtered.end(), 0.0);
    double mu  = sum / m;
    // Std dev (population)
    double sq_sum = 0.0;
    for (double t : filtered) sq_sum += (t - mu) * (t - mu);
    double sigma = (m > 1) ? std::sqrt(sq_sum / m) : 0.0;

    return { med, mu, filtered.front(), filtered.back(), sigma };
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------
static void print_row(const char* model, const char* option_type, const char* engine,
                       std::size_t n_paths, Stats s, double price, double se)
{
    std::printf("%s,%s,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f\n",
                model, option_type, engine, n_paths,
                s.median, s.mean, s.mn, s.mx, s.std_dev,
                price, se);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// GBM benchmarks
// ---------------------------------------------------------------------------
static void bench_gbm_cpu(std::size_t n_paths, int n_runs) {
    GBM<double> model(RATE, VOL);
    double price = 0.0, se = 0.0;

    // European
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            CPUEngine eng(n_paths, STEPS, static_cast<unsigned long>(SEED));
            auto t0 = Clock::now();
            auto res = eng.price_european(model, SPOT, STRIKE, MATURITY, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("GBM", "European", "CPU", n_paths, compute_stats(times), price, se);
    }
    // Asian
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            CPUEngine eng(n_paths, STEPS, static_cast<unsigned long>(SEED));
            auto t0 = Clock::now();
            auto res = eng.price_asian(model, SPOT, STRIKE, MATURITY, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("GBM", "Asian", "CPU", n_paths, compute_stats(times), price, se);
    }
    // Barrier
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            CPUEngine eng(n_paths, STEPS, static_cast<unsigned long>(SEED));
            auto t0 = Clock::now();
            auto res = eng.price_barrier(model, SPOT, STRIKE, MATURITY, BARRIER,
                                          BarrierType::UpAndOut, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("GBM", "Barrier", "CPU", n_paths, compute_stats(times), price, se);
    }
}

static void bench_gbm_gpu(std::size_t n_paths, int n_runs) {
    double price = 0.0, se = 0.0;

    // European
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            GPUEngine eng(n_paths, STEPS, SEED);
            auto t0 = Clock::now();
            auto res = eng.price_european_gbm(SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("GBM", "European", "GPU", n_paths, compute_stats(times), price, se);
    }
    // Asian
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            GPUEngine eng(n_paths, STEPS, SEED);
            auto t0 = Clock::now();
            auto res = eng.price_asian_gbm(SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("GBM", "Asian", "GPU", n_paths, compute_stats(times), price, se);
    }
    // Barrier
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            GPUEngine eng(n_paths, STEPS, SEED);
            auto t0 = Clock::now();
            auto res = eng.price_barrier_gbm(SPOT, STRIKE, RATE, VOL, MATURITY,
                                              BARRIER, BarrierType::UpAndOut, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("GBM", "Barrier", "GPU", n_paths, compute_stats(times), price, se);
    }
}

// ---------------------------------------------------------------------------
// Heston benchmarks
// ---------------------------------------------------------------------------
static void bench_heston_cpu(std::size_t n_paths, int n_runs) {
    Heston<double> model(RATE, H_V0, H_KAPPA, H_THETA, H_XI, H_RHO);
    double price = 0.0, se = 0.0;

    // European
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            CPUEngine eng(n_paths, STEPS, static_cast<unsigned long>(SEED));
            auto t0 = Clock::now();
            auto res = eng.price_european(model, SPOT, STRIKE, MATURITY, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("Heston", "European", "CPU", n_paths, compute_stats(times), price, se);
    }
    // Asian
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            CPUEngine eng(n_paths, STEPS, static_cast<unsigned long>(SEED));
            auto t0 = Clock::now();
            auto res = eng.price_asian(model, SPOT, STRIKE, MATURITY, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("Heston", "Asian", "CPU", n_paths, compute_stats(times), price, se);
    }
    // Barrier
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            CPUEngine eng(n_paths, STEPS, static_cast<unsigned long>(SEED));
            auto t0 = Clock::now();
            auto res = eng.price_barrier(model, SPOT, STRIKE, MATURITY, BARRIER,
                                          BarrierType::UpAndOut, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("Heston", "Barrier", "CPU", n_paths, compute_stats(times), price, se);
    }
}

static void bench_heston_gpu(std::size_t n_paths, int n_runs) {
    double price = 0.0, se = 0.0;

    // European
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            GPUEngine eng(n_paths, STEPS, SEED);
            auto t0 = Clock::now();
            auto res = eng.price_european_heston(SPOT, STRIKE, RATE, MATURITY,
                                                  H_V0, H_KAPPA, H_THETA, H_XI, H_RHO,
                                                  OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("Heston", "European", "GPU", n_paths, compute_stats(times), price, se);
    }
    // Asian
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            GPUEngine eng(n_paths, STEPS, SEED);
            auto t0 = Clock::now();
            auto res = eng.price_asian_heston(SPOT, STRIKE, RATE, MATURITY,
                                               H_V0, H_KAPPA, H_THETA, H_XI, H_RHO,
                                               OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("Heston", "Asian", "GPU", n_paths, compute_stats(times), price, se);
    }
    // Barrier
    {
        std::vector<double> times;
        times.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            GPUEngine eng(n_paths, STEPS, SEED);
            auto t0 = Clock::now();
            auto res = eng.price_barrier_heston(SPOT, STRIKE, RATE, MATURITY,
                                                 H_V0, H_KAPPA, H_THETA, H_XI, H_RHO,
                                                 BARRIER, BarrierType::UpAndOut, OptionType::Call);
            times.push_back(elapsed_ms(t0));
            price = res.price; se = res.standard_error;
        }
        print_row("Heston", "Barrier", "GPU", n_paths, compute_stats(times), price, se);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // GPU warm-up: avoids measuring JIT / CUDA context init on the first timed run.
    {
        GPUEngine warmup(1000, 10, 1ULL);
        warmup.price_european_gbm(SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    }

    std::printf("model,option_type,engine,n_paths,"
                "time_ms_median,time_ms_mean,time_ms_min,time_ms_max,time_ms_std,"
                "price,se\n");
    std::fflush(stdout);

    // CPU: 3 runs each; cap at 1M paths (10M would take ~30 min per call)
    std::size_t cpu_counts[] = { 10'000, 100'000, 1'000'000 };
    // GPU: 5 runs each; go up to 10M paths
    std::size_t gpu_counts[] = { 10'000, 100'000, 1'000'000, 10'000'000 };

    for (std::size_t n : cpu_counts) {
        bench_gbm_cpu(n, N_RUNS_CPU);
        bench_heston_cpu(n, N_RUNS_CPU);
    }

    for (std::size_t n : gpu_counts) {
        bench_gbm_gpu(n, N_RUNS_GPU);
        bench_heston_gpu(n, N_RUNS_GPU);
    }

    return 0;
}
