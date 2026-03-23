/// Convergence benchmark: RMSE vs number of paths for four MC methods.
///
/// Methods compared:
///   plain     — standard Monte Carlo
///   antithetic — antithetic variates
///   cv         — control variates (S_T as control)
///   both       — antithetic + control variates
///
/// Output: CSV to stdout with columns:
///   method, n_paths, rmse
///
/// Usage:
///   ./benchmark_convergence > convergence_results.csv

#include "engine/gpu_engine.h"
#include "utils/black_scholes.h"
#include "variance/variance_reduction.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

static constexpr double SPOT       = 100.0;
static constexpr double STRIKE     = 105.0;
static constexpr double RATE       = 0.05;
static constexpr double VOLATILITY = 0.20;
static constexpr double MATURITY   = 1.0;
static constexpr std::size_t STEPS = 252;
static constexpr int REPS          = 30;   // replications per (method, N)

struct BenchmarkResult {
    const char* method;
    std::size_t n_paths;
    double rmse;
};

static double compute_rmse(
    std::size_t n_paths,
    const VarianceReductionConfig& vr,
    double bs_price)
{
    double sum_sq_err = 0.0;
    for (int rep = 0; rep < REPS; ++rep) {
        unsigned long long seed = static_cast<unsigned long long>(rep + 1) * 1000ULL;
        GPUEngine engine(n_paths, STEPS, seed);
        auto result = engine.price_european_gbm(
            SPOT, STRIKE, RATE, VOLATILITY, MATURITY,
            OptionType::Call, vr);
        double err = result.price - bs_price;
        sum_sq_err += err * err;
    }
    return std::sqrt(sum_sq_err / REPS);
}

int main()
{
    auto bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double bs_price = bs.price;

    // GPU warm-up (avoids measuring JIT overhead in first timing).
    {
        GPUEngine warmup(1000, STEPS, 1ULL);
        VarianceReductionConfig none;
        warmup.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY, MATURITY,
                                  OptionType::Call, none);
    }

    std::vector<std::size_t> path_counts = {
        10'000, 50'000, 100'000, 500'000, 1'000'000
    };

    struct Method {
        const char* name;
        VarianceReductionConfig vr;
    };

    VarianceReductionConfig plain{};
    VarianceReductionConfig av{};   av.antithetic = true;
    VarianceReductionConfig cv{};   cv.control_variate = true;
    VarianceReductionConfig both{}; both.antithetic = true; both.control_variate = true;

    Method methods[] = {
        {"plain",      plain},
        {"antithetic", av},
        {"cv",         cv},
        {"both",       both},
    };

    // Print CSV header.
    std::printf("method,n_paths,rmse\n");

    for (const auto& m : methods) {
        for (std::size_t n : path_counts) {
            double rmse = compute_rmse(n, m.vr, bs_price);
            std::printf("%s,%zu,%.8f\n", m.name, n, rmse);
            std::fflush(stdout);
        }
    }

    return 0;
}
