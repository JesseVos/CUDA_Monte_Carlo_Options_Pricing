#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "engine/cpu_engine.h"
#include "engine/gpu_engine.h"
#include "models/gbm.h"
#include "pricing/european.h"
#include "utils/black_scholes.h"

/// Simple wall-clock timer.
class Timer {
public:
    void start() { t0_ = std::chrono::high_resolution_clock::now(); }
    void stop()  { t1_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t0_, t1_;
};

int main() {
    // Standard test parameters.
    constexpr double SPOT       = 100.0;
    constexpr double STRIKE     = 105.0;
    constexpr double RATE       = 0.05;
    constexpr double VOLATILITY = 0.2;
    constexpr double MATURITY   = 1.0;
    constexpr std::size_t NUM_STEPS = 252;

    BSResult bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    std::cout << "Black-Scholes reference call price: " << std::fixed
              << std::setprecision(6) << bs.price << "\n\n";

    std::cout << std::left
              << std::setw(14) << "Paths"
              << std::setw(14) << "CPU (ms)"
              << std::setw(14) << "GPU (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "CPU Price"
              << std::setw(14) << "GPU Price"
              << std::setw(12) << "CPU SE"
              << std::setw(12) << "GPU SE"
              << "\n";
    std::cout << std::string(102, '-') << "\n";

    std::size_t path_counts[] = {100'000, 1'000'000, 10'000'000};

    Timer timer;
    GBM<double> model(RATE, VOLATILITY);

    for (std::size_t num_paths : path_counts) {
        // --- CPU ---
        CPUEngine cpu_engine(num_paths, NUM_STEPS, /*seed=*/42);
        timer.start();
        PricingResult cpu_result = cpu_engine.price_european(
            model, SPOT, STRIKE, MATURITY, OptionType::Call);
        timer.stop();
        double cpu_ms = timer.elapsed_ms();

        // --- GPU (warm-up + timed run) ---
        // First call includes kernel compilation / context init overhead.
        {
            GPUEngine warmup(1000, 10, 0ULL);
            warmup.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY, MATURITY,
                                      OptionType::Call);
        }

        GPUEngine gpu_engine(num_paths, NUM_STEPS, /*seed=*/42ULL);
        timer.start();
        PricingResult gpu_result = gpu_engine.price_european_gbm(
            SPOT, STRIKE, RATE, VOLATILITY, MATURITY, OptionType::Call);
        timer.stop();
        double gpu_ms = timer.elapsed_ms();

        double speedup = cpu_ms / gpu_ms;

        // Build speedup string so "x" suffix doesn't break column alignment.
        std::ostringstream speedup_str;
        speedup_str << std::fixed << std::setprecision(1) << speedup << "x";

        std::cout << std::left
                  << std::setw(14) << num_paths
                  << std::setw(14) << std::fixed << std::setprecision(1) << cpu_ms
                  << std::setw(14) << gpu_ms
                  << std::setw(12) << speedup_str.str()
                  << std::setw(14) << std::setprecision(4) << cpu_result.price
                  << std::setw(14) << gpu_result.price
                  << std::setw(12) << std::setprecision(4) << cpu_result.standard_error
                  << std::setw(12) << gpu_result.standard_error
                  << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
