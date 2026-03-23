#include <cmath>
#include <gtest/gtest.h>

#include "engine/cpu_engine.h"
#include "engine/gpu_engine.h"
#include "models/gbm.h"
#include "pricing/european.h"
#include "utils/black_scholes.h"

// ---------------------------------------------------------------------------
// Standard test parameters (AGENTS.md):
//   S=100, K=105, r=0.05, sigma=0.2, T=1.0
// ---------------------------------------------------------------------------
static constexpr double SPOT       = 100.0;
static constexpr double STRIKE     = 105.0;
static constexpr double RATE       = 0.05;
static constexpr double VOLATILITY = 0.2;
static constexpr double MATURITY   = 1.0;

static constexpr std::size_t NUM_PATHS = 1'000'000;
static constexpr std::size_t NUM_STEPS = 252;

// ---------------------------------------------------------------------------
// Test 1: GPU European call converges to Black-Scholes within 2 SE
// ---------------------------------------------------------------------------
TEST(GPUEuropeanTest, CallConvergesToBlackScholes) {
    BSResult bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);

    GPUEngine engine(NUM_PATHS, NUM_STEPS, /*seed=*/42ULL);
    PricingResult mc = engine.price_european_gbm(
        SPOT, STRIKE, RATE, VOLATILITY, MATURITY, OptionType::Call);

    double tolerance = 2.0 * mc.standard_error;

    EXPECT_NEAR(mc.price, bs.price, tolerance)
        << "GPU MC call price " << mc.price
        << " not within 2 SE (" << mc.standard_error << ") of BS price "
        << bs.price;

    EXPECT_LT(mc.standard_error, 0.01 * bs.price)
        << "Standard error too large: " << mc.standard_error;
}

// ---------------------------------------------------------------------------
// Test 2: GPU European put converges to Black-Scholes within 2 SE
// ---------------------------------------------------------------------------
TEST(GPUEuropeanTest, PutConvergesToBlackScholes) {
    BSResult bs = bs_put(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);

    GPUEngine engine(NUM_PATHS, NUM_STEPS, /*seed=*/123ULL);
    PricingResult mc = engine.price_european_gbm(
        SPOT, STRIKE, RATE, VOLATILITY, MATURITY, OptionType::Put);

    // 3 SE tolerance per AGENTS.md: "Use generous tolerance (2-3 standard errors)"
    double tolerance = 3.0 * mc.standard_error;

    EXPECT_NEAR(mc.price, bs.price, tolerance)
        << "GPU MC put price " << mc.price
        << " not within 3 SE (" << mc.standard_error << ") of BS price "
        << bs.price;

    EXPECT_LT(mc.standard_error, 0.01 * bs.price)
        << "Standard error too large: " << mc.standard_error;
}

// ---------------------------------------------------------------------------
// Test 3: GPU put-call parity
// ---------------------------------------------------------------------------
TEST(GPUEuropeanTest, PutCallParity) {
    double parity_exact = SPOT - STRIKE * std::exp(-RATE * MATURITY);

    GPUEngine engine_call(NUM_PATHS, NUM_STEPS, /*seed=*/999ULL);
    GPUEngine engine_put(NUM_PATHS, NUM_STEPS, /*seed=*/999ULL);

    PricingResult mc_call = engine_call.price_european_gbm(
        SPOT, STRIKE, RATE, VOLATILITY, MATURITY, OptionType::Call);
    PricingResult mc_put = engine_put.price_european_gbm(
        SPOT, STRIKE, RATE, VOLATILITY, MATURITY, OptionType::Put);

    double mc_parity = mc_call.price - mc_put.price;
    double tolerance = 3.0 * mc_call.standard_error;

    EXPECT_NEAR(mc_parity, parity_exact, tolerance)
        << "GPU put-call parity violated: C - P = " << mc_parity
        << ", expected " << parity_exact;
}

// ---------------------------------------------------------------------------
// Test 4: GPU and CPU produce statistically consistent results
// ---------------------------------------------------------------------------
TEST(GPUEuropeanTest, GPUMatchesCPU) {
    // CPU price
    GBM<double> model(RATE, VOLATILITY);
    CPUEngine cpu_engine(NUM_PATHS, NUM_STEPS, /*seed=*/77);
    PricingResult cpu_result = cpu_engine.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);

    // GPU price (different seed is fine — we just check statistical consistency)
    GPUEngine gpu_engine(NUM_PATHS, NUM_STEPS, /*seed=*/77ULL);
    PricingResult gpu_result = gpu_engine.price_european_gbm(
        SPOT, STRIKE, RATE, VOLATILITY, MATURITY, OptionType::Call);

    // Both should be within 3 SE of the BS price, and therefore close
    // to each other. Use combined SE as tolerance.
    double combined_se = std::sqrt(cpu_result.standard_error * cpu_result.standard_error
                                   + gpu_result.standard_error * gpu_result.standard_error);
    double tolerance = 3.0 * combined_se;

    EXPECT_NEAR(cpu_result.price, gpu_result.price, tolerance)
        << "CPU price " << cpu_result.price << " vs GPU price " << gpu_result.price
        << " differ by more than 3 combined SE (" << combined_se << ")";
}

// ---------------------------------------------------------------------------
// Test 5: Single-precision GPU call converges to BS (looser tolerance)
// ---------------------------------------------------------------------------
TEST(GPUEuropeanTest, FloatCallConvergesToBlackScholes) {
    BSResult bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);

    GPUEngine engine(NUM_PATHS, NUM_STEPS, /*seed=*/42ULL);
    PricingResult mc = engine.price_european_gbm_f(
        static_cast<float>(SPOT),
        static_cast<float>(STRIKE),
        static_cast<float>(RATE),
        static_cast<float>(VOLATILITY),
        static_cast<float>(MATURITY),
        OptionType::Call);

    // Float has less precision; use 3 SE tolerance.
    double tolerance = 3.0 * mc.standard_error;

    EXPECT_NEAR(mc.price, bs.price, tolerance)
        << "GPU float MC call price " << mc.price
        << " not within 3 SE (" << mc.standard_error << ") of BS price "
        << bs.price;
}
