#include <cmath>

#include <gtest/gtest.h>

#include "engine/cpu_engine.h"
#include "engine/gpu_engine.h"
#include "models/gbm.h"
#include "models/heston.h"
#include "pricing/european.h"

namespace {

constexpr double SPOT     = 100.0;
constexpr double STRIKE   = 100.0;
constexpr double RATE     = 0.05;
constexpr double VOL      = 0.2;
constexpr double MATURITY = 1.0;

constexpr double V0    = 0.04;
constexpr double KAPPA = 2.0;
constexpr double THETA = 0.04;
constexpr double XI    = 0.3;
constexpr double RHO   = -0.7;

constexpr std::size_t NUM_PATHS = 500'000;
constexpr std::size_t NUM_STEPS = 252;

}  // namespace

// Asian call price must be strictly below the European call price for the same
// parameters.  For a non-dividend-paying asset, E[avg(S)] <= E[S_T] (by
// Jensen's inequality applied to the convex payoff), so this always holds.
TEST(AsianGBMCPUTest, CallBoundedByEuropeanCall) {
    GBM<double> model(RATE, VOL);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, 42UL);

    const PricingResult asian = engine.price_asian(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);
    const PricingResult european = engine.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);

    EXPECT_GT(european.price, asian.price);
}

// Asian call price must be positive.
TEST(AsianGBMCPUTest, CallPricePositive) {
    GBM<double> model(RATE, VOL);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, 123UL);

    const PricingResult result = engine.price_asian(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);

    EXPECT_GT(result.price, 0.0);
    EXPECT_GT(result.standard_error, 0.0);
}

// GPU Asian price must be statistically consistent with CPU price.
TEST(AsianGBMGPUTest, GPUMatchesCPU) {
    GBM<double> model(RATE, VOL);
    CPUEngine cpu_engine(NUM_PATHS, NUM_STEPS, 999UL);
    GPUEngine gpu_engine(NUM_PATHS, NUM_STEPS, 999ULL);

    const PricingResult cpu = cpu_engine.price_asian(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);
    const PricingResult gpu = gpu_engine.price_asian_gbm(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);

    const double combined_se = std::sqrt(
        cpu.standard_error * cpu.standard_error
        + gpu.standard_error * gpu.standard_error);
    EXPECT_NEAR(cpu.price, gpu.price, 3.0 * combined_se);
}

// GPU Asian call is bounded by GPU European call.
TEST(AsianGBMGPUTest, CallBoundedByEuropeanCall) {
    GPUEngine engine(NUM_PATHS, NUM_STEPS, 77ULL);

    const PricingResult asian = engine.price_asian_gbm(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    const PricingResult european = engine.price_european_gbm(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);

    EXPECT_GT(european.price, asian.price);
}

// Heston Asian call is positive (CPU).
TEST(AsianHestonCPUTest, CallPricePositive) {
    Heston<double> model(RATE, V0, KAPPA, THETA, XI, RHO);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, 42UL);

    const PricingResult result = engine.price_asian(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);

    EXPECT_GT(result.price, 0.0);
}

// Heston Asian call is bounded by Heston European call (CPU).
TEST(AsianHestonCPUTest, CallBoundedByEuropeanCall) {
    Heston<double> model(RATE, V0, KAPPA, THETA, XI, RHO);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, 55UL);

    const PricingResult asian = engine.price_asian(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);
    const PricingResult european = engine.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);

    EXPECT_GT(european.price, asian.price);
}

// Heston GPU Asian matches CPU Asian.
TEST(AsianHestonGPUTest, GPUMatchesCPU) {
    Heston<double> model(RATE, V0, KAPPA, THETA, XI, RHO);
    CPUEngine cpu_engine(NUM_PATHS, NUM_STEPS, 314UL);
    GPUEngine gpu_engine(NUM_PATHS, NUM_STEPS, 314ULL);

    const PricingResult cpu = cpu_engine.price_asian(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);
    const PricingResult gpu = gpu_engine.price_asian_heston(
        SPOT, STRIKE, RATE, MATURITY,
        V0, KAPPA, THETA, XI, RHO,
        OptionType::Call);

    const double combined_se = std::sqrt(
        cpu.standard_error * cpu.standard_error
        + gpu.standard_error * gpu.standard_error);
    EXPECT_NEAR(cpu.price, gpu.price, 3.0 * combined_se);
}
