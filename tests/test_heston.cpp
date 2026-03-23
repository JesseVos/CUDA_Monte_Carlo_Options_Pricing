#include <cmath>

#include <gtest/gtest.h>

#include "engine/cpu_engine.h"
#include "engine/gpu_engine.h"
#include "models/heston.h"
#include "pricing/european.h"
#include "utils/heston_analytical.h"

namespace {

constexpr double SPOT = 100.0;
constexpr double STRIKE = 100.0;
constexpr double RATE = 0.05;
constexpr double MATURITY = 1.0;

constexpr double V0 = 0.04;
constexpr double KAPPA = 2.0;
constexpr double THETA = 0.04;
constexpr double XI = 0.3;
constexpr double RHO = -0.7;

constexpr std::size_t NUM_PATHS = 500'000;
constexpr std::size_t NUM_STEPS = 252;

}  // namespace

TEST(HestonAnalyticalTest, ReturnsPositiveCallPrice) {
    const double price = heston_call_price(
        SPOT, STRIKE, RATE, MATURITY, V0, KAPPA, THETA, XI, RHO);
    EXPECT_GT(price, 0.0);
}

TEST(HestonCPUTest, MCMatchesAnalyticalWithinConfidenceBand) {
    Heston<double> model(RATE, V0, KAPPA, THETA, XI, RHO);
    CPUEngine cpu_engine(NUM_PATHS, NUM_STEPS, 42UL);

    const PricingResult mc = cpu_engine.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);
    const double analytical = heston_call_price(
        SPOT, STRIKE, RATE, MATURITY, V0, KAPPA, THETA, XI, RHO);

    EXPECT_NEAR(mc.price, analytical, 3.0 * mc.standard_error);
}

TEST(HestonGPUTest, MCMatchesAnalyticalWithinConfidenceBand) {
    GPUEngine gpu_engine(NUM_PATHS, NUM_STEPS, 42ULL);

    const PricingResult mc = gpu_engine.price_european_heston(
        SPOT, STRIKE, RATE, MATURITY, V0, KAPPA, THETA, XI, RHO,
        OptionType::Call);
    const double analytical = heston_call_price(
        SPOT, STRIKE, RATE, MATURITY, V0, KAPPA, THETA, XI, RHO);

    EXPECT_NEAR(mc.price, analytical, 3.0 * mc.standard_error);
}

TEST(HestonGPUVsCPUTest, StatisticalConsistency) {
    Heston<double> model(RATE, V0, KAPPA, THETA, XI, RHO);
    CPUEngine cpu_engine(NUM_PATHS, NUM_STEPS, 777UL);
    GPUEngine gpu_engine(NUM_PATHS, NUM_STEPS, 777ULL);

    const PricingResult cpu = cpu_engine.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);
    const PricingResult gpu = gpu_engine.price_european_heston(
        SPOT, STRIKE, RATE, MATURITY, V0, KAPPA, THETA, XI, RHO,
        OptionType::Call);

    const double combined_se = std::sqrt(cpu.standard_error * cpu.standard_error
                                         + gpu.standard_error * gpu.standard_error);
    EXPECT_NEAR(cpu.price, gpu.price, 3.0 * combined_se);
}
