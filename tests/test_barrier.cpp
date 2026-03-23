#include <cmath>

#include <gtest/gtest.h>

#include "engine/cpu_engine.h"
#include "engine/gpu_engine.h"
#include "models/gbm.h"
#include "models/heston.h"
#include "pricing/european.h"
#include "pricing/barrier.h"

namespace {

constexpr double SPOT     = 100.0;
constexpr double STRIKE   = 100.0;
constexpr double RATE     = 0.05;
constexpr double VOL      = 0.2;
constexpr double MATURITY = 1.0;

// Up-barrier well above spot: knock-out price should approach European.
constexpr double HIGH_BARRIER = 200.0;
// Down-barrier well below spot: knock-out price should approach European.
constexpr double LOW_BARRIER  = 10.0;
// Standard barriers used for parity tests.
constexpr double UP_BARRIER   = 120.0;
constexpr double DOWN_BARRIER = 85.0;

constexpr double V0    = 0.04;
constexpr double KAPPA = 2.0;
constexpr double THETA = 0.04;
constexpr double XI    = 0.3;
constexpr double RHO   = -0.7;

constexpr std::size_t NUM_PATHS = 500'000;
constexpr std::size_t NUM_STEPS = 252;

}  // namespace

// As the up-barrier → ∞ the UpAndOut call converges to the European call.
TEST(BarrierGBMCPUTest, UpOutHighBarrierApproachesEuropean) {
    GBM<double> model(RATE, VOL);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, 42UL);

    const PricingResult european = engine.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);
    const PricingResult up_out = engine.price_barrier(
        model, SPOT, STRIKE, MATURITY, HIGH_BARRIER,
        BarrierType::UpAndOut, OptionType::Call);

    // The two prices should be very close; use a generous tolerance.
    EXPECT_NEAR(up_out.price, european.price, 0.05 * european.price);
}

// As the down-barrier → 0 the DownAndOut call converges to the European call.
TEST(BarrierGBMCPUTest, DownOutLowBarrierApproachesEuropean) {
    GBM<double> model(RATE, VOL);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, 43UL);

    const PricingResult european = engine.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);
    const PricingResult down_out = engine.price_barrier(
        model, SPOT, STRIKE, MATURITY, LOW_BARRIER,
        BarrierType::DownAndOut, OptionType::Call);

    EXPECT_NEAR(down_out.price, european.price, 0.05 * european.price);
}

// Parity: UpAndIn + UpAndOut = European call (same seed for identical paths).
TEST(BarrierGBMCPUTest, UpInPlusUpOutEqualsEuropean) {
    GBM<double> model(RATE, VOL);
    CPUEngine engine_in(NUM_PATHS,  NUM_STEPS, 100UL);
    CPUEngine engine_out(NUM_PATHS, NUM_STEPS, 100UL);
    CPUEngine engine_eu(NUM_PATHS,  NUM_STEPS, 100UL);

    const PricingResult up_in = engine_in.price_barrier(
        model, SPOT, STRIKE, MATURITY, UP_BARRIER,
        BarrierType::UpAndIn, OptionType::Call);
    const PricingResult up_out = engine_out.price_barrier(
        model, SPOT, STRIKE, MATURITY, UP_BARRIER,
        BarrierType::UpAndOut, OptionType::Call);
    const PricingResult european = engine_eu.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);

    const double parity_sum = up_in.price + up_out.price;
    // Allow 3 SE of either side.
    const double tol = 3.0 * european.standard_error;
    EXPECT_NEAR(parity_sum, european.price, tol);
}

// Parity: DownAndIn + DownAndOut = European call.
TEST(BarrierGBMCPUTest, DownInPlusDownOutEqualsEuropean) {
    GBM<double> model(RATE, VOL);
    CPUEngine engine_in(NUM_PATHS,  NUM_STEPS, 200UL);
    CPUEngine engine_out(NUM_PATHS, NUM_STEPS, 200UL);
    CPUEngine engine_eu(NUM_PATHS,  NUM_STEPS, 200UL);

    const PricingResult down_in = engine_in.price_barrier(
        model, SPOT, STRIKE, MATURITY, DOWN_BARRIER,
        BarrierType::DownAndIn, OptionType::Call);
    const PricingResult down_out = engine_out.price_barrier(
        model, SPOT, STRIKE, MATURITY, DOWN_BARRIER,
        BarrierType::DownAndOut, OptionType::Call);
    const PricingResult european = engine_eu.price_european(
        model, SPOT, STRIKE, MATURITY, OptionType::Call);

    const double parity_sum = down_in.price + down_out.price;
    const double tol = 3.0 * european.standard_error;
    EXPECT_NEAR(parity_sum, european.price, tol);
}

// Barrier price must be non-negative.
TEST(BarrierGBMCPUTest, PriceNonNegative) {
    GBM<double> model(RATE, VOL);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, 77UL);

    for (BarrierType bt : {BarrierType::UpAndOut, BarrierType::UpAndIn,
                           BarrierType::DownAndOut, BarrierType::DownAndIn}) {
        const double bar = (bt == BarrierType::UpAndOut || bt == BarrierType::UpAndIn)
                               ? UP_BARRIER : DOWN_BARRIER;
        const PricingResult r = engine.price_barrier(
            model, SPOT, STRIKE, MATURITY, bar, bt, OptionType::Call);
        EXPECT_GE(r.price, 0.0) << "Negative price for barrier type "
                                 << static_cast<int>(bt);
    }
}

// GPU UpAndOut matches CPU UpAndOut.
TEST(BarrierGBMGPUTest, GPUMatchesCPU) {
    GBM<double> model(RATE, VOL);
    CPUEngine cpu_engine(NUM_PATHS, NUM_STEPS, 888UL);
    GPUEngine gpu_engine(NUM_PATHS, NUM_STEPS, 888ULL);

    const PricingResult cpu = cpu_engine.price_barrier(
        model, SPOT, STRIKE, MATURITY, UP_BARRIER,
        BarrierType::UpAndOut, OptionType::Call);
    const PricingResult gpu = gpu_engine.price_barrier_gbm(
        SPOT, STRIKE, RATE, VOL, MATURITY, UP_BARRIER,
        BarrierType::UpAndOut, OptionType::Call);

    const double combined_se = std::sqrt(
        cpu.standard_error * cpu.standard_error
        + gpu.standard_error * gpu.standard_error);
    EXPECT_NEAR(cpu.price, gpu.price, 3.0 * combined_se);
}

// GPU parity: UpAndIn + UpAndOut = European (GPU only).
TEST(BarrierGBMGPUTest, UpInPlusUpOutEqualsEuropean) {
    GPUEngine engine_in(NUM_PATHS,  NUM_STEPS, 500ULL);
    GPUEngine engine_out(NUM_PATHS, NUM_STEPS, 500ULL);
    GPUEngine engine_eu(NUM_PATHS,  NUM_STEPS, 500ULL);

    const PricingResult up_in = engine_in.price_barrier_gbm(
        SPOT, STRIKE, RATE, VOL, MATURITY, UP_BARRIER,
        BarrierType::UpAndIn, OptionType::Call);
    const PricingResult up_out = engine_out.price_barrier_gbm(
        SPOT, STRIKE, RATE, VOL, MATURITY, UP_BARRIER,
        BarrierType::UpAndOut, OptionType::Call);
    const PricingResult european = engine_eu.price_european_gbm(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);

    const double parity_sum = up_in.price + up_out.price;
    const double tol = 3.0 * european.standard_error;
    EXPECT_NEAR(parity_sum, european.price, tol);
}

// Heston barrier call is positive (CPU).
TEST(BarrierHestonCPUTest, CallPricePositive) {
    Heston<double> model(RATE, V0, KAPPA, THETA, XI, RHO);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, 42UL);

    const PricingResult result = engine.price_barrier(
        model, SPOT, STRIKE, MATURITY, UP_BARRIER,
        BarrierType::UpAndOut, OptionType::Call);

    EXPECT_GE(result.price, 0.0);
}

// Heston GPU barrier matches CPU barrier.
TEST(BarrierHestonGPUTest, GPUMatchesCPU) {
    Heston<double> model(RATE, V0, KAPPA, THETA, XI, RHO);
    CPUEngine cpu_engine(NUM_PATHS, NUM_STEPS, 271UL);
    GPUEngine gpu_engine(NUM_PATHS, NUM_STEPS, 271ULL);

    const PricingResult cpu = cpu_engine.price_barrier(
        model, SPOT, STRIKE, MATURITY, UP_BARRIER,
        BarrierType::UpAndOut, OptionType::Call);
    const PricingResult gpu = gpu_engine.price_barrier_heston(
        SPOT, STRIKE, RATE, MATURITY,
        V0, KAPPA, THETA, XI, RHO,
        UP_BARRIER, BarrierType::UpAndOut, OptionType::Call);

    const double combined_se = std::sqrt(
        cpu.standard_error * cpu.standard_error
        + gpu.standard_error * gpu.standard_error);
    EXPECT_NEAR(cpu.price, gpu.price, 3.0 * combined_se);
}
