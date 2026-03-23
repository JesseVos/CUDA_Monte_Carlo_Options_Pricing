#include <cmath>

#include <gtest/gtest.h>

#include "greeks/greeks.h"
#include "pricing/european.h"

namespace {

constexpr double SPOT     = 100.0;
constexpr double STRIKE   = 105.0;
constexpr double RATE     = 0.05;
constexpr double VOL      = 0.2;
constexpr double MATURITY = 1.0;

// 2M paths: enough that CRN reduces FD noise to well below the tolerances.
constexpr std::size_t NUM_PATHS = 2'000'000;
constexpr std::size_t NUM_STEPS = 252;

// Tolerances: chosen based on expected Greek magnitudes and CRN noise levels.
// Delta ~0.44, Gamma ~0.018, Vega ~38, Theta ~-6/yr, Rho ~33.
constexpr double TOL_DELTA = 0.01;
constexpr double TOL_GAMMA = 0.003;
constexpr double TOL_VEGA  = 1.0;
constexpr double TOL_THETA = 0.5;
constexpr double TOL_RHO   = 1.5;

}  // namespace

// ---------------------------------------------------------------------------
// Black-Scholes analytical Greeks — sanity checks
// ---------------------------------------------------------------------------

TEST(BSGreeksTest, CallDeltaInUnitInterval) {
    const Greeks g = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_GT(g.delta, 0.0);
    EXPECT_LT(g.delta, 1.0);
}

TEST(BSGreeksTest, PutDeltaNegative) {
    const Greeks g = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Put);
    EXPECT_LT(g.delta, 0.0);
    EXPECT_GT(g.delta, -1.0);
}

TEST(BSGreeksTest, GammaPositive) {
    const Greeks g = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_GT(g.gamma, 0.0);
}

TEST(BSGreeksTest, VegaPositive) {
    const Greeks g = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_GT(g.vega, 0.0);
}

TEST(BSGreeksTest, ThetaNegativeForLongCall) {
    const Greeks g = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_LT(g.theta, 0.0);
}

TEST(BSGreeksTest, CallPutGammaSame) {
    const Greeks gc = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    const Greeks gp = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Put);
    // Put-call parity implies identical gamma.
    EXPECT_NEAR(gc.gamma, gp.gamma, 1e-10);
}

TEST(BSGreeksTest, CallPutVegaSame) {
    const Greeks gc = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    const Greeks gp = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Put);
    EXPECT_NEAR(gc.vega, gp.vega, 1e-10);
}

// ---------------------------------------------------------------------------
// CPU finite-difference Greeks vs Black-Scholes analytical
// ---------------------------------------------------------------------------

TEST(CPUGreeksTest, DeltaMatchesBS) {
    const Greeks fd = compute_greeks_gbm_cpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42UL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.delta, bs.delta, TOL_DELTA);
}

TEST(CPUGreeksTest, GammaMatchesBS) {
    const Greeks fd = compute_greeks_gbm_cpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42UL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.gamma, bs.gamma, TOL_GAMMA);
}

TEST(CPUGreeksTest, VegaMatchesBS) {
    const Greeks fd = compute_greeks_gbm_cpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42UL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.vega, bs.vega, TOL_VEGA);
}

TEST(CPUGreeksTest, ThetaMatchesBS) {
    const Greeks fd = compute_greeks_gbm_cpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42UL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.theta, bs.theta, TOL_THETA);
}

TEST(CPUGreeksTest, RhoMatchesBS) {
    const Greeks fd = compute_greeks_gbm_cpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42UL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.rho, bs.rho, TOL_RHO);
}

// ---------------------------------------------------------------------------
// GPU finite-difference Greeks vs Black-Scholes analytical
// ---------------------------------------------------------------------------

TEST(GPUGreeksTest, DeltaMatchesBS) {
    const Greeks fd = compute_greeks_gbm_gpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42ULL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.delta, bs.delta, TOL_DELTA);
}

TEST(GPUGreeksTest, GammaMatchesBS) {
    const Greeks fd = compute_greeks_gbm_gpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42ULL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.gamma, bs.gamma, TOL_GAMMA);
}

TEST(GPUGreeksTest, VegaMatchesBS) {
    const Greeks fd = compute_greeks_gbm_gpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42ULL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.vega, bs.vega, TOL_VEGA);
}

TEST(GPUGreeksTest, ThetaMatchesBS) {
    const Greeks fd = compute_greeks_gbm_gpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42ULL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.theta, bs.theta, TOL_THETA);
}

TEST(GPUGreeksTest, RhoMatchesBS) {
    const Greeks fd = compute_greeks_gbm_gpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 42ULL);
    const Greeks bs = black_scholes_greeks(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call);
    EXPECT_NEAR(fd.rho, bs.rho, TOL_RHO);
}

// GPU Greeks are statistically consistent with CPU Greeks.
TEST(GPUGreeksTest, GPUMatchesCPU) {
    const Greeks cpu = compute_greeks_gbm_cpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 123UL);
    const Greeks gpu = compute_greeks_gbm_gpu(
        SPOT, STRIKE, RATE, VOL, MATURITY, OptionType::Call,
        NUM_PATHS, NUM_STEPS, 123ULL);

    EXPECT_NEAR(cpu.delta, gpu.delta, TOL_DELTA);
    EXPECT_NEAR(cpu.gamma, gpu.gamma, TOL_GAMMA);
    EXPECT_NEAR(cpu.vega,  gpu.vega,  TOL_VEGA);
    EXPECT_NEAR(cpu.theta, gpu.theta, TOL_THETA);
    EXPECT_NEAR(cpu.rho,   gpu.rho,   TOL_RHO);
}
