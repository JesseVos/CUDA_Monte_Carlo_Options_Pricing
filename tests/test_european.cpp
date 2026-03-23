#include <cmath>
#include <gtest/gtest.h>

#include "engine/cpu_engine.h"
#include "models/gbm.h"
#include "pricing/european.h"
#include "utils/black_scholes.h"

// ---------------------------------------------------------------------------
// Standard test parameters from AGENTS.md:
//   S=100, K=105, r=0.05, sigma=0.2, T=1.0
// ---------------------------------------------------------------------------
static constexpr double SPOT       = 100.0;
static constexpr double STRIKE     = 105.0;
static constexpr double RATE       = 0.05;
static constexpr double VOLATILITY = 0.2;
static constexpr double MATURITY   = 1.0;

static constexpr std::size_t NUM_PATHS = 1'000'000;
static constexpr std::size_t NUM_STEPS = 252;  // daily steps for 1 year

// ---------------------------------------------------------------------------
// Test 1: MC European call converges to Black-Scholes within 2 standard errors
// ---------------------------------------------------------------------------
TEST(EuropeanTest, CallConvergesToBlackScholes) {
    // Closed-form reference price.
    BSResult bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double bs_price = bs.price;

    // Monte Carlo price via CPUEngine.
    GBM<double> model(RATE, VOLATILITY);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, /*seed=*/42);
    PricingResult mc = engine.price_european(model, SPOT, STRIKE, MATURITY,
                                             OptionType::Call);

    // The MC price should be within 2 standard errors of the BS price.
    double tolerance = 2.0 * mc.standard_error;

    EXPECT_NEAR(mc.price, bs_price, tolerance)
        << "MC call price " << mc.price
        << " not within 2 SE (" << mc.standard_error << ") of BS price "
        << bs_price;

    // Sanity: standard error should be small relative to price (< 1%).
    EXPECT_LT(mc.standard_error, 0.01 * bs_price)
        << "Standard error too large: " << mc.standard_error;
}

// ---------------------------------------------------------------------------
// Test 2: MC European put converges to Black-Scholes within 2 standard errors
// ---------------------------------------------------------------------------
TEST(EuropeanTest, PutConvergesToBlackScholes) {
    BSResult bs = bs_put(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double bs_price = bs.price;

    GBM<double> model(RATE, VOLATILITY);
    CPUEngine engine(NUM_PATHS, NUM_STEPS, /*seed=*/123);
    PricingResult mc = engine.price_european(model, SPOT, STRIKE, MATURITY,
                                             OptionType::Put);

    double tolerance = 2.0 * mc.standard_error;

    EXPECT_NEAR(mc.price, bs_price, tolerance)
        << "MC put price " << mc.price
        << " not within 2 SE (" << mc.standard_error << ") of BS price "
        << bs_price;

    EXPECT_LT(mc.standard_error, 0.01 * bs_price)
        << "Standard error too large: " << mc.standard_error;
}

// ---------------------------------------------------------------------------
// Test 3: Put-call parity
//   C - P = S - K*exp(-r*T)   (no dividends, so q=0)
//
// We run call and put with the SAME seed so that the random paths are
// identical, which makes the parity check tighter. But even with different
// seeds at 1M paths, this should hold within statistical tolerance.
// ---------------------------------------------------------------------------
TEST(EuropeanTest, PutCallParity) {
    // Analytical parity value.
    double parity_exact = SPOT - STRIKE * std::exp(-RATE * MATURITY);

    // MC prices — use same seed for common random numbers.
    GBM<double> model(RATE, VOLATILITY);
    CPUEngine engine_call(NUM_PATHS, NUM_STEPS, /*seed=*/999);
    CPUEngine engine_put(NUM_PATHS, NUM_STEPS, /*seed=*/999);

    PricingResult mc_call = engine_call.price_european(model, SPOT, STRIKE,
                                                       MATURITY, OptionType::Call);
    PricingResult mc_put = engine_put.price_european(model, SPOT, STRIKE,
                                                      MATURITY, OptionType::Put);

    double mc_parity = mc_call.price - mc_put.price;

    // With common random numbers the parity should hold very tightly.
    // Use 3 standard errors of the call as a generous tolerance.
    double tolerance = 3.0 * mc_call.standard_error;

    EXPECT_NEAR(mc_parity, parity_exact, tolerance)
        << "Put-call parity violated: C - P = " << mc_parity
        << ", expected " << parity_exact
        << " (call=" << mc_call.price << ", put=" << mc_put.price << ")";
}

// ---------------------------------------------------------------------------
// Test 4: Black-Scholes closed-form sanity checks
// ---------------------------------------------------------------------------
TEST(BlackScholesTest, KnownValues) {
    // ATM call with known approximate value.
    // S=100, K=100, r=0.05, sigma=0.2, T=1.0
    // BS call ~ 10.4506
    BSResult bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    EXPECT_NEAR(bs.price, 10.4506, 0.001);

    // Put via put-call parity: P = C - S + K*exp(-rT)
    BSResult bs_p = bs_put(100.0, 100.0, 0.05, 0.2, 1.0);
    double parity_put = bs.price - 100.0 + 100.0 * std::exp(-0.05);
    EXPECT_NEAR(bs_p.price, parity_put, 1e-10)
        << "BS put-call parity inconsistency";
}

TEST(BlackScholesTest, DeepITMCall) {
    // Deep in-the-money call: S=200, K=100 should be close to
    // S - K*exp(-rT) = 200 - 100*exp(-0.05) ~ 104.878
    BSResult bs = bs_call(200.0, 100.0, 0.05, 0.2, 1.0);
    EXPECT_GT(bs.price, 100.0);
    EXPECT_LT(bs.price, 110.0);
}

TEST(BlackScholesTest, DeepOTMCall) {
    // Deep out-of-the-money call: S=50, K=100, should be very small.
    BSResult bs = bs_call(50.0, 100.0, 0.05, 0.2, 1.0);
    EXPECT_LT(bs.price, 0.01);
}
