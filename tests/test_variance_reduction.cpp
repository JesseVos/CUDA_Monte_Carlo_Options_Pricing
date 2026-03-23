#include "engine/cpu_engine.h"
#include "engine/gpu_engine.h"
#include "models/gbm.h"
#include "utils/black_scholes.h"
#include "variance/variance_reduction.h"

#include <gtest/gtest.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Shared parameters
// ---------------------------------------------------------------------------
static constexpr double SPOT       = 100.0;
static constexpr double STRIKE     = 105.0;
static constexpr double RATE       = 0.05;
static constexpr double VOLATILITY = 0.20;
static constexpr double MATURITY   = 1.0;
static constexpr std::size_t PATHS = 1'000'000;
static constexpr std::size_t STEPS = 252;
static constexpr unsigned long long SEED = 42ULL;

// ---------------------------------------------------------------------------
// CPU antithetic variates
// ---------------------------------------------------------------------------

TEST(CPUAntithetic, CallConvergesToBlackScholes) {
    GBM<double> model(RATE, VOLATILITY);
    CPUEngine engine(PATHS, STEPS, static_cast<unsigned long>(SEED));

    VarianceReductionConfig vr;
    vr.antithetic = true;

    auto result = engine.price_european(model, SPOT, STRIKE, MATURITY,
                                        OptionType::Call, vr);

    auto bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double tol = 2.0 * result.standard_error;
    EXPECT_NEAR(result.price, bs.price, tol)
        << "Antithetic call too far from BS. SE=" << result.standard_error;
}

TEST(CPUAntithetic, PutConvergesToBlackScholes) {
    GBM<double> model(RATE, VOLATILITY);
    CPUEngine engine(PATHS, STEPS, static_cast<unsigned long>(SEED));

    VarianceReductionConfig vr;
    vr.antithetic = true;

    auto result = engine.price_european(model, SPOT, STRIKE, MATURITY,
                                        OptionType::Put, vr);

    auto bs = bs_put(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double tol = 2.0 * result.standard_error;
    EXPECT_NEAR(result.price, bs.price, tol)
        << "Antithetic put too far from BS. SE=" << result.standard_error;
}

TEST(CPUAntithetic, ReducesStandardError) {
    GBM<double> model(RATE, VOLATILITY);

    // Plain MC.
    CPUEngine plain_engine(PATHS, STEPS, static_cast<unsigned long>(SEED));
    auto plain = plain_engine.price_european(model, SPOT, STRIKE, MATURITY,
                                             OptionType::Call);

    // Antithetic with same N.
    CPUEngine av_engine(PATHS, STEPS, static_cast<unsigned long>(SEED));
    VarianceReductionConfig vr;
    vr.antithetic = true;
    auto av = av_engine.price_european(model, SPOT, STRIKE, MATURITY,
                                       OptionType::Call, vr);

    EXPECT_LT(av.standard_error, plain.standard_error)
        << "Antithetic SE=" << av.standard_error
        << " should be < plain SE=" << plain.standard_error;
}

// ---------------------------------------------------------------------------
// CPU control variates
// ---------------------------------------------------------------------------

TEST(CPUControlVariate, CallConvergesToBlackScholes) {
    GBM<double> model(RATE, VOLATILITY);
    CPUEngine engine(PATHS, STEPS, static_cast<unsigned long>(SEED));

    VarianceReductionConfig vr;
    vr.control_variate = true;

    auto result = engine.price_european(model, SPOT, STRIKE, MATURITY,
                                        OptionType::Call, vr);

    auto bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double tol = 2.0 * result.standard_error;
    EXPECT_NEAR(result.price, bs.price, tol)
        << "CV call too far from BS. SE=" << result.standard_error;
}

TEST(CPUControlVariate, ReducesStandardError) {
    GBM<double> model(RATE, VOLATILITY);

    CPUEngine plain_engine(PATHS, STEPS, static_cast<unsigned long>(SEED));
    auto plain = plain_engine.price_european(model, SPOT, STRIKE, MATURITY,
                                             OptionType::Call);

    CPUEngine cv_engine(PATHS, STEPS, static_cast<unsigned long>(SEED));
    VarianceReductionConfig vr;
    vr.control_variate = true;
    auto cv = cv_engine.price_european(model, SPOT, STRIKE, MATURITY,
                                       OptionType::Call, vr);

    EXPECT_LT(cv.standard_error, plain.standard_error)
        << "CV SE=" << cv.standard_error
        << " should be < plain SE=" << plain.standard_error;
}

// ---------------------------------------------------------------------------
// CPU antithetic + control variates combined
// ---------------------------------------------------------------------------

TEST(CPUCombined, CallConvergesToBlackScholes) {
    GBM<double> model(RATE, VOLATILITY);
    CPUEngine engine(PATHS, STEPS, static_cast<unsigned long>(SEED));

    VarianceReductionConfig vr;
    vr.antithetic      = true;
    vr.control_variate = true;

    auto result = engine.price_european(model, SPOT, STRIKE, MATURITY,
                                        OptionType::Call, vr);

    auto bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double tol = 2.0 * result.standard_error;
    EXPECT_NEAR(result.price, bs.price, tol)
        << "Combined VR call too far from BS. SE=" << result.standard_error;
}

TEST(CPUCombined, ReducesStandardErrorVsAntithetic) {
    GBM<double> model(RATE, VOLATILITY);

    VarianceReductionConfig av_only;
    av_only.antithetic = true;
    CPUEngine av_engine(PATHS, STEPS, static_cast<unsigned long>(SEED));
    auto av = av_engine.price_european(model, SPOT, STRIKE, MATURITY,
                                       OptionType::Call, av_only);

    VarianceReductionConfig both;
    both.antithetic      = true;
    both.control_variate = true;
    CPUEngine both_engine(PATHS, STEPS, static_cast<unsigned long>(SEED));
    auto combined = both_engine.price_european(model, SPOT, STRIKE, MATURITY,
                                               OptionType::Call, both);

    EXPECT_LT(combined.standard_error, av.standard_error)
        << "Combined SE=" << combined.standard_error
        << " should be < antithetic-only SE=" << av.standard_error;
}

// ---------------------------------------------------------------------------
// GPU antithetic variates
// ---------------------------------------------------------------------------

TEST(GPUAntithetic, CallConvergesToBlackScholes) {
    GPUEngine engine(PATHS, STEPS, SEED);

    VarianceReductionConfig vr;
    vr.antithetic = true;

    auto result = engine.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY,
                                            MATURITY, OptionType::Call, vr);

    auto bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double tol = 2.0 * result.standard_error;
    EXPECT_NEAR(result.price, bs.price, tol)
        << "GPU antithetic call too far from BS. SE=" << result.standard_error;
}

TEST(GPUAntithetic, ReducesStandardError) {
    GPUEngine plain_engine(PATHS, STEPS, SEED);
    auto plain = plain_engine.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY,
                                                 MATURITY, OptionType::Call);

    GPUEngine av_engine(PATHS, STEPS, SEED);
    VarianceReductionConfig vr;
    vr.antithetic = true;
    auto av = av_engine.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY,
                                           MATURITY, OptionType::Call, vr);

    EXPECT_LT(av.standard_error, plain.standard_error)
        << "GPU antithetic SE=" << av.standard_error
        << " should be < plain SE=" << plain.standard_error;
}

// ---------------------------------------------------------------------------
// GPU control variates
// ---------------------------------------------------------------------------

TEST(GPUControlVariate, CallConvergesToBlackScholes) {
    GPUEngine engine(PATHS, STEPS, SEED);

    VarianceReductionConfig vr;
    vr.control_variate = true;

    auto result = engine.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY,
                                            MATURITY, OptionType::Call, vr);

    auto bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double tol = 2.0 * result.standard_error;
    EXPECT_NEAR(result.price, bs.price, tol)
        << "GPU CV call too far from BS. SE=" << result.standard_error;
}

TEST(GPUControlVariate, ReducesStandardError) {
    GPUEngine plain_engine(PATHS, STEPS, SEED);
    auto plain = plain_engine.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY,
                                                 MATURITY, OptionType::Call);

    GPUEngine cv_engine(PATHS, STEPS, SEED);
    VarianceReductionConfig vr;
    vr.control_variate = true;
    auto cv = cv_engine.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY,
                                           MATURITY, OptionType::Call, vr);

    EXPECT_LT(cv.standard_error, plain.standard_error)
        << "GPU CV SE=" << cv.standard_error
        << " should be < plain SE=" << plain.standard_error;
}

// ---------------------------------------------------------------------------
// GPU antithetic + control variates combined
// ---------------------------------------------------------------------------

TEST(GPUCombined, CallConvergesToBlackScholes) {
    GPUEngine engine(PATHS, STEPS, SEED);

    VarianceReductionConfig vr;
    vr.antithetic      = true;
    vr.control_variate = true;

    auto result = engine.price_european_gbm(SPOT, STRIKE, RATE, VOLATILITY,
                                            MATURITY, OptionType::Call, vr);

    auto bs = bs_call(SPOT, STRIKE, RATE, VOLATILITY, MATURITY);
    double tol = 2.0 * result.standard_error;
    EXPECT_NEAR(result.price, bs.price, tol)
        << "GPU combined VR call too far from BS. SE=" << result.standard_error;
}
