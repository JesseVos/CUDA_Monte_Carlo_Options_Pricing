#include "engine/gpu_engine.h"
#include "engine/device_buffer.h"
#include "pricing/european.h"
#include "pricing/barrier.h"

#include "../../kernels/gbm_kernel.h"
#include "../../kernels/gbm_antithetic_kernel.h"
#include "../../kernels/heston_kernel.h"
#include "../../kernels/asian_kernel.h"
#include "../../kernels/asian_heston_kernel.h"
#include "../../kernels/barrier_kernel.h"
#include "../../kernels/barrier_heston_kernel.h"
#include "../../kernels/reduction.h"
#include "../../kernels/reduction_cv.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

GPUEngine::GPUEngine(std::size_t num_paths, std::size_t num_steps, unsigned long long seed)
    : num_paths_(num_paths), num_steps_(num_steps), seed_(seed)
{
}

PricingResult GPUEngine::price_european_gbm(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type)
{
    // Allocate device memory for terminal asset values.
    DeviceBuffer<double> d_terminal(num_paths_);

    // Launch GBM path generation kernel.
    launch_gbm_kernel<double>(
        d_terminal.data(),
        spot, rate, volatility, maturity,
        num_paths_, num_steps_, seed_);

    // Launch payoff reduction kernel.
    bool is_call = (option_type == OptionType::Call);
    ReductionResult<double> reduction;
    launch_payoff_reduction<double>(
        d_terminal.data(),
        num_paths_,
        strike,
        is_call,
        reduction);

    // Compute price and standard error on host.
    double n = static_cast<double>(num_paths_);
    double discount = std::exp(-rate * maturity);
    double mean_payoff = reduction.sum / n;
    double price = discount * mean_payoff;

    double variance = (reduction.sum_sq / n) - (mean_payoff * mean_payoff);
    if (variance < 0.0) variance = 0.0;  // Clamp numerical noise.
    double standard_error = discount * std::sqrt(variance / n);

    return {price, standard_error};
}

PricingResult GPUEngine::price_european_gbm(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type,
    const VarianceReductionConfig& vr_config)
{
    if (vr_config.antithetic && (num_paths_ % 2 != 0)) {
        throw std::invalid_argument(
            "Antithetic variates require num_paths to be even.");
    }

    DeviceBuffer<double> d_terminal(num_paths_);

    // Path generation — standard or antithetic.
    if (vr_config.antithetic) {
        launch_gbm_antithetic_kernel<double>(
            d_terminal.data(),
            spot, rate, volatility, maturity,
            num_paths_, num_steps_, seed_);
    } else {
        launch_gbm_kernel<double>(
            d_terminal.data(),
            spot, rate, volatility, maturity,
            num_paths_, num_steps_, seed_);
    }

    bool is_call = (option_type == OptionType::Call);
    double discount = std::exp(-rate * maturity);

    // Reduction — standard or five-moment for control variate.
    if (vr_config.control_variate) {
        // For control variate (with or without antithetic), use the five-moment
        // reduction on all num_paths_ terminal values.  When antithetic is also
        // active the beta and price are still correct; the reported SE is a
        // slightly conservative overestimate but remains well below plain MC.
        double n = static_cast<double>(num_paths_);
        // Control variable: S_T with known expectation E[S_T] = S_0 * exp(r*T).
        double expected_s = spot * std::exp(rate * maturity);

        CVReductionResult<double> cv;
        launch_payoff_cv_reduction<double>(
            d_terminal.data(), num_paths_, strike, is_call, cv);

        double mean_h  = cv.sum_h    / n;
        double mean_s  = cv.sum_s    / n;
        double var_s   = (cv.sum_s_sq / n) - (mean_s * mean_s);
        double cov_hs  = (cv.sum_hs   / n) - (mean_h * mean_s);

        if (var_s <= 0.0) {
            // Degenerate: fall back to plain estimate.
            double var_h = std::max((cv.sum_h_sq / n) - (mean_h * mean_h), 0.0);
            return {discount * mean_h, discount * std::sqrt(var_h / n)};
        }

        double beta       = cov_hs / var_s;
        double mean_h_cv  = mean_h - beta * (mean_s - expected_s);
        double price      = discount * mean_h_cv;

        // Var(h_cv) = Var(h) - Cov(h,S)^2 / Var(S)
        double var_h      = (cv.sum_h_sq / n) - (mean_h * mean_h);
        double var_h_cv   = std::max(var_h - (cov_hs * cov_hs) / var_s, 0.0);
        double se         = discount * std::sqrt(var_h_cv / n);

        return {price, se};
    }

    if (vr_config.antithetic) {
        // Antithetic-only: reduce pair averages to get the correct SE.
        // Effective sample size is N/2 independent pair averages.
        ReductionResult<double> reduction;
        launch_antithetic_payoff_reduction<double>(
            d_terminal.data(), num_paths_, strike, is_call, reduction);

        double half     = static_cast<double>(num_paths_ / 2);
        double mean_avg = reduction.sum / half;
        double price    = discount * mean_avg;
        double variance = std::max((reduction.sum_sq / half) - (mean_avg * mean_avg), 0.0);
        double se       = discount * std::sqrt(variance / half);
        return {price, se};
    }

    // Plain MC (no variance reduction).
    double n = static_cast<double>(num_paths_);
    ReductionResult<double> reduction;
    launch_payoff_reduction<double>(
        d_terminal.data(), num_paths_, strike, is_call, reduction);

    double mean_payoff = reduction.sum / n;
    double price       = discount * mean_payoff;
    double variance    = std::max((reduction.sum_sq / n) - (mean_payoff * mean_payoff), 0.0);
    double se          = discount * std::sqrt(variance / n);

    return {price, se};
}

PricingResult GPUEngine::price_european_gbm_f(
    float spot,
    float strike,
    float rate,
    float volatility,
    float maturity,
    OptionType option_type)
{
    DeviceBuffer<float> d_terminal(num_paths_);

    launch_gbm_kernel<float>(
        d_terminal.data(),
        spot, rate, volatility, maturity,
        num_paths_, num_steps_, seed_);

    bool is_call = (option_type == OptionType::Call);
    ReductionResult<float> reduction;
    launch_payoff_reduction<float>(
        d_terminal.data(),
        num_paths_,
        strike,
        is_call,
        reduction);

    // Promote to double for price/SE computation to avoid FP32 precision loss.
    double n = static_cast<double>(num_paths_);
    double discount = std::exp(-static_cast<double>(rate) * static_cast<double>(maturity));
    double mean_payoff = static_cast<double>(reduction.sum) / n;
    double price = discount * mean_payoff;

    double variance = (static_cast<double>(reduction.sum_sq) / n) - (mean_payoff * mean_payoff);
    if (variance < 0.0) variance = 0.0;
    double standard_error = discount * std::sqrt(variance / n);

    return {price, standard_error};
}

PricingResult GPUEngine::price_european_heston(
    double spot,
    double strike,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    OptionType option_type)
{
    DeviceBuffer<double> d_terminal(num_paths_);

    launch_heston_kernel<double>(
        d_terminal.data(),
        spot,
        rate,
        maturity,
        v0,
        kappa,
        theta,
        xi,
        rho,
        num_paths_,
        num_steps_,
        seed_);

    const bool is_call = (option_type == OptionType::Call);
    ReductionResult<double> reduction;
    launch_payoff_reduction<double>(
        d_terminal.data(),
        num_paths_,
        strike,
        is_call,
        reduction);

    const double n = static_cast<double>(num_paths_);
    const double discount = std::exp(-rate * maturity);
    const double mean_payoff = reduction.sum / n;
    const double price = discount * mean_payoff;
    const double variance = std::max((reduction.sum_sq / n)
                                     - (mean_payoff * mean_payoff), 0.0);
    const double se = discount * std::sqrt(variance / n);

    return {price, se};
}

// ---------------------------------------------------------------------------
// Asian options
// ---------------------------------------------------------------------------

PricingResult GPUEngine::price_asian_gbm(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type)
{
    DeviceBuffer<double> d_averages(num_paths_);

    launch_asian_gbm_kernel<double>(
        d_averages.data(),
        spot, rate, volatility, maturity,
        num_paths_, num_steps_, seed_);

    const bool is_call = (option_type == OptionType::Call);
    ReductionResult<double> reduction;
    launch_payoff_reduction<double>(
        d_averages.data(), num_paths_, strike, is_call, reduction);

    const double n           = static_cast<double>(num_paths_);
    const double discount    = std::exp(-rate * maturity);
    const double mean_payoff = reduction.sum / n;
    const double price       = discount * mean_payoff;
    const double variance    = std::max(
        (reduction.sum_sq / n) - (mean_payoff * mean_payoff), 0.0);
    const double se          = discount * std::sqrt(variance / n);

    return {price, se};
}

PricingResult GPUEngine::price_asian_heston(
    double spot,
    double strike,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    OptionType option_type)
{
    DeviceBuffer<double> d_averages(num_paths_);

    launch_asian_heston_kernel(
        d_averages.data(),
        spot, rate, maturity,
        v0, kappa, theta, xi, rho,
        num_paths_, num_steps_, seed_);

    const bool is_call = (option_type == OptionType::Call);
    ReductionResult<double> reduction;
    launch_payoff_reduction<double>(
        d_averages.data(), num_paths_, strike, is_call, reduction);

    const double n           = static_cast<double>(num_paths_);
    const double discount    = std::exp(-rate * maturity);
    const double mean_payoff = reduction.sum / n;
    const double price       = discount * mean_payoff;
    const double variance    = std::max(
        (reduction.sum_sq / n) - (mean_payoff * mean_payoff), 0.0);
    const double se          = discount * std::sqrt(variance / n);

    return {price, se};
}

// ---------------------------------------------------------------------------
// Barrier options
// ---------------------------------------------------------------------------

PricingResult GPUEngine::price_barrier_gbm(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    double barrier,
    BarrierType barrier_type,
    OptionType option_type)
{
    const bool is_upper    = (barrier_type == BarrierType::UpAndOut
                              || barrier_type == BarrierType::UpAndIn);
    const bool is_knockout = (barrier_type == BarrierType::UpAndOut
                              || barrier_type == BarrierType::DownAndOut);

    DeviceBuffer<double> d_terminal(num_paths_);

    launch_barrier_gbm_kernel<double>(
        d_terminal.data(),
        spot, rate, volatility, maturity, barrier,
        is_upper, is_knockout,
        num_paths_, num_steps_, seed_);

    const bool is_call = (option_type == OptionType::Call);
    ReductionResult<double> reduction;
    launch_payoff_reduction<double>(
        d_terminal.data(), num_paths_, strike, is_call, reduction);

    const double n           = static_cast<double>(num_paths_);
    const double discount    = std::exp(-rate * maturity);
    const double mean_payoff = reduction.sum / n;
    const double price       = discount * mean_payoff;
    const double variance    = std::max(
        (reduction.sum_sq / n) - (mean_payoff * mean_payoff), 0.0);
    const double se          = discount * std::sqrt(variance / n);

    return {price, se};
}

PricingResult GPUEngine::price_barrier_heston(
    double spot,
    double strike,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double barrier,
    BarrierType barrier_type,
    OptionType option_type)
{
    const bool is_upper    = (barrier_type == BarrierType::UpAndOut
                              || barrier_type == BarrierType::UpAndIn);
    const bool is_knockout = (barrier_type == BarrierType::UpAndOut
                              || barrier_type == BarrierType::DownAndOut);

    DeviceBuffer<double> d_terminal(num_paths_);

    launch_barrier_heston_kernel(
        d_terminal.data(),
        spot, rate, maturity,
        v0, kappa, theta, xi, rho,
        barrier, is_upper, is_knockout,
        num_paths_, num_steps_, seed_);

    const bool is_call = (option_type == OptionType::Call);
    ReductionResult<double> reduction;
    launch_payoff_reduction<double>(
        d_terminal.data(), num_paths_, strike, is_call, reduction);

    const double n           = static_cast<double>(num_paths_);
    const double discount    = std::exp(-rate * maturity);
    const double mean_payoff = reduction.sum / n;
    const double price       = discount * mean_payoff;
    const double variance    = std::max(
        (reduction.sum_sq / n) - (mean_payoff * mean_payoff), 0.0);
    const double se          = discount * std::sqrt(variance / n);

    return {price, se};
}
