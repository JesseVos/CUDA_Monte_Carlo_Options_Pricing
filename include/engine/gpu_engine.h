#pragma once

#include "pricing/european.h"
#include "pricing/barrier.h"
#include "variance/variance_reduction.h"

#include <cstddef>

/// GPU-based Monte Carlo simulation engine.
///
/// Launches CUDA kernels for path generation and payoff reduction.
/// Same public API as CPUEngine so they can be used interchangeably.
///
/// Note: The GPU engine does NOT take a Model& — it calls the CUDA kernel
/// directly. Model parameters (rate, volatility) are passed explicitly.
/// This avoids coupling the Model hierarchy to CUDA headers.
class GPUEngine {
public:
    /// @param num_paths  Number of Monte Carlo paths to simulate.
    /// @param num_steps  Number of time steps per path.
    /// @param seed       RNG seed for reproducibility.
    GPUEngine(std::size_t num_paths, std::size_t num_steps, unsigned long long seed);

    /// Price a European option under GBM on the GPU.
    ///
    /// @param spot        Initial asset price.
    /// @param strike      Strike price.
    /// @param rate        Risk-free rate.
    /// @param volatility  Constant volatility sigma.
    /// @param maturity    Time to expiration (years).
    /// @param option_type Call or Put.
    /// @return PricingResult with price and standard error.
    PricingResult price_european_gbm(
        double spot,
        double strike,
        double rate,
        double volatility,
        double maturity,
        OptionType option_type);

    /// Single-precision variant.
    PricingResult price_european_gbm_f(
        float spot,
        float strike,
        float rate,
        float volatility,
        float maturity,
        OptionType option_type);

    /// Price a European option under GBM with optional variance reduction.
    ///
    /// @param vr_config  Variance reduction flags (antithetic, control_variate).
    ///                   Flags can be combined; antithetic requires num_paths even.
    PricingResult price_european_gbm(
        double spot,
        double strike,
        double rate,
        double volatility,
        double maturity,
        OptionType option_type,
        const VarianceReductionConfig& vr_config);

    PricingResult price_european_heston(
        double spot,
        double strike,
        double rate,
        double maturity,
        double v0,
        double kappa,
        double theta,
        double xi,
        double rho,
        OptionType option_type);

    /// Price an arithmetic Asian option under GBM on the GPU.
    PricingResult price_asian_gbm(
        double spot,
        double strike,
        double rate,
        double volatility,
        double maturity,
        OptionType option_type);

    /// Price an arithmetic Asian option under Heston on the GPU.
    PricingResult price_asian_heston(
        double spot,
        double strike,
        double rate,
        double maturity,
        double v0,
        double kappa,
        double theta,
        double xi,
        double rho,
        OptionType option_type);

    /// Price a barrier option under GBM on the GPU.
    ///
    /// Uses discrete monitoring with Brownian bridge correction.
    PricingResult price_barrier_gbm(
        double spot,
        double strike,
        double rate,
        double volatility,
        double maturity,
        double barrier,
        BarrierType barrier_type,
        OptionType option_type);

    /// Price a barrier option under Heston on the GPU.
    PricingResult price_barrier_heston(
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
        OptionType option_type);

private:
    std::size_t num_paths_;
    std::size_t num_steps_;
    unsigned long long seed_;
};
