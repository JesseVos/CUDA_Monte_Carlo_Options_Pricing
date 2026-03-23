#pragma once

#include "models/model.h"
#include "pricing/european.h"
#include "pricing/barrier.h"
#include "variance/variance_reduction.h"

#include <cstddef>
#include <memory>

/// CPU-based Monte Carlo simulation engine.
///
/// Drives path generation via a Model and computes option prices by
/// averaging discounted payoffs. Serves as the reference implementation
/// for validation against the GPU engine (Week 2).
class CPUEngine {
public:
    /// @param num_paths  Number of Monte Carlo paths to simulate.
    /// @param num_steps  Number of time steps per path.
    /// @param seed       RNG seed for reproducibility.
    CPUEngine(std::size_t num_paths, std::size_t num_steps, unsigned long seed);

    /// Price a European option using the given model.
    ///
    /// @param model       Stochastic model (e.g., GBM) for path generation.
    /// @param spot        Initial asset price.
    /// @param strike      Strike price.
    /// @param maturity    Time to expiration (years).
    /// @param option_type Call or Put.
    /// @return PricingResult with price and standard error.
    PricingResult price_european(
        const Model<double>& model,
        double spot,
        double strike,
        double maturity,
        OptionType option_type);

    /// Price a European option with optional variance reduction.
    ///
    /// @param vr_config  Variance reduction flags (antithetic, control_variate).
    PricingResult price_european(
        const Model<double>& model,
        double spot,
        double strike,
        double maturity,
        OptionType option_type,
        const VarianceReductionConfig& vr_config);

    /// Price an arithmetic Asian option using the given model.
    ///
    /// The payoff is max(arithmetic_avg(S) - K, 0) for a call.
    /// Averaging is over all num_steps monitoring times (S_0 excluded).
    PricingResult price_asian(
        const Model<double>& model,
        double spot,
        double strike,
        double maturity,
        OptionType option_type);

    /// Price a barrier option using the given model.
    ///
    /// Discrete monitoring with Brownian bridge correction applied by the model.
    ///
    /// @param barrier      Barrier level B.
    /// @param barrier_type UpAndOut / DownAndOut / UpAndIn / DownAndIn.
    PricingResult price_barrier(
        const Model<double>& model,
        double spot,
        double strike,
        double maturity,
        double barrier,
        BarrierType barrier_type,
        OptionType option_type);

private:
    std::size_t num_paths_;
    std::size_t num_steps_;
    unsigned long seed_;
};
