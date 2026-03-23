#pragma once

#include "pricing/european.h"

#include <cstddef>

/// Risk sensitivities of an option price.
///
/// All values are in the same units as the option price (dollars per dollar
/// of notional) unless otherwise noted.
///
///   delta  — ∂V/∂S         (per dollar of spot)
///   gamma  — ∂²V/∂S²       (per dollar² of spot)
///   vega   — ∂V/∂σ         (per unit of volatility, e.g., +0.01 vol → vega*0.01 P&L)
///   theta  — ∂V/∂t         ((V(T-ε) - V(T)) / ε, negative for long options)
///   rho    — ∂V/∂r         (per unit of interest rate)
struct Greeks {
    double delta = 0.0;
    double gamma = 0.0;
    double vega  = 0.0;
    double theta = 0.0;
    double rho   = 0.0;
};

/// Finite-difference bump sizes (following AGENTS.md conventions).
struct GreeksBumps {
    double eps_spot = 0.0;   ///< Filled as 0.01 * spot at call time.
    double eps_vol  = 0.001;
    double eps_time = 1.0 / 365.0;
    double eps_rate = 0.0001;
};

/// Compute Black-Scholes closed-form Greeks for a European option.
///
/// Used as the validation reference for the Monte Carlo estimates.
Greeks black_scholes_greeks(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type);

/// Compute finite-difference Greeks on the CPU using 8 re-pricings.
///
/// Common random numbers: all 8 scenarios use the same RNG seed, so the
/// finite-difference noise is dramatically reduced compared to independent
/// simulations.
///
/// Scenarios:
///   base     (S,   σ,   T,   r  )
///   delta_up (S+ε, σ,   T,   r  )   delta_down (S-ε, ...)
///   vega_up  (S,   σ+ε, T,   r  )   vega_down  (S,   σ-ε, ...)
///   theta    (S,   σ,   T-ε, r  )
///   rho_up   (S,   σ,   T,   r+ε)   rho_down   (S,   σ,   T,   r-ε)
Greeks compute_greeks_gbm_cpu(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long seed);

/// Compute finite-difference Greeks on the GPU using a single batched kernel.
///
/// All 8 * num_paths paths are launched together.  Common random numbers are
/// ensured by initialising every thread for path p (in any scenario) with
/// curand_init(seed, p, 0, &state).
Greeks compute_greeks_gbm_gpu(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);
