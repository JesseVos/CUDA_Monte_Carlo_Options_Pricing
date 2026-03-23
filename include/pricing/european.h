#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

/// European option payoff computation.
///
/// European options have payoffs that depend only on the terminal asset price:
///   Call: max(S_T - K, 0)
///   Put:  max(K - S_T, 0)
///
/// The price is the discounted average of payoffs across all Monte Carlo paths.

enum class OptionType {
    Call,
    Put
};

/// Result of a Monte Carlo pricing run, including the estimated standard error.
struct PricingResult {
    double price;           ///< Discounted expected payoff.
    double standard_error;  ///< Standard error of the Monte Carlo estimate.
};

/// Compute the discounted Monte Carlo price and standard error for a
/// European option given a vector of terminal asset values.
///
/// @param terminal_values  Terminal asset prices S(T) from simulation.
/// @param strike           Strike price K.
/// @param rate             Risk-free rate r.
/// @param maturity         Time to expiration T.
/// @param option_type      Call or Put.
/// @return PricingResult containing the price and standard error.
PricingResult price_european(
    const std::vector<double>& terminal_values,
    double strike,
    double rate,
    double maturity,
    OptionType option_type);
