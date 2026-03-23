#include "pricing/european.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

PricingResult price_european(
    const std::vector<double>& terminal_values,
    double strike,
    double rate,
    double maturity,
    OptionType option_type)
{
    std::size_t n = terminal_values.size();
    if (n == 0) {
        throw std::invalid_argument("terminal_values must not be empty");
    }

    double discount = std::exp(-rate * maturity);

    // Compute payoffs and accumulate sum and sum-of-squares for standard error.
    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        double payoff = 0.0;
        if (option_type == OptionType::Call) {
            payoff = std::max(terminal_values[i] - strike, 0.0);
        } else {
            payoff = std::max(strike - terminal_values[i], 0.0);
        }
        sum_payoff += payoff;
        sum_payoff_sq += payoff * payoff;
    }

    double mean_payoff = sum_payoff / static_cast<double>(n);
    double price = discount * mean_payoff;

    // Standard error of the discounted payoff estimator.
    // Var(payoff) = E[payoff^2] - (E[payoff])^2
    // SE(price) = discount * sqrt(Var(payoff) / N)
    double variance = (sum_payoff_sq / static_cast<double>(n))
                      - (mean_payoff * mean_payoff);
    // Clamp numerical noise to zero.
    variance = std::max(variance, 0.0);
    double standard_error = discount * std::sqrt(variance / static_cast<double>(n));

    return {price, standard_error};
}
