#include "utils/black_scholes.h"

#include <cmath>
#include <stdexcept>

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

BSResult bs_call(double spot, double strike, double rate,
                 double volatility, double maturity)
{
    if (maturity <= 0.0) {
        // At expiry, intrinsic value.
        double intrinsic = std::max(spot - strike, 0.0);
        return {intrinsic, 0.0, 0.0};
    }
    if (volatility <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }

    double sqrt_t = std::sqrt(maturity);
    double d1 = (std::log(spot / strike) + (rate + 0.5 * volatility * volatility) * maturity)
                / (volatility * sqrt_t);
    double d2 = d1 - volatility * sqrt_t;

    double price = spot * normal_cdf(d1)
                   - strike * std::exp(-rate * maturity) * normal_cdf(d2);

    return {price, d1, d2};
}

BSResult bs_put(double spot, double strike, double rate,
                double volatility, double maturity)
{
    if (maturity <= 0.0) {
        double intrinsic = std::max(strike - spot, 0.0);
        return {intrinsic, 0.0, 0.0};
    }
    if (volatility <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }

    double sqrt_t = std::sqrt(maturity);
    double d1 = (std::log(spot / strike) + (rate + 0.5 * volatility * volatility) * maturity)
                / (volatility * sqrt_t);
    double d2 = d1 - volatility * sqrt_t;

    double price = strike * std::exp(-rate * maturity) * normal_cdf(-d2)
                   - spot * normal_cdf(-d1);

    return {price, d1, d2};
}
