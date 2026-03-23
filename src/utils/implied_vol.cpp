#include "utils/implied_vol.h"

#include "utils/black_scholes.h"

#include <algorithm>
#include <stdexcept>

double implied_vol_call_bisection(
    double target_call_price,
    double spot,
    double strike,
    double rate,
    double maturity,
    double vol_low,
    double vol_high,
    int max_iter,
    double tol)
{
    if (target_call_price < 0.0) {
        throw std::invalid_argument("Negative option price");
    }

    double low = vol_low;
    double high = vol_high;
    double mid = 0.5 * (low + high);

    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5 * (low + high);
        const double price = bs_call(spot, strike, rate, mid, maturity).price;

        if (std::abs(price - target_call_price) < tol) {
            return mid;
        }
        if (price > target_call_price) {
            high = mid;
        } else {
            low = mid;
        }
    }
    return mid;
}
