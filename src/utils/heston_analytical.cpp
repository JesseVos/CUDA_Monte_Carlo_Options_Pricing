#include "utils/heston_analytical.h"

#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>

namespace {

using Complex = std::complex<double>;

double simpson_integral(const std::function<double(double)>& f,
                        double a,
                        double b,
                        int n)
{
    if (n % 2 != 0) {
        ++n;
    }
    const double h = (b - a) / static_cast<double>(n);

    double sum = f(a) + f(b);
    for (int i = 1; i < n; ++i) {
        const double x = a + h * static_cast<double>(i);
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }
    return sum * h / 3.0;
}

Complex heston_cf(
    Complex u,
    double spot,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho)
{
    const Complex i(0.0, 1.0);
    const Complex alpha = -0.5 * (u * u + i * u);
    const Complex beta = kappa - rho * xi * i * u;
    const Complex gamma = 0.5 * xi * xi;

    const Complex d = std::sqrt(beta * beta - 4.0 * alpha * gamma);
    const Complex g = (beta - d) / (beta + d);

    const Complex exp_dt = std::exp(-d * maturity);
    const Complex log_term = std::log((1.0 - g * exp_dt) / (1.0 - g));

    const Complex A = (kappa * theta / (xi * xi)) *
                      ((beta - d) * maturity - 2.0 * log_term);
    const Complex B = ((beta - d) / (xi * xi)) *
                      ((1.0 - exp_dt) / (1.0 - g * exp_dt));

    return std::exp(A + B * v0 + i * u * (std::log(spot) + rate * maturity));
}

}  // namespace

double heston_call_price(
    double spot,
    double strike,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho)
{
    if (spot <= 0.0 || strike <= 0.0 || maturity <= 0.0 || xi <= 0.0) {
        throw std::invalid_argument("Invalid Heston parameters");
    }

    const double log_k = std::log(strike);
    const Complex i(0.0, 1.0);
    const Complex cf_minus_i = heston_cf(Complex(0.0, -1.0),
                                         spot, rate, maturity, v0, kappa, theta, xi, rho);

    auto integrand_p1 = [&](double u) -> double {
        if (u == 0.0) return 0.0;
        const Complex uc(u, 0.0);
        const Complex numer = std::exp(-i * uc * log_k) * heston_cf(uc - i, spot, rate, maturity,
                                                                     v0, kappa, theta, xi, rho);
        const Complex denom = i * uc * cf_minus_i;
        return std::real(numer / denom);
    };

    auto integrand_p2 = [&](double u) -> double {
        if (u == 0.0) return 0.0;
        const Complex uc(u, 0.0);
        const Complex numer = std::exp(-i * uc * log_k) *
                              heston_cf(uc, spot, rate, maturity, v0, kappa, theta, xi, rho);
        const Complex denom = i * uc;
        return std::real(numer / denom);
    };

    constexpr double integration_upper = 100.0;
    constexpr int integration_steps = 6000;

    const double p1 = 0.5 + (1.0 / M_PI) *
        simpson_integral(integrand_p1, 0.0, integration_upper, integration_steps);
    const double p2 = 0.5 + (1.0 / M_PI) *
        simpson_integral(integrand_p2, 0.0, integration_upper, integration_steps);

    return spot * p1 - strike * std::exp(-rate * maturity) * p2;
}

double heston_put_price(
    double spot,
    double strike,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho)
{
    const double call = heston_call_price(
        spot, strike, rate, maturity, v0, kappa, theta, xi, rho);
    return call - spot + strike * std::exp(-rate * maturity);
}
