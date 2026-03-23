#include "greeks/greeks.h"
#include "engine/cpu_engine.h"
#include "models/gbm.h"
#include "utils/black_scholes.h"

#include <cmath>
#include <cstddef>

// ---------------------------------------------------------------------------
// Black-Scholes analytical Greeks
// ---------------------------------------------------------------------------

Greeks black_scholes_greeks(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type)
{
    // Use the existing bs_call/put to get d1 and d2.
    const BSResult bs = (option_type == OptionType::Call)
        ? bs_call(spot, strike, rate, volatility, maturity)
        : bs_put(spot, strike, rate, volatility, maturity);

    const double d1       = bs.d1;
    const double d2       = bs.d2;
    const double sqrt_t   = std::sqrt(maturity);
    const double disc     = std::exp(-rate * maturity);
    const double nd1      = normal_cdf(d1);
    const double nd2      = normal_cdf(d2);
    // Standard normal PDF N'(x) = exp(-x²/2) / √(2π)
    const double nprime_d1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);

    Greeks g;

    if (option_type == OptionType::Call) {
        g.delta = nd1;
        g.gamma = nprime_d1 / (spot * volatility * sqrt_t);
        g.vega  = spot * nprime_d1 * sqrt_t;
        // Theta = ∂V/∂t where t is calendar time (not maturity).
        // Theta = -S*N'(d1)*σ/(2√T) - r*K*e^{-rT}*N(d2)
        g.theta = -spot * nprime_d1 * volatility / (2.0 * sqrt_t)
                  - rate * strike * disc * nd2;
        g.rho   = strike * maturity * disc * nd2;
    } else {
        g.delta = nd1 - 1.0;
        g.gamma = nprime_d1 / (spot * volatility * sqrt_t);
        g.vega  = spot * nprime_d1 * sqrt_t;
        // Theta = -S*N'(d1)*σ/(2√T) + r*K*e^{-rT}*N(-d2)
        g.theta = -spot * nprime_d1 * volatility / (2.0 * sqrt_t)
                  + rate * strike * disc * normal_cdf(-d2);
        g.rho   = -strike * maturity * disc * normal_cdf(-d2);
    }

    return g;
}

// ---------------------------------------------------------------------------
// CPU finite-difference Greeks (common random numbers via same seed)
// ---------------------------------------------------------------------------

Greeks compute_greeks_gbm_cpu(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long seed)
{
    const double eps_s = 0.01 * spot;
    const double eps_v = 0.001;
    const double eps_t = 1.0 / 365.0;
    const double eps_r = 0.0001;

    // Lambda: price under modified params, same seed for CRN.
    auto price = [&](double s, double v, double t, double r) -> double {
        GBM<double> model(r, v);
        CPUEngine engine(num_paths, num_steps, seed);
        return engine.price_european(model, s, strike, t, option_type).price;
    };

    const double v0   = price(spot,        volatility,       maturity,       rate);
    const double v_sp = price(spot + eps_s, volatility,       maturity,       rate);
    const double v_sm = price(spot - eps_s, volatility,       maturity,       rate);
    const double v_vp = price(spot,        volatility + eps_v, maturity,      rate);
    const double v_vm = price(spot,        volatility - eps_v, maturity,      rate);
    const double v_tm = price(spot,        volatility,       maturity - eps_t, rate);
    const double v_rp = price(spot,        volatility,       maturity,       rate + eps_r);
    const double v_rm = price(spot,        volatility,       maturity,       rate - eps_r);

    Greeks g;
    g.delta = (v_sp - v_sm) / (2.0 * eps_s);
    g.gamma = (v_sp - 2.0 * v0 + v_sm) / (eps_s * eps_s);
    g.vega  = (v_vp - v_vm) / (2.0 * eps_v);
    g.theta = (v_tm - v0) / eps_t;   // negative for long options (value decays)
    g.rho   = (v_rp - v_rm) / (2.0 * eps_r);

    return g;
}
