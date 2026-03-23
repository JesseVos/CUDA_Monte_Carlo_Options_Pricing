#pragma once

/// Closed-form Black-Scholes prices and Greeks for European options.
///
/// Used as the validation reference for Monte Carlo results.
///
/// Formulas:
///   d1 = (ln(S/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T))
///   d2 = d1 - sigma*sqrt(T)
///   Call = S*N(d1) - K*exp(-r*T)*N(d2)
///   Put  = K*exp(-r*T)*N(-d2) - S*N(-d1)

struct BSResult {
    double price;
    double d1;
    double d2;
};

/// Compute Black-Scholes European call price.
BSResult bs_call(double spot, double strike, double rate,
                 double volatility, double maturity);

/// Compute Black-Scholes European put price.
BSResult bs_put(double spot, double strike, double rate,
                double volatility, double maturity);

/// Standard normal CDF (cumulative distribution function).
double normal_cdf(double x);
