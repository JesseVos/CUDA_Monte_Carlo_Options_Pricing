#pragma once

// Semi-analytical Heston call pricing via characteristic-function integration.
//
// Parameters:
//   spot, strike, rate, maturity
//   v0, kappa, theta, xi, rho
//
// Returns European call price under risk-neutral Heston dynamics.
double heston_call_price(
    double spot,
    double strike,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho);

double heston_put_price(
    double spot,
    double strike,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho);
