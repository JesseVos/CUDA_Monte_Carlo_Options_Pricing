#pragma once

double implied_vol_call_bisection(
    double target_call_price,
    double spot,
    double strike,
    double rate,
    double maturity,
    double vol_low = 1e-6,
    double vol_high = 3.0,
    int max_iter = 200,
    double tol = 1e-8);
