#pragma once

/// Configuration for variance reduction techniques.
///
/// Both flags are independent and can be combined:
///   antithetic:     for each draw Z, simulate two paths using +Z and -Z.
///                   Requires num_paths to be even.
///   control_variate: use S_T as control variable with known risk-neutral
///                   expectation E[S_T] = S_0 * exp(r*T).
struct VarianceReductionConfig {
    bool antithetic      = false;
    bool control_variate = false;
};
