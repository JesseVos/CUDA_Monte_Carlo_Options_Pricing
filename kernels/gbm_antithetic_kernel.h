#pragma once

#include <cstddef>

/// Launch the antithetic GBM path generation kernel on the GPU.
///
/// Each of (num_paths / 2) threads draws a single standard normal Z and
/// simulates two paths: one with +Z (stored at index tid) and one with -Z
/// (stored at index tid + num_paths/2).  Because the two payoffs are
/// negatively correlated the average has lower variance than two independent
/// draws — this is the antithetic variates technique.
///
/// The full output array has num_paths elements and is passed to the standard
/// payoff reduction kernel unchanged.
///
/// @pre  num_paths must be even.
/// @tparam Real              float or double.
/// @param d_terminal_values  Device pointer, output array of size num_paths.
/// @param spot               Initial asset price S(0).
/// @param rate               Risk-free rate r.
/// @param volatility         Constant volatility sigma.
/// @param maturity           Time to expiration T (years).
/// @param num_paths          Total number of terminal values to generate (even).
/// @param num_steps          Number of time steps per path.
/// @param seed               Base RNG seed.
template <typename Real>
void launch_gbm_antithetic_kernel(
    Real* d_terminal_values,
    Real spot,
    Real rate,
    Real volatility,
    Real maturity,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);

extern template void launch_gbm_antithetic_kernel<float>(
    float*, float, float, float, float,
    std::size_t, std::size_t, unsigned long long);

extern template void launch_gbm_antithetic_kernel<double>(
    double*, double, double, double, double,
    std::size_t, std::size_t, unsigned long long);
