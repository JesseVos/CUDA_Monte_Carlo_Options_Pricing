#pragma once

#include <cstddef>

/// Launch the GBM path generation kernel on the GPU.
///
/// One thread per Monte Carlo path. Each thread initializes its own cuRAND
/// Philox4_32_10 state, simulates a single GBM path using the log-Euler
/// scheme, and writes the terminal asset value to d_terminal_values[thread_id].
///
/// @tparam Real          float or double.
/// @param d_terminal_values  Device pointer, output array of size num_paths.
/// @param spot               Initial asset price S(0).
/// @param rate               Risk-free rate r.
/// @param volatility         Constant volatility sigma.
/// @param maturity           Time to expiration T (years).
/// @param num_paths          Number of Monte Carlo paths.
/// @param num_steps          Number of time steps per path.
/// @param seed               Base RNG seed.
template <typename Real>
void launch_gbm_kernel(
    Real* d_terminal_values,
    Real spot,
    Real rate,
    Real volatility,
    Real maturity,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);

// Explicit declarations for float and double.
extern template void launch_gbm_kernel<float>(
    float*, float, float, float, float,
    std::size_t, std::size_t, unsigned long long);

extern template void launch_gbm_kernel<double>(
    double*, double, double, double, double,
    std::size_t, std::size_t, unsigned long long);
