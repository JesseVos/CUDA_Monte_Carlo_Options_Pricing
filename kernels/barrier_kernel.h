#pragma once

#include <cstddef>

/// Launch the GBM barrier path-generation kernel.
///
/// Each thread simulates one GBM path, checking the barrier at every timestep
/// (discrete monitoring).  When both the previous and current S lie on the safe
/// side, a Brownian bridge correction is applied:
///
///   P_cross = exp( -2 * |ln(B/S_prev)| * |ln(B/S)| / (sigma^2 * dt) )
///
/// A uniform draw decides whether the path crossed between steps.
///
/// Output: d_terminal_values[tid] = S(T) if the path pays, 0 otherwise.
///   Knock-out pays if barrier never hit; knock-in pays if barrier was hit.
template <typename Real>
void launch_barrier_gbm_kernel(
    Real* d_terminal_values,
    Real spot,
    Real rate,
    Real volatility,
    Real maturity,
    Real barrier,
    bool is_upper,
    bool is_knockout,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);

extern template void launch_barrier_gbm_kernel<float>(
    float*, float, float, float, float, float, bool, bool,
    std::size_t, std::size_t, unsigned long long);

extern template void launch_barrier_gbm_kernel<double>(
    double*, double, double, double, double, double, bool, bool,
    std::size_t, std::size_t, unsigned long long);
