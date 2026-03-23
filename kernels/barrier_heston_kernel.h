#pragma once

#include <cstddef>

/// Launch the Heston barrier path-generation kernel (double precision only).
///
/// Discrete barrier monitoring with Brownian bridge correction using the
/// instantaneous variance v_pos as a proxy for sigma^2.
///
/// Output: d_terminal_values[tid] = S(T) if path pays, 0 otherwise.
void launch_barrier_heston_kernel(
    double* d_terminal_values,
    double spot,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double barrier,
    bool is_upper,
    bool is_knockout,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);
