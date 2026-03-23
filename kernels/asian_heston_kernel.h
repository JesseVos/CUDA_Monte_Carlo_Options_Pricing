#pragma once

#include <cstddef>

/// Launch the Heston Asian path-generation kernel (double precision only).
///
/// Each thread simulates one Heston path and writes the arithmetic mean of
/// S(t_1), ..., S(t_N) into d_path_averages[tid].
void launch_asian_heston_kernel(
    double* d_path_averages,
    double spot,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);
