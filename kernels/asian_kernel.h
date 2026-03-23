#pragma once

#include <cstddef>

/// Launch the GBM Asian path-generation kernel.
///
/// Each thread simulates one path under GBM and writes the arithmetic mean of
/// S(t_1), ..., S(t_N) into path_averages[tid].  The reduction kernel can then
/// be reused directly on these averages to compute max(avg - K, 0).
template <typename Real>
void launch_asian_gbm_kernel(
    Real* d_path_averages,
    Real spot,
    Real rate,
    Real volatility,
    Real maturity,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);

extern template void launch_asian_gbm_kernel<float>(
    float*, float, float, float, float,
    std::size_t, std::size_t, unsigned long long);

extern template void launch_asian_gbm_kernel<double>(
    double*, double, double, double, double,
    std::size_t, std::size_t, unsigned long long);
