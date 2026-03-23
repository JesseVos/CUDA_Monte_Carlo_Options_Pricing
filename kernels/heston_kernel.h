#pragma once

#include <cstddef>

template <typename Real>
void launch_heston_kernel(
    Real* d_terminal_values,
    Real spot,
    Real rate,
    Real maturity,
    Real v0,
    Real kappa,
    Real theta,
    Real xi,
    Real rho,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);

extern template void launch_heston_kernel<float>(
    float*, float, float, float, float, float, float, float, float,
    std::size_t, std::size_t, unsigned long long);

extern template void launch_heston_kernel<double>(
    double*, double, double, double, double, double, double, double, double,
    std::size_t, std::size_t, unsigned long long);
