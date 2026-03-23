#pragma once

#include <cstddef>

/// Result of a GPU reduction: sum of values and sum of squared values.
/// Used to compute mean and variance (for standard error) on the host.
template <typename Real>
struct ReductionResult {
    Real sum;
    Real sum_sq;
};

/// Launch a two-pass reduction kernel that computes the sum and sum-of-squares
/// of payoffs from terminal asset values.
///
/// Pass 1: Each block reduces its chunk using shared memory, writing
///          per-block partial sums to d_block_sums and d_block_sums_sq.
/// Pass 2: Host-side final reduction across blocks.
///
/// The payoff is computed inline in the kernel:
///   Call: max(S_T - strike, 0)
///   Put:  max(strike - S_T, 0)
///
/// @tparam Real             float or double.
/// @param d_terminal_values Device pointer to terminal asset values (num_paths).
/// @param num_paths         Number of elements.
/// @param strike            Strike price K.
/// @param is_call           true for Call, false for Put.
/// @param[out] result       Host-side sum and sum_sq of payoffs.
template <typename Real>
void launch_payoff_reduction(
    const Real* d_terminal_values,
    std::size_t num_paths,
    Real strike,
    bool is_call,
    ReductionResult<Real>& result);

extern template void launch_payoff_reduction<float>(
    const float*, std::size_t, float, bool, ReductionResult<float>&);

extern template void launch_payoff_reduction<double>(
    const double*, std::size_t, double, bool, ReductionResult<double>&);

/// Antithetic-pair reduction.
///
/// For an array of num_paths terminal values produced by the antithetic kernel
/// (first half = positive-draw paths, second half = negative-draw paths),
/// this computes the sum and sum-of-squares of the N/2 pair averages:
///
///   h_avg_i = (payoff(S_T[i]) + payoff(S_T[i + num_paths/2])) / 2
///
/// The caller computes:
///   effective_n = num_paths / 2
///   mean  = result.sum / effective_n
///   variance = (result.sum_sq / effective_n) - mean^2
///   SE = discount * sqrt(variance / effective_n)
///
/// This gives the correct standard error for the antithetic estimator.
///
/// @pre num_paths must be even.
template <typename Real>
void launch_antithetic_payoff_reduction(
    const Real* d_terminal_values,
    std::size_t num_paths,
    Real strike,
    bool is_call,
    ReductionResult<Real>& result);

extern template void launch_antithetic_payoff_reduction<float>(
    const float*, std::size_t, float, bool, ReductionResult<float>&);

extern template void launch_antithetic_payoff_reduction<double>(
    const double*, std::size_t, double, bool, ReductionResult<double>&);
