#pragma once

#include <cstddef>

/// Five-moment reduction result for the control-variate estimator.
///
/// Stores the per-block partial sums of:
///   sum_h   — payoffs  h_i
///   sum_h_sq — payoffs squared  h_i^2
///   sum_s   — terminal asset values  S_T_i
///   sum_s_sq — terminal values squared  S_T_i^2
///   sum_hs  — cross-product  h_i * S_T_i
///
/// These five quantities are sufficient to estimate, on the host:
///   beta = Cov(h, S_T) / Var(S_T)
///   mean_h_cv = mean(h) - beta * (mean(S_T) - E[S_T])
///   Var(h_cv) = Var(h) - Cov(h, S_T)^2 / Var(S_T)
template <typename Real>
struct CVReductionResult {
    Real sum_h;
    Real sum_h_sq;
    Real sum_s;
    Real sum_s_sq;
    Real sum_hs;
};

/// Launch a two-pass reduction that computes the five moments needed for the
/// control-variate estimator using S_T as the control variable.
///
/// Pass 1: Each block reduces its chunk in shared memory (5 arrays of BLOCK_SIZE).
/// Pass 2: Host-side final reduction across block partial results.
///
/// @tparam Real             float or double.
/// @param d_terminal_values Device pointer to terminal asset values (num_paths).
/// @param num_paths         Number of elements.
/// @param strike            Strike price K.
/// @param is_call           true for Call payoff, false for Put.
/// @param[out] result       Host-side five-moment result.
template <typename Real>
void launch_payoff_cv_reduction(
    const Real* d_terminal_values,
    std::size_t num_paths,
    Real strike,
    bool is_call,
    CVReductionResult<Real>& result);

extern template void launch_payoff_cv_reduction<float>(
    const float*, std::size_t, float, bool, CVReductionResult<float>&);

extern template void launch_payoff_cv_reduction<double>(
    const double*, std::size_t, double, bool, CVReductionResult<double>&);
