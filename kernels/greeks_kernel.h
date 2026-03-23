#pragma once

#include <cstddef>

/// Launch the batched GBM Greeks kernel.
///
/// Runs 8 * num_paths threads in a single kernel launch.
/// Thread ordering: scenario-major — threads [s*num_paths, (s+1)*num_paths)
/// belong to scenario s, ensuring coalesced writes.
///
/// Scenarios (s):
///   0 — base         (S,    σ,    T,    r   )
///   1 — delta_up     (S+εS, σ,    T,    r   )
///   2 — delta_down   (S-εS, σ,    T,    r   )
///   3 — vega_up      (S,    σ+εσ, T,    r   )
///   4 — vega_down    (S,    σ-εσ, T,    r   )
///   5 — theta        (S,    σ,    T-εT, r   )
///   6 — rho_up       (S,    σ,    T,    r+εr)
///   7 — rho_down     (S,    σ,    T,    r-εr)
///
/// CRN: thread for (scenario s, path p) initialises cuRAND as
///      curand_init(seed, p, 0, &state) — identical to every other scenario's
///      path p — ensuring maximum cancellation in finite differences.
///
/// Output d_payoffs[s * num_paths + p] = exp(-r_s * T_s) * payoff_s(p).
void launch_greeks_gbm_kernel(
    double* d_payoffs,
    double spot,
    double rate,
    double volatility,
    double maturity,
    double strike,
    bool   is_call,
    double eps_spot,
    double eps_vol,
    double eps_time,
    double eps_rate,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed);
