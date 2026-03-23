#pragma once

#include "models/model.h"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#include <algorithm>

/// Geometric Brownian Motion (GBM) model under risk-neutral measure.
///
/// dS = r * S * dt + sigma * S * dW
///
/// Path generation uses the log-Euler (exact) scheme:
///   S(t+dt) = S(t) * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
///
/// This avoids discretization bias and guarantees positive stock prices.
template <typename Real = double>
class GBM : public Model<Real> {
public:
    /// @param rate       Risk-free interest rate (annualized).
    /// @param volatility Constant volatility sigma (annualized).
    GBM(Real rate, Real volatility)
        : rate_(rate), volatility_(volatility) {}

    void generate_paths_cpu(
        Real spot,
        Real maturity,
        std::size_t num_paths,
        std::size_t num_steps,
        unsigned long seed,
        std::vector<Real>& terminal_values) const override
    {
        terminal_values.resize(num_paths);

        Real dt = maturity / static_cast<Real>(num_steps);
        Real drift = (rate_ - static_cast<Real>(0.5) * volatility_ * volatility_) * dt;
        Real diffusion = volatility_ * std::sqrt(dt);

        std::mt19937 rng(seed);
        std::normal_distribution<Real> normal(static_cast<Real>(0.0),
                                              static_cast<Real>(1.0));

        for (std::size_t i = 0; i < num_paths; ++i) {
            Real log_s = std::log(spot);
            for (std::size_t step = 0; step < num_steps; ++step) {
                Real z = normal(rng);
                log_s += drift + diffusion * z;
            }
            terminal_values[i] = std::exp(log_s);
        }
    }

    /// Antithetic variates: num_paths/2 draws, each producing two terminal
    /// values with opposite Brownian increments.
    ///
    /// Layout:
    ///   terminal_values[i]             — positive-draw path
    ///   terminal_values[i + half]      — antithetic path (−Z draws)
    ///
    /// @pre num_paths must be even.
    void generate_paths_antithetic_cpu(
        Real spot,
        Real maturity,
        std::size_t num_paths,
        std::size_t num_steps,
        unsigned long seed,
        std::vector<Real>& terminal_values) const override
    {
        terminal_values.resize(num_paths);

        std::size_t half = num_paths / 2;
        Real dt        = maturity / static_cast<Real>(num_steps);
        Real drift     = (rate_ - static_cast<Real>(0.5) * volatility_ * volatility_) * dt;
        Real diffusion = volatility_ * std::sqrt(dt);

        std::mt19937 rng(seed);
        std::normal_distribution<Real> normal(static_cast<Real>(0.0),
                                              static_cast<Real>(1.0));

        for (std::size_t i = 0; i < half; ++i) {
            Real log_s_pos = std::log(spot);
            Real log_s_neg = std::log(spot);
            for (std::size_t step = 0; step < num_steps; ++step) {
                Real z  = normal(rng);
                Real dz = diffusion * z;
                log_s_pos += drift + dz;
                log_s_neg += drift - dz;
            }
            terminal_values[i]        = std::exp(log_s_pos);
            terminal_values[i + half] = std::exp(log_s_neg);
        }
    }

    void generate_asian_paths_cpu(
        Real spot,
        Real maturity,
        std::size_t num_paths,
        std::size_t num_steps,
        unsigned long seed,
        std::vector<Real>& averages) const override
    {
        averages.resize(num_paths);

        const Real dt        = maturity / static_cast<Real>(num_steps);
        const Real drift     = (rate_ - static_cast<Real>(0.5) * volatility_ * volatility_) * dt;
        const Real diffusion = volatility_ * std::sqrt(dt);

        std::mt19937 rng(seed);
        std::normal_distribution<Real> normal(Real(0), Real(1));

        for (std::size_t i = 0; i < num_paths; ++i) {
            Real s   = spot;
            Real sum = Real(0);
            for (std::size_t step = 0; step < num_steps; ++step) {
                s *= std::exp(drift + diffusion * normal(rng));
                sum += s;
            }
            averages[i] = sum / static_cast<Real>(num_steps);
        }
    }

    /// Barrier option path generation with Brownian bridge correction.
    ///
    /// At each step, after moving S to its new value, a discrete barrier check
    /// is performed.  When both the previous and new S lie on the safe side of
    /// the barrier, the Brownian bridge crossing probability
    ///
    ///   P = exp( -2 * |ln(B/S_prev)| * |ln(B/S_new)| / (sigma^2 * dt) )
    ///
    /// is computed and compared to a uniform draw to correct for the bias
    /// introduced by discrete monitoring.
    void generate_barrier_paths_cpu(
        Real spot,
        Real maturity,
        std::size_t num_paths,
        std::size_t num_steps,
        unsigned long seed,
        Real barrier,
        bool is_upper,
        bool is_knockout,
        std::vector<Real>& terminal_values) const override
    {
        terminal_values.resize(num_paths);

        const Real dt        = maturity / static_cast<Real>(num_steps);
        const Real drift     = (rate_ - Real(0.5) * volatility_ * volatility_) * dt;
        const Real diffusion = volatility_ * std::sqrt(dt);
        const Real vol_sq_dt = volatility_ * volatility_ * dt;

        std::mt19937 rng(seed);
        std::normal_distribution<Real> normal(Real(0), Real(1));
        std::uniform_real_distribution<Real> uniform(Real(0), Real(1));

        for (std::size_t i = 0; i < num_paths; ++i) {
            Real s            = spot;
            bool barrier_hit  = false;

            for (std::size_t step = 0; step < num_steps; ++step) {
                // Early exit for knock-out: once dead the payoff is 0.
                if (is_knockout && barrier_hit) break;

                const Real s_prev = s;
                s *= std::exp(drift + diffusion * normal(rng));

                // Discrete crossing check.
                const bool crossed = is_upper ? (s >= barrier) : (s <= barrier);
                if (crossed) {
                    barrier_hit = true;
                    continue;
                }

                // Brownian bridge correction (both endpoints on safe side).
                const Real la = is_upper
                    ? std::log(barrier / s_prev)
                    : std::log(s_prev / barrier);
                const Real lb = is_upper
                    ? std::log(barrier / s)
                    : std::log(s / barrier);

                if (la > Real(0) && lb > Real(0) && vol_sq_dt > Real(0)) {
                    const Real p_cross =
                        std::exp(Real(-2) * la * lb / vol_sq_dt);
                    if (uniform(rng) < p_cross) {
                        barrier_hit = true;
                    }
                }
            }

            // Knock-out pays if never hit; knock-in pays only if hit.
            const bool pays = is_knockout ? !barrier_hit : barrier_hit;
            terminal_values[i] = pays ? s : Real(0);
        }
    }

    Real rate() const override { return rate_; }

    Real volatility() const { return volatility_; }

private:
    Real rate_;
    Real volatility_;
};
