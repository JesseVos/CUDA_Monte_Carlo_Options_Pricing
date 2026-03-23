#pragma once

#include "models/model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

// Heston stochastic volatility model under risk-neutral measure.
//
//   dS = r S dt + sqrt(v+) S dW_s
//   dv = kappa(theta - v+) dt + xi sqrt(v+) dW_v
//   dW_s dW_v = rho dt
//
// with v+ = max(v, 0) (full truncation Euler scheme).
template <typename Real = double>
class Heston : public Model<Real> {
public:
    Heston(Real rate,
           Real v0,
           Real kappa,
           Real theta,
           Real xi,
           Real rho)
        : rate_(rate), v0_(v0), kappa_(kappa), theta_(theta), xi_(xi), rho_(rho)
    {
    }

    void generate_paths_cpu(
        Real spot,
        Real maturity,
        std::size_t num_paths,
        std::size_t num_steps,
        unsigned long seed,
        std::vector<Real>& terminal_values) const override
    {
        terminal_values.resize(num_paths);

        const Real dt = maturity / static_cast<Real>(num_steps);
        const Real sqrt_dt = std::sqrt(dt);
        const Real rho_bar = std::sqrt(std::max(static_cast<Real>(0),
                                                static_cast<Real>(1) - rho_ * rho_));

        std::mt19937 rng(seed);
        std::normal_distribution<Real> normal(static_cast<Real>(0), static_cast<Real>(1));

        for (std::size_t i = 0; i < num_paths; ++i) {
            Real s = spot;
            Real v = v0_;

            for (std::size_t step = 0; step < num_steps; ++step) {
                const Real z_v = normal(rng);
                const Real z_i = normal(rng);
                const Real z_s = rho_ * z_v + rho_bar * z_i;

                const Real v_pos = std::max(v, static_cast<Real>(0));
                const Real sqrt_v = std::sqrt(v_pos);

                v += kappa_ * (theta_ - v_pos) * dt + xi_ * sqrt_v * sqrt_dt * z_v;
                s *= std::exp((rate_ - static_cast<Real>(0.5) * v_pos) * dt
                              + sqrt_v * sqrt_dt * z_s);
            }

            terminal_values[i] = s;
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

        const Real dt      = maturity / static_cast<Real>(num_steps);
        const Real sqrt_dt = std::sqrt(dt);
        const Real rho_bar = std::sqrt(std::max(Real(0), Real(1) - rho_ * rho_));

        std::mt19937 rng(seed);
        std::normal_distribution<Real> normal(Real(0), Real(1));

        for (std::size_t i = 0; i < num_paths; ++i) {
            Real s   = spot;
            Real v   = v0_;
            Real sum = Real(0);

            for (std::size_t step = 0; step < num_steps; ++step) {
                const Real z_v  = normal(rng);
                const Real z_i  = normal(rng);
                const Real z_s  = rho_ * z_v + rho_bar * z_i;
                const Real v_pos   = std::max(v, Real(0));
                const Real sqrt_v  = std::sqrt(v_pos);

                v += kappa_ * (theta_ - v_pos) * dt + xi_ * sqrt_v * sqrt_dt * z_v;
                s *= std::exp((rate_ - Real(0.5) * v_pos) * dt
                              + sqrt_v * sqrt_dt * z_s);
                sum += s;
            }
            averages[i] = sum / static_cast<Real>(num_steps);
        }
    }

    /// Heston barrier path generation with Brownian bridge correction.
    ///
    /// The BB correction uses the instantaneous variance v_pos as a proxy for
    /// the local volatility squared (standard approximation for Heston).
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

        const Real dt      = maturity / static_cast<Real>(num_steps);
        const Real sqrt_dt = std::sqrt(dt);
        const Real rho_bar = std::sqrt(std::max(Real(0), Real(1) - rho_ * rho_));

        std::mt19937 rng(seed);
        std::normal_distribution<Real> normal(Real(0), Real(1));
        std::uniform_real_distribution<Real> uniform(Real(0), Real(1));

        for (std::size_t i = 0; i < num_paths; ++i) {
            Real s           = spot;
            Real v           = v0_;
            bool barrier_hit = false;

            for (std::size_t step = 0; step < num_steps; ++step) {
                if (is_knockout && barrier_hit) break;

                const Real s_prev  = s;
                const Real z_v     = normal(rng);
                const Real z_i     = normal(rng);
                const Real z_s     = rho_ * z_v + rho_bar * z_i;
                const Real v_pos   = std::max(v, Real(0));
                const Real sqrt_v  = std::sqrt(v_pos);

                v += kappa_ * (theta_ - v_pos) * dt + xi_ * sqrt_v * sqrt_dt * z_v;
                s *= std::exp((rate_ - Real(0.5) * v_pos) * dt
                              + sqrt_v * sqrt_dt * z_s);

                // Discrete crossing check.
                const bool crossed = is_upper ? (s >= barrier) : (s <= barrier);
                if (crossed) {
                    barrier_hit = true;
                    continue;
                }

                // Brownian bridge correction using local variance v_pos as
                // proxy for sigma^2 in the GBM bridge formula.
                const Real var_dt = v_pos * dt;
                const Real la = is_upper
                    ? std::log(barrier / s_prev)
                    : std::log(s_prev / barrier);
                const Real lb = is_upper
                    ? std::log(barrier / s)
                    : std::log(s / barrier);

                if (la > Real(0) && lb > Real(0) && var_dt > Real(0)) {
                    const Real p_cross = std::exp(Real(-2) * la * lb / var_dt);
                    if (uniform(rng) < p_cross) {
                        barrier_hit = true;
                    }
                }
            }

            const bool pays = is_knockout ? !barrier_hit : barrier_hit;
            terminal_values[i] = pays ? s : Real(0);
        }
    }

    Real rate() const override { return rate_; }

private:
    Real rate_;
    Real v0_;
    Real kappa_;
    Real theta_;
    Real xi_;
    Real rho_;
};
