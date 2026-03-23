#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

/// Abstract base class for stochastic models used in Monte Carlo simulation.
///
/// Each derived model implements path generation on the CPU (and later GPU).
/// The model is templated on the floating-point type to support both float
/// and double precision.
///
/// Path generation writes terminal asset values into the provided output
/// vector. For path-dependent options (Asian, barrier), models must also
/// populate the full path matrix when requested.
template <typename Real = double>
class Model {
public:
    virtual ~Model() = default;

    /// Generate terminal asset values on the CPU.
    ///
    /// @param spot        Initial asset price S(0).
    /// @param maturity    Time to expiration T (in years).
    /// @param num_paths   Number of Monte Carlo paths to simulate.
    /// @param num_steps   Number of time steps per path.
    /// @param seed        RNG seed for reproducibility.
    /// @param[out] terminal_values  Output vector resized to num_paths,
    ///                              filled with S(T) for each path.
    virtual void generate_paths_cpu(
        Real spot,
        Real maturity,
        std::size_t num_paths,
        std::size_t num_steps,
        unsigned long seed,
        std::vector<Real>& terminal_values) const = 0;

    /// Generate terminal asset values using antithetic variates on the CPU.
    ///
    /// Produces num_paths terminal values from num_paths/2 RNG draws.
    /// Each draw Z generates a positive-draw path (stored at index i) and an
    /// antithetic path using -Z (stored at index i + num_paths/2).
    ///
    /// Default implementation falls back to standard path generation (no VR).
    /// Models should override this to realise the variance reduction benefit.
    ///
    /// @pre num_paths must be even.
    virtual void generate_paths_antithetic_cpu(
        Real spot,
        Real maturity,
        std::size_t num_paths,
        std::size_t num_steps,
        unsigned long seed,
        std::vector<Real>& terminal_values) const
    {
        generate_paths_cpu(spot, maturity, num_paths, num_steps, seed,
                           terminal_values);
    }

    /// Return the risk-free rate used by the model.
    virtual Real rate() const = 0;

    /// Generate per-path arithmetic averages of S for Asian option pricing.
    ///
    /// For each path the arithmetic mean of S(t_1), ..., S(t_N) is computed
    /// over all num_steps monitoring times (S_0 excluded).
    ///
    /// @param[out] averages  Resized to num_paths; element i is the mean S
    ///                       along path i.
    virtual void generate_asian_paths_cpu(
        Real /*spot*/,
        Real /*maturity*/,
        std::size_t /*num_paths*/,
        std::size_t /*num_steps*/,
        unsigned long /*seed*/,
        std::vector<Real>& /*averages*/) const
    {
        throw std::runtime_error(
            "generate_asian_paths_cpu not implemented for this model");
    }

    /// Generate terminal asset values for barrier option pricing.
    ///
    /// At each timestep the barrier is checked directly (discrete monitoring).
    /// When both endpoints of a step lie on the safe side of the barrier, a
    /// Brownian bridge correction is applied to account for unobserved crossings
    /// between monitoring dates.
    ///
    /// @param barrier      Barrier level B.
    /// @param is_upper     True for an up-barrier (B above spot), false for
    ///                     a down-barrier (B below spot).
    /// @param is_knockout  True for knock-out (payoff = 0 on hit); false for
    ///                     knock-in (payoff = 0 unless hit).
    /// @param[out] terminal_values  S(T) for paths that pay, 0 otherwise.
    virtual void generate_barrier_paths_cpu(
        Real /*spot*/,
        Real /*maturity*/,
        std::size_t /*num_paths*/,
        std::size_t /*num_steps*/,
        unsigned long /*seed*/,
        Real /*barrier*/,
        bool /*is_upper*/,
        bool /*is_knockout*/,
        std::vector<Real>& /*terminal_values*/) const
    {
        throw std::runtime_error(
            "generate_barrier_paths_cpu not implemented for this model");
    }
};
