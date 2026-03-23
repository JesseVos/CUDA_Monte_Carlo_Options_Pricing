#include "engine/cpu_engine.h"
#include "models/model.h"
#include "pricing/european.h"
#include "pricing/barrier.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

CPUEngine::CPUEngine(std::size_t num_paths, std::size_t num_steps, unsigned long seed)
    : num_paths_(num_paths), num_steps_(num_steps), seed_(seed)
{
}

PricingResult CPUEngine::price_european(
    const Model<double>& model,
    double spot,
    double strike,
    double maturity,
    OptionType option_type)
{
    // Generate terminal asset values using the model.
    std::vector<double> terminal_values;
    model.generate_paths_cpu(spot, maturity, num_paths_, num_steps_, seed_,
                             terminal_values);

    // Compute discounted payoff average and standard error.
    return ::price_european(terminal_values, strike, model.rate(), maturity,
                            option_type);
}

PricingResult CPUEngine::price_european(
    const Model<double>& model,
    double spot,
    double strike,
    double maturity,
    OptionType option_type,
    const VarianceReductionConfig& vr_config)
{
    if (vr_config.antithetic && (num_paths_ % 2 != 0)) {
        throw std::invalid_argument(
            "Antithetic variates require num_paths to be even.");
    }

    std::vector<double> terminal_values;

    if (vr_config.antithetic) {
        model.generate_paths_antithetic_cpu(spot, maturity, num_paths_,
                                            num_steps_, seed_, terminal_values);
    } else {
        model.generate_paths_cpu(spot, maturity, num_paths_, num_steps_,
                                 seed_, terminal_values);
    }

    double rate     = model.rate();
    double discount = std::exp(-rate * maturity);

    auto payoff_fn = [&](double s) -> double {
        return (option_type == OptionType::Call)
                   ? std::max(s - strike, 0.0)
                   : std::max(strike - s, 0.0);
    };

    if (vr_config.antithetic && vr_config.control_variate) {
        // Antithetic + CV: work with pair averages and apply CV correction.
        // For pair i: h_avg = (h(S^+) + h(S^-))/2, s_avg = (S^+ + S^-)/2.
        // E[s_avg] = E[S_T] = spot * exp(r*T) (same control mean, by symmetry).
        std::size_t half      = terminal_values.size() / 2;
        double effective_n    = static_cast<double>(half);
        double expected_s     = spot * std::exp(rate * maturity);

        double sum_h = 0, sum_h2 = 0, sum_s = 0, sum_s2 = 0, sum_hs = 0;
        for (std::size_t i = 0; i < half; ++i) {
            double h_avg = (payoff_fn(terminal_values[i]) +
                            payoff_fn(terminal_values[i + half])) * 0.5;
            double s_avg = (terminal_values[i] + terminal_values[i + half]) * 0.5;
            sum_h  += h_avg;
            sum_h2 += h_avg * h_avg;
            sum_s  += s_avg;
            sum_s2 += s_avg * s_avg;
            sum_hs += h_avg * s_avg;
        }

        double mean_h = sum_h / effective_n;
        double mean_s = sum_s / effective_n;
        double var_s  = (sum_s2 / effective_n) - (mean_s * mean_s);
        double cov_hs = (sum_hs / effective_n) - (mean_h * mean_s);

        if (var_s <= 0.0) {
            double var_h = std::max((sum_h2 / effective_n) - (mean_h * mean_h), 0.0);
            return {discount * mean_h, discount * std::sqrt(var_h / effective_n)};
        }

        double beta      = cov_hs / var_s;
        double mean_h_cv = mean_h - beta * (mean_s - expected_s);
        double price     = discount * mean_h_cv;

        double var_h    = (sum_h2 / effective_n) - (mean_h * mean_h);
        double var_h_cv = std::max(var_h - (cov_hs * cov_hs) / var_s, 0.0);
        double se       = discount * std::sqrt(var_h_cv / effective_n);

        return {price, se};
    }

    if (vr_config.antithetic) {
        // Antithetic-only: compute pair averages and their variance.
        // Effective N = num_paths/2 independent pair averages.
        std::size_t half    = terminal_values.size() / 2;
        double effective_n  = static_cast<double>(half);

        double sum_avg = 0, sum_avg_sq = 0;
        for (std::size_t i = 0; i < half; ++i) {
            double h_avg = (payoff_fn(terminal_values[i]) +
                            payoff_fn(terminal_values[i + half])) * 0.5;
            sum_avg    += h_avg;
            sum_avg_sq += h_avg * h_avg;
        }

        double mean_avg = sum_avg / effective_n;
        double price    = discount * mean_avg;
        double variance = std::max((sum_avg_sq / effective_n) - (mean_avg * mean_avg), 0.0);
        double se       = discount * std::sqrt(variance / effective_n);

        return {price, se};
    }

    if (vr_config.control_variate) {
        // Control variable: S_T with known E[S_T] = S_0 * exp(r*T).
        double n          = static_cast<double>(terminal_values.size());
        double expected_s = spot * std::exp(rate * maturity);

        double sum_h = 0, sum_h2 = 0, sum_s = 0, sum_s2 = 0, sum_hs = 0;
        for (std::size_t i = 0; i < terminal_values.size(); ++i) {
            double s = terminal_values[i];
            double h = payoff_fn(s);
            sum_h  += h;
            sum_h2 += h * h;
            sum_s  += s;
            sum_s2 += s * s;
            sum_hs += h * s;
        }

        double mean_h = sum_h / n;
        double mean_s = sum_s / n;
        double var_s  = (sum_s2 / n) - (mean_s * mean_s);
        double cov_hs = (sum_hs / n) - (mean_h * mean_s);

        if (var_s <= 0.0) {
            double var_h = std::max((sum_h2 / n) - (mean_h * mean_h), 0.0);
            return {discount * mean_h, discount * std::sqrt(var_h / n)};
        }

        double beta      = cov_hs / var_s;
        double mean_h_cv = mean_h - beta * (mean_s - expected_s);
        double price     = discount * mean_h_cv;

        double var_h    = (sum_h2 / n) - (mean_h * mean_h);
        double var_h_cv = std::max(var_h - (cov_hs * cov_hs) / var_s, 0.0);
        double se       = discount * std::sqrt(var_h_cv / n);

        return {price, se};
    }

    return ::price_european(terminal_values, strike, rate, maturity, option_type);
}

PricingResult CPUEngine::price_asian(
    const Model<double>& model,
    double spot,
    double strike,
    double maturity,
    OptionType option_type)
{
    std::vector<double> averages;
    model.generate_asian_paths_cpu(
        spot, maturity, num_paths_, num_steps_, seed_, averages);

    // Reuse European payoff/reduction logic — the "terminal value" here is the
    // path's arithmetic average, so max(avg - K, 0) is the Asian payoff.
    return ::price_european(averages, strike, model.rate(), maturity, option_type);
}

PricingResult CPUEngine::price_barrier(
    const Model<double>& model,
    double spot,
    double strike,
    double maturity,
    double barrier,
    BarrierType barrier_type,
    OptionType option_type)
{
    const bool is_upper    = (barrier_type == BarrierType::UpAndOut
                              || barrier_type == BarrierType::UpAndIn);
    const bool is_knockout = (barrier_type == BarrierType::UpAndOut
                              || barrier_type == BarrierType::DownAndOut);

    std::vector<double> terminal_values;
    model.generate_barrier_paths_cpu(
        spot, maturity, num_paths_, num_steps_, seed_,
        barrier, is_upper, is_knockout, terminal_values);

    // terminal_values[i] is S(T) for paying paths and 0 for non-paying paths.
    // Applying max(S-K,0) on 0 correctly gives 0 payoff.
    return ::price_european(
        terminal_values, strike, model.rate(), maturity, option_type);
}
