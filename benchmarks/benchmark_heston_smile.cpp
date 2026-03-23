#include <cstdio>
#include <vector>

#include "engine/gpu_engine.h"
#include "pricing/european.h"
#include "utils/heston_analytical.h"
#include "utils/implied_vol.h"

int main() {
    constexpr double spot = 100.0;
    constexpr double rate = 0.05;
    constexpr double maturity = 1.0;

    constexpr double v0 = 0.04;
    constexpr double kappa = 2.0;
    constexpr double theta = 0.04;
    constexpr double xi = 0.3;
    constexpr double rho = -0.7;

    constexpr std::size_t num_paths = 500'000;
    constexpr std::size_t num_steps = 252;

    GPUEngine gpu_engine(num_paths, num_steps, 2026ULL);

    std::vector<double> strikes{70, 80, 90, 100, 110, 120, 130};

    std::printf("strike,mc_price,analytical_price,iv_mc,iv_analytical\n");
    for (double strike : strikes) {
        PricingResult mc = gpu_engine.price_european_heston(
            spot, strike, rate, maturity, v0, kappa, theta, xi, rho,
            OptionType::Call);

        double analytical = heston_call_price(
            spot, strike, rate, maturity, v0, kappa, theta, xi, rho);

        double iv_mc = implied_vol_call_bisection(
            mc.price, spot, strike, rate, maturity);
        double iv_analytical = implied_vol_call_bisection(
            analytical, spot, strike, rate, maturity);

        std::printf("%.6f,%.10f,%.10f,%.10f,%.10f\n",
                    strike, mc.price, analytical, iv_mc, iv_analytical);
    }

    return 0;
}
