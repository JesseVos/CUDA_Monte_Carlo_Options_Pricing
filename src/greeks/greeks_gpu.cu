#include "greeks/greeks.h"
#include "engine/device_buffer.h"

#include "../../kernels/greeks_kernel.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)
#endif

static constexpr int NUM_SCENARIOS = 8;

Greeks compute_greeks_gbm_gpu(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    OptionType option_type,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const double eps_s = 0.01 * spot;
    const double eps_v = 0.001;
    const double eps_t = 1.0 / 365.0;
    const double eps_r = 0.0001;

    const bool is_call = (option_type == OptionType::Call);

    // Allocate device output: 8 * num_paths discounted payoffs.
    const std::size_t total = static_cast<std::size_t>(NUM_SCENARIOS) * num_paths;
    DeviceBuffer<double> d_payoffs(total);

    launch_greeks_gbm_kernel(
        d_payoffs.data(),
        spot, rate, volatility, maturity, strike, is_call,
        eps_s, eps_v, eps_t, eps_r,
        num_paths, num_steps, seed);

    // Copy to host and average each scenario's payoffs.
    std::vector<double> h_payoffs(total);
    CUDA_CHECK(cudaMemcpy(h_payoffs.data(), d_payoffs.data(),
                          total * sizeof(double), cudaMemcpyDeviceToHost));

    const double n = static_cast<double>(num_paths);
    double v[NUM_SCENARIOS];
    for (int s = 0; s < NUM_SCENARIOS; ++s) {
        double sum = 0.0;
        const double* begin = h_payoffs.data() + static_cast<std::size_t>(s) * num_paths;
        for (std::size_t p = 0; p < num_paths; ++p) {
            sum += begin[p];
        }
        v[s] = sum / n;
    }

    // v[0]=base, v[1]=S+ε, v[2]=S-ε, v[3]=σ+ε, v[4]=σ-ε,
    // v[5]=T-εt, v[6]=r+εr, v[7]=r-εr
    Greeks g;
    g.delta = (v[1] - v[2]) / (2.0 * eps_s);
    g.gamma = (v[1] - 2.0 * v[0] + v[2]) / (eps_s * eps_s);
    g.vega  = (v[3] - v[4]) / (2.0 * eps_v);
    g.theta = (v[5] - v[0]) / eps_t;
    g.rho   = (v[6] - v[7]) / (2.0 * eps_r);

    return g;
}
