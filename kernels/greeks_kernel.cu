#include "greeks_kernel.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            std::string("CUDA error at ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
    } \
} while(0)
#endif

static constexpr int GREEKS_BLOCK_SIZE = 256;
static constexpr int NUM_SCENARIOS     = 8;

/// Batched GBM Greeks kernel.
///
/// Thread layout: global_tid = s * num_paths + p.
/// CRN: curand_init(seed, p, 0) — same for every scenario at path p.
__global__ void greeks_gbm_kernel(
    double* __restrict__ d_payoffs,
    double spot,
    double rate,
    double vol,
    double maturity,
    double strike,
    bool   is_call,
    double eps_s,
    double eps_v,
    double eps_t,
    double eps_r,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const std::size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= static_cast<std::size_t>(NUM_SCENARIOS) * num_paths) return;

    const std::size_t s = global_tid / num_paths;   // scenario index
    const std::size_t p = global_tid % num_paths;   // path index

    // CRN: initialise with path index, not global index.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, p, 0, &state);

    // Scenario-specific parameters.
    double s_spot = spot, s_vol = vol, s_rate = rate, s_mat = maturity;
    switch (s) {
        case 1: s_spot += eps_s; break;  // delta_up
        case 2: s_spot -= eps_s; break;  // delta_down
        case 3: s_vol  += eps_v; break;  // vega_up
        case 4: s_vol  -= eps_v; break;  // vega_down
        case 5: s_mat  -= eps_t; break;  // theta
        case 6: s_rate += eps_r; break;  // rho_up
        case 7: s_rate -= eps_r; break;  // rho_down
        default: break;                  // case 0: base
    }

    const double dt        = s_mat / static_cast<double>(num_steps);
    const double drift     = (s_rate - 0.5 * s_vol * s_vol) * dt;
    const double diffusion = s_vol * sqrt(dt);

    double s_price = s_spot;
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double z = curand_normal_double(&state);
        s_price *= exp(drift + diffusion * z);
    }

    const double payoff =
        is_call ? fmax(s_price - strike, 0.0) : fmax(strike - s_price, 0.0);
    const double discount = exp(-s_rate * s_mat);

    d_payoffs[global_tid] = discount * payoff;
}

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
    unsigned long long seed)
{
    const std::size_t total_threads =
        static_cast<std::size_t>(NUM_SCENARIOS) * num_paths;

    const int num_blocks =
        static_cast<int>((total_threads + GREEKS_BLOCK_SIZE - 1)
                         / GREEKS_BLOCK_SIZE);

    greeks_gbm_kernel<<<num_blocks, GREEKS_BLOCK_SIZE>>>(
        d_payoffs,
        spot, rate, volatility, maturity, strike, is_call,
        eps_spot, eps_vol, eps_time, eps_rate,
        num_paths, num_steps, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
