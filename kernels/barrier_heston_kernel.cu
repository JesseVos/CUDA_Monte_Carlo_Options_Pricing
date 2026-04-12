#include "barrier_heston_kernel.h"

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

static constexpr int BARRIER_HESTON_BLOCK_SIZE = 256;

__global__ void barrier_heston_kernel(
    double* __restrict__ terminal_values,
    double spot,
    double rate,
    double dt,
    double sqrt_dt,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double rho_bar,
    double barrier,
    bool is_upper,
    bool is_knockout,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double s           = spot;
    double v           = v0;
    bool   barrier_hit = false;

    for (std::size_t step = 0; step < num_steps; ++step) {
        if (is_knockout && barrier_hit) break;

        const double s_prev  = s;
        const double z_v     = curand_normal_double(&state);
        const double z_i     = curand_normal_double(&state);
        const double z_s     = rho * z_v + rho_bar * z_i;
        const double v_pos   = v > 0.0 ? v : 0.0;
        const double sv      = sqrt(v_pos);

        v += kappa * (theta - v_pos) * dt + xi * sv * sqrt_dt * z_v;
        s *= exp((rate - 0.5 * v_pos) * dt + sv * sqrt_dt * z_s);

        // Discrete crossing check.
        const bool crossed = is_upper ? (s >= barrier) : (s <= barrier);
        if (crossed) {
            barrier_hit = true;
            continue;
        }

        // Brownian bridge correction, v_pos used as local variance proxy.
        const double var_dt = v_pos * dt;
        const double la     = is_upper ? log(barrier / s_prev) : log(s_prev / barrier);
        const double lb     = is_upper ? log(barrier / s)      : log(s      / barrier);

        if (la > 0.0 && lb > 0.0 && var_dt > 0.0) {
            const double p_cross = exp(-2.0 * la * lb / var_dt);
            const double u       = curand_uniform_double(&state);
            if (u < p_cross) {
                barrier_hit = true;
            }
        }
    }

    const bool pays      = is_knockout ? !barrier_hit : barrier_hit;
    terminal_values[tid] = pays ? s : 0.0;
}

void launch_barrier_heston_kernel(
    double* d_terminal_values,
    double spot,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double barrier,
    bool is_upper,
    bool is_knockout,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const double dt      = maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double rho_bar = sqrt(
        (1.0 - rho * rho) > 0.0 ? (1.0 - rho * rho) : 0.0);

    const int num_blocks =
        static_cast<int>((num_paths + BARRIER_HESTON_BLOCK_SIZE - 1)
                         / BARRIER_HESTON_BLOCK_SIZE);

    barrier_heston_kernel<<<num_blocks, BARRIER_HESTON_BLOCK_SIZE>>>(
        d_terminal_values, spot, rate, dt, sqrt_dt,
        v0, kappa, theta, xi, rho, rho_bar,
        barrier, is_upper, is_knockout,
        num_paths, num_steps, seed);

    CUDA_CHECK(cudaGetLastError());
}
