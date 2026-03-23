#include "barrier_kernel.h"

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

static constexpr int BARRIER_BLOCK_SIZE = 256;

// Float kernel.
template <typename Real>
__global__ void barrier_gbm_kernel(
    Real* __restrict__ terminal_values,
    Real spot,
    Real drift,
    Real diffusion,
    Real barrier,
    Real vol_sq_dt,
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

    Real s           = spot;
    bool barrier_hit = false;

    for (std::size_t step = 0; step < num_steps; ++step) {
        // Early exit for knock-out paths — payoff is already 0.
        if (is_knockout && barrier_hit) break;

        const Real s_prev = s;
        const Real z      = curand_normal(&state);
        s *= exp(drift + diffusion * z);

        // Discrete crossing check.
        const bool crossed = is_upper ? (s >= barrier) : (s <= barrier);
        if (crossed) {
            barrier_hit = true;
            continue;
        }

        // Brownian bridge correction.
        const Real la = is_upper ? log(barrier / s_prev) : log(s_prev / barrier);
        const Real lb = is_upper ? log(barrier / s)      : log(s      / barrier);

        if (la > Real(0) && lb > Real(0) && vol_sq_dt > Real(0)) {
            const Real p_cross = exp(Real(-2) * la * lb / vol_sq_dt);
            const Real u       = curand_uniform(&state);
            if (u < p_cross) {
                barrier_hit = true;
            }
        }
    }

    const bool pays        = is_knockout ? !barrier_hit : barrier_hit;
    terminal_values[tid]   = pays ? s : Real(0);
}

// Double specialization — uses curand_normal_double / curand_uniform_double.
template <>
__global__ void barrier_gbm_kernel<double>(
    double* __restrict__ terminal_values,
    double spot,
    double drift,
    double diffusion,
    double barrier,
    double vol_sq_dt,
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
    bool   barrier_hit = false;

    for (std::size_t step = 0; step < num_steps; ++step) {
        if (is_knockout && barrier_hit) break;

        const double s_prev = s;
        const double z      = curand_normal_double(&state);
        s *= exp(drift + diffusion * z);

        const bool crossed = is_upper ? (s >= barrier) : (s <= barrier);
        if (crossed) {
            barrier_hit = true;
            continue;
        }

        const double la = is_upper ? log(barrier / s_prev) : log(s_prev / barrier);
        const double lb = is_upper ? log(barrier / s)      : log(s      / barrier);

        if (la > 0.0 && lb > 0.0 && vol_sq_dt > 0.0) {
            const double p_cross = exp(-2.0 * la * lb / vol_sq_dt);
            const double u       = curand_uniform_double(&state);
            if (u < p_cross) {
                barrier_hit = true;
            }
        }
    }

    const bool pays      = is_knockout ? !barrier_hit : barrier_hit;
    terminal_values[tid] = pays ? s : 0.0;
}

template <typename Real>
void launch_barrier_gbm_kernel(
    Real* d_terminal_values,
    Real spot,
    Real rate,
    Real volatility,
    Real maturity,
    Real barrier,
    bool is_upper,
    bool is_knockout,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const Real dt        = maturity / static_cast<Real>(num_steps);
    const Real drift     = (rate - Real(0.5) * volatility * volatility) * dt;
    const Real diffusion =
        volatility * static_cast<Real>(sqrt(static_cast<double>(dt)));
    const Real vol_sq_dt = volatility * volatility * dt;

    const int num_blocks =
        static_cast<int>((num_paths + BARRIER_BLOCK_SIZE - 1) / BARRIER_BLOCK_SIZE);

    barrier_gbm_kernel<Real><<<num_blocks, BARRIER_BLOCK_SIZE>>>(
        d_terminal_values, spot, drift, diffusion, barrier, vol_sq_dt,
        is_upper, is_knockout, num_paths, num_steps, seed);

    CUDA_CHECK(cudaGetLastError());
}

template void launch_barrier_gbm_kernel<float>(
    float*, float, float, float, float, float, bool, bool,
    std::size_t, std::size_t, unsigned long long);

template void launch_barrier_gbm_kernel<double>(
    double*, double, double, double, double, double, bool, bool,
    std::size_t, std::size_t, unsigned long long);
