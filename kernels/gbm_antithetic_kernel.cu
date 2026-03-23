#include "gbm_antithetic_kernel.h"

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
            std::to_string(__LINE__) + " — " + cudaGetErrorString(err)); \
    } \
} while(0)
#endif

static constexpr int BLOCK_SIZE = 256;

/// Antithetic GBM kernel (float).
///
/// Thread tid (0 .. half_paths-1) draws Z at each timestep.
/// Path A accumulates +Z draws; path B accumulates -Z draws.
/// Results are interleaved in memory as:
///   [0 .. half-1]        positive-draw paths
///   [half .. num_paths-1] antithetic paths
template <typename Real>
__global__ void gbm_antithetic_kernel(
    Real* __restrict__ terminal_values,
    Real spot,
    Real drift_per_step,
    Real diffusion_per_step,
    std::size_t half_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    Real log_s_pos = log(spot);
    Real log_s_neg = log_s_pos;

    for (std::size_t step = 0; step < num_steps; ++step) {
        Real z = curand_normal(&state);
        Real dz = diffusion_per_step * z;
        log_s_pos += drift_per_step + dz;
        log_s_neg += drift_per_step - dz;
    }

    terminal_values[tid]             = exp(log_s_pos);
    terminal_values[tid + half_paths] = exp(log_s_neg);
}

/// Double specialisation: uses curand_normal_double for full FP64 quality.
template <>
__global__ void gbm_antithetic_kernel<double>(
    double* __restrict__ terminal_values,
    double spot,
    double drift_per_step,
    double diffusion_per_step,
    std::size_t half_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double log_s_pos = log(spot);
    double log_s_neg = log_s_pos;

    for (std::size_t step = 0; step < num_steps; ++step) {
        double z = curand_normal_double(&state);
        double dz = diffusion_per_step * z;
        log_s_pos += drift_per_step + dz;
        log_s_neg += drift_per_step - dz;
    }

    terminal_values[tid]             = exp(log_s_pos);
    terminal_values[tid + half_paths] = exp(log_s_neg);
}

template <typename Real>
void launch_gbm_antithetic_kernel(
    Real* d_terminal_values,
    Real spot,
    Real rate,
    Real volatility,
    Real maturity,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    Real dt               = maturity / static_cast<Real>(num_steps);
    Real drift_per_step   = (rate - static_cast<Real>(0.5) * volatility * volatility) * dt;
    Real diffusion_per_step = volatility * static_cast<Real>(sqrt(static_cast<double>(dt)));

    std::size_t half_paths = num_paths / 2;
    int num_blocks = static_cast<int>((half_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gbm_antithetic_kernel<Real><<<num_blocks, BLOCK_SIZE>>>(
        d_terminal_values,
        spot,
        drift_per_step,
        diffusion_per_step,
        half_paths,
        num_steps,
        seed);

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations.
template void launch_gbm_antithetic_kernel<float>(
    float*, float, float, float, float,
    std::size_t, std::size_t, unsigned long long);

template void launch_gbm_antithetic_kernel<double>(
    double*, double, double, double, double,
    std::size_t, std::size_t, unsigned long long);
