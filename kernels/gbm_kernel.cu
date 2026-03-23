#include "gbm_kernel.h"

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

/// Block size for all MC kernels (per AGENTS.md: 256 threads).
static constexpr int BLOCK_SIZE = 256;

/// GBM path generation kernel.
///
/// Each thread simulates one complete path using the log-Euler scheme:
///   S(t+dt) = S(t) * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
///
/// We work in log-space to accumulate the exponent, then exponentiate
/// once at the end. This is exact for GBM (no discretization error).
template <typename Real>
__global__ void gbm_kernel(
    Real* __restrict__ terminal_values,
    Real spot,
    Real drift_per_step,
    Real diffusion_per_step,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    // Initialize per-thread cuRAND state (Philox4_32_10).
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    // Accumulate in log-space.
    Real log_s = log(spot);

    for (std::size_t step = 0; step < num_steps; ++step) {
        Real z = curand_normal(&state);
        log_s += drift_per_step + diffusion_per_step * z;
    }

    terminal_values[tid] = exp(log_s);
}

// Specialization for double: curand_normal_double
template <>
__global__ void gbm_kernel<double>(
    double* __restrict__ terminal_values,
    double spot,
    double drift_per_step,
    double diffusion_per_step,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double log_s = log(spot);

    for (std::size_t step = 0; step < num_steps; ++step) {
        double z = curand_normal_double(&state);
        log_s += drift_per_step + diffusion_per_step * z;
    }

    terminal_values[tid] = exp(log_s);
}

template <typename Real>
void launch_gbm_kernel(
    Real* d_terminal_values,
    Real spot,
    Real rate,
    Real volatility,
    Real maturity,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    Real dt = maturity / static_cast<Real>(num_steps);
    Real drift_per_step = (rate - static_cast<Real>(0.5) * volatility * volatility) * dt;
    Real diffusion_per_step = volatility * sqrt(static_cast<double>(dt));

    int num_blocks = static_cast<int>((num_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gbm_kernel<Real><<<num_blocks, BLOCK_SIZE>>>(
        d_terminal_values,
        spot,
        drift_per_step,
        diffusion_per_step,
        num_paths,
        num_steps,
        seed);

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations.
template void launch_gbm_kernel<float>(
    float*, float, float, float, float,
    std::size_t, std::size_t, unsigned long long);

template void launch_gbm_kernel<double>(
    double*, double, double, double, double,
    std::size_t, std::size_t, unsigned long long);
