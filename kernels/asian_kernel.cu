#include "asian_kernel.h"

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

static constexpr int ASIAN_BLOCK_SIZE = 256;

// Float kernel — uses curand_normal (single precision).
template <typename Real>
__global__ void asian_gbm_kernel(
    Real* __restrict__ path_averages,
    Real spot,
    Real drift,
    Real diffusion,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    Real s   = spot;
    Real sum = Real(0);

    for (std::size_t step = 0; step < num_steps; ++step) {
        const Real z = curand_normal(&state);
        s *= exp(drift + diffusion * z);
        sum += s;
    }

    path_averages[tid] = sum / static_cast<Real>(num_steps);
}

// Double specialization — uses curand_normal_double.
template <>
__global__ void asian_gbm_kernel<double>(
    double* __restrict__ path_averages,
    double spot,
    double drift,
    double diffusion,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double s   = spot;
    double sum = 0.0;

    for (std::size_t step = 0; step < num_steps; ++step) {
        const double z = curand_normal_double(&state);
        s *= exp(drift + diffusion * z);
        sum += s;
    }

    path_averages[tid] = sum / static_cast<double>(num_steps);
}

template <typename Real>
void launch_asian_gbm_kernel(
    Real* d_path_averages,
    Real spot,
    Real rate,
    Real volatility,
    Real maturity,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const Real dt        = maturity / static_cast<Real>(num_steps);
    const Real drift     = (rate - Real(0.5) * volatility * volatility) * dt;
    const Real diffusion = volatility * static_cast<Real>(sqrt(static_cast<double>(dt)));

    const int num_blocks =
        static_cast<int>((num_paths + ASIAN_BLOCK_SIZE - 1) / ASIAN_BLOCK_SIZE);

    asian_gbm_kernel<Real><<<num_blocks, ASIAN_BLOCK_SIZE>>>(
        d_path_averages, spot, drift, diffusion, num_paths, num_steps, seed);

    CUDA_CHECK(cudaGetLastError());
}

template void launch_asian_gbm_kernel<float>(
    float*, float, float, float, float,
    std::size_t, std::size_t, unsigned long long);

template void launch_asian_gbm_kernel<double>(
    double*, double, double, double, double,
    std::size_t, std::size_t, unsigned long long);
