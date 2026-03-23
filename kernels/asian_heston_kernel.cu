#include "asian_heston_kernel.h"

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

static constexpr int ASIAN_HESTON_BLOCK_SIZE = 256;

__global__ void asian_heston_kernel(
    double* __restrict__ path_averages,
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
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double s   = spot;
    double v   = v0;
    double sum = 0.0;

    for (std::size_t step = 0; step < num_steps; ++step) {
        const double z_v   = curand_normal_double(&state);
        const double z_i   = curand_normal_double(&state);
        const double z_s   = rho * z_v + rho_bar * z_i;
        const double v_pos = v > 0.0 ? v : 0.0;
        const double sv    = sqrt(v_pos);

        v += kappa * (theta - v_pos) * dt + xi * sv * sqrt_dt * z_v;
        s *= exp((rate - 0.5 * v_pos) * dt + sv * sqrt_dt * z_s);
        sum += s;
    }

    path_averages[tid] = sum / static_cast<double>(num_steps);
}

void launch_asian_heston_kernel(
    double* d_path_averages,
    double spot,
    double rate,
    double maturity,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    const double dt      = maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double rho_bar = sqrt(
        (1.0 - rho * rho) > 0.0 ? (1.0 - rho * rho) : 0.0);

    const int num_blocks =
        static_cast<int>((num_paths + ASIAN_HESTON_BLOCK_SIZE - 1)
                         / ASIAN_HESTON_BLOCK_SIZE);

    asian_heston_kernel<<<num_blocks, ASIAN_HESTON_BLOCK_SIZE>>>(
        d_path_averages, spot, rate, dt, sqrt_dt,
        v0, kappa, theta, xi, rho, rho_bar,
        num_paths, num_steps, seed);

    CUDA_CHECK(cudaGetLastError());
}
