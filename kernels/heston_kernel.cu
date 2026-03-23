#include "heston_kernel.h"

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

static constexpr int BLOCK_SIZE = 256;

template <typename Real>
__global__ void heston_kernel(
    Real* __restrict__ terminal_values,
    Real spot,
    Real rate,
    Real dt,
    Real sqrt_dt,
    Real v0,
    Real kappa,
    Real theta,
    Real xi,
    Real rho,
    Real rho_bar,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    Real s = spot;
    Real v = v0;

    for (std::size_t step = 0; step < num_steps; ++step) {
        Real z_v = curand_normal(&state);
        Real z_i = curand_normal(&state);
        Real z_s = rho * z_v + rho_bar * z_i;

        Real v_pos = v > static_cast<Real>(0) ? v : static_cast<Real>(0);
        Real sqrt_v = sqrt(v_pos);

        v += kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * z_v;
        s *= exp((rate - static_cast<Real>(0.5) * v_pos) * dt + sqrt_v * sqrt_dt * z_s);
    }

    terminal_values[tid] = s;
}

template <>
__global__ void heston_kernel<double>(
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
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double s = spot;
    double v = v0;

    for (std::size_t step = 0; step < num_steps; ++step) {
        double z_v = curand_normal_double(&state);
        double z_i = curand_normal_double(&state);
        double z_s = rho * z_v + rho_bar * z_i;

        double v_pos = v > 0.0 ? v : 0.0;
        double sqrt_v = sqrt(v_pos);

        v += kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * z_v;
        s *= exp((rate - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z_s);
    }

    terminal_values[tid] = s;
}

template <typename Real>
void launch_heston_kernel(
    Real* d_terminal_values,
    Real spot,
    Real rate,
    Real maturity,
    Real v0,
    Real kappa,
    Real theta,
    Real xi,
    Real rho,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned long long seed)
{
    Real dt = maturity / static_cast<Real>(num_steps);
    Real sqrt_dt = static_cast<Real>(sqrt(static_cast<double>(dt)));
    Real rho_bar = static_cast<Real>(sqrt(static_cast<double>(
        (static_cast<Real>(1) - rho * rho) > static_cast<Real>(0)
            ? (static_cast<Real>(1) - rho * rho)
            : static_cast<Real>(0))));

    int num_blocks = static_cast<int>((num_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);
    heston_kernel<Real><<<num_blocks, BLOCK_SIZE>>>(
        d_terminal_values,
        spot,
        rate,
        dt,
        sqrt_dt,
        v0,
        kappa,
        theta,
        xi,
        rho,
        rho_bar,
        num_paths,
        num_steps,
        seed);

    CUDA_CHECK(cudaGetLastError());
}

template void launch_heston_kernel<float>(
    float*, float, float, float, float, float, float, float, float,
    std::size_t, std::size_t, unsigned long long);

template void launch_heston_kernel<double>(
    double*, double, double, double, double, double, double, double, double,
    std::size_t, std::size_t, unsigned long long);
