#include "reduction.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

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

/// Shared-memory tree reduction kernel.
///
/// Each thread computes the payoff for one element, then participates in
/// a block-level tree reduction in shared memory. The per-block partial
/// sum and partial sum-of-squares are written to global memory.
///
/// No atomicAdd for floating-point accumulation (per AGENTS.md).
template <typename Real>
__global__ void payoff_reduce_kernel(
    const Real* __restrict__ terminal_values,
    std::size_t num_paths,
    Real strike,
    bool is_call,
    Real* __restrict__ block_sums,
    Real* __restrict__ block_sums_sq)
{
    extern __shared__ char shared_mem[];
    Real* s_sum = reinterpret_cast<Real*>(shared_mem);
    Real* s_sum_sq = s_sum + blockDim.x;

    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_id = threadIdx.x;

    // Compute payoff for this thread's element.
    Real payoff = static_cast<Real>(0);
    if (tid < num_paths) {
        Real s_t = terminal_values[tid];
        if (is_call) {
            payoff = (s_t > strike) ? (s_t - strike) : static_cast<Real>(0);
        } else {
            payoff = (strike > s_t) ? (strike - s_t) : static_cast<Real>(0);
        }
    }

    s_sum[local_id] = payoff;
    s_sum_sq[local_id] = payoff * payoff;
    __syncthreads();

    // Tree reduction in shared memory.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            s_sum[local_id] += s_sum[local_id + stride];
            s_sum_sq[local_id] += s_sum_sq[local_id + stride];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the partial result.
    if (local_id == 0) {
        block_sums[blockIdx.x] = s_sum[0];
        block_sums_sq[blockIdx.x] = s_sum_sq[0];
    }
}

template <typename Real>
void launch_payoff_reduction(
    const Real* d_terminal_values,
    std::size_t num_paths,
    Real strike,
    bool is_call,
    ReductionResult<Real>& result)
{
    int num_blocks = static_cast<int>((num_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Allocate device memory for per-block partial sums.
    Real* d_block_sums = nullptr;
    Real* d_block_sums_sq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_block_sums_sq, num_blocks * sizeof(Real)));

    // Shared memory: two arrays of BLOCK_SIZE Reals (sum + sum_sq).
    std::size_t shared_mem_size = 2 * BLOCK_SIZE * sizeof(Real);

    payoff_reduce_kernel<Real><<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        d_terminal_values,
        num_paths,
        strike,
        is_call,
        d_block_sums,
        d_block_sums_sq);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy per-block results to host and do final reduction on CPU.
    std::vector<Real> h_block_sums(num_blocks);
    std::vector<Real> h_block_sums_sq(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_sums_sq.data(), d_block_sums_sq,
                          num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));

    cudaFree(d_block_sums);
    cudaFree(d_block_sums_sq);

    // Host-side final sum across blocks.
    Real total_sum = static_cast<Real>(0);
    Real total_sum_sq = static_cast<Real>(0);
    for (int i = 0; i < num_blocks; ++i) {
        total_sum += h_block_sums[i];
        total_sum_sq += h_block_sums_sq[i];
    }

    result.sum = total_sum;
    result.sum_sq = total_sum_sq;
}

// Explicit instantiations.
template void launch_payoff_reduction<float>(
    const float*, std::size_t, float, bool, ReductionResult<float>&);

template void launch_payoff_reduction<double>(
    const double*, std::size_t, double, bool, ReductionResult<double>&);

/// Antithetic-pair reduction kernel.
///
/// Thread tid (0 .. half_paths-1) reads two terminal values: S_T[tid] and
/// S_T[tid + half_paths], computes their payoffs, averages them, and
/// participates in a shared-memory tree reduction of (h_avg, h_avg^2).
template <typename Real>
__global__ void antithetic_payoff_reduce_kernel(
    const Real* __restrict__ terminal_values,
    std::size_t half_paths,
    Real strike,
    bool is_call,
    Real* __restrict__ block_sums,
    Real* __restrict__ block_sums_sq)
{
    extern __shared__ char shared_mem[];
    Real* s_sum    = reinterpret_cast<Real*>(shared_mem);
    Real* s_sum_sq = s_sum + blockDim.x;

    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;

    Real h_avg = static_cast<Real>(0);
    if (tid < half_paths) {
        Real s_pos = terminal_values[tid];
        Real s_neg = terminal_values[tid + half_paths];
        Real h_pos = is_call ? ((s_pos > strike) ? (s_pos - strike) : static_cast<Real>(0))
                              : ((strike > s_pos) ? (strike - s_pos) : static_cast<Real>(0));
        Real h_neg = is_call ? ((s_neg > strike) ? (s_neg - strike) : static_cast<Real>(0))
                              : ((strike > s_neg) ? (strike - s_neg) : static_cast<Real>(0));
        h_avg = (h_pos + h_neg) * static_cast<Real>(0.5);
    }

    s_sum[lid]    = h_avg;
    s_sum_sq[lid] = h_avg * h_avg;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            s_sum[lid]    += s_sum[lid + stride];
            s_sum_sq[lid] += s_sum_sq[lid + stride];
        }
        __syncthreads();
    }

    if (lid == 0) {
        block_sums[blockIdx.x]    = s_sum[0];
        block_sums_sq[blockIdx.x] = s_sum_sq[0];
    }
}

template <typename Real>
void launch_antithetic_payoff_reduction(
    const Real* d_terminal_values,
    std::size_t num_paths,
    Real strike,
    bool is_call,
    ReductionResult<Real>& result)
{
    std::size_t half_paths = num_paths / 2;
    int num_blocks = static_cast<int>((half_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);

    Real* d_block_sums    = nullptr;
    Real* d_block_sums_sq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums,    num_blocks * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_block_sums_sq, num_blocks * sizeof(Real)));

    std::size_t shared_mem_size = 2 * BLOCK_SIZE * sizeof(Real);

    antithetic_payoff_reduce_kernel<Real><<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        d_terminal_values,
        half_paths,
        strike,
        is_call,
        d_block_sums,
        d_block_sums_sq);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Real> h_block_sums(num_blocks);
    std::vector<Real> h_block_sums_sq(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_sums_sq.data(), d_block_sums_sq,
                          num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));

    cudaFree(d_block_sums);
    cudaFree(d_block_sums_sq);

    Real total_sum    = static_cast<Real>(0);
    Real total_sum_sq = static_cast<Real>(0);
    for (int i = 0; i < num_blocks; ++i) {
        total_sum    += h_block_sums[i];
        total_sum_sq += h_block_sums_sq[i];
    }

    result.sum    = total_sum;
    result.sum_sq = total_sum_sq;
}

template void launch_antithetic_payoff_reduction<float>(
    const float*, std::size_t, float, bool, ReductionResult<float>&);

template void launch_antithetic_payoff_reduction<double>(
    const double*, std::size_t, double, bool, ReductionResult<double>&);
