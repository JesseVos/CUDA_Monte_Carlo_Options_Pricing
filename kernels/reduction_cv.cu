#include "reduction_cv.h"

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

/// Five-moment shared-memory tree reduction kernel.
///
/// Each thread computes one element's payoff h = max(S_T - K, 0) (or put),
/// then five arrays in shared memory are reduced in parallel:
///   s_h, s_h2, s_s, s_s2, s_hs
///
/// Shared memory layout (5 arrays of BLOCK_SIZE Reals):
///   [0 .. BS-1]       s_h
///   [BS .. 2BS-1]     s_h2
///   [2BS .. 3BS-1]    s_s
///   [3BS .. 4BS-1]    s_s2
///   [4BS .. 5BS-1]    s_hs
///
/// Total: 5 * 256 * sizeof(Real) = 10240 bytes (double) — within limits.
template <typename Real>
__global__ void payoff_cv_reduce_kernel(
    const Real* __restrict__ terminal_values,
    std::size_t num_paths,
    Real strike,
    bool is_call,
    Real* __restrict__ block_sum_h,
    Real* __restrict__ block_sum_h_sq,
    Real* __restrict__ block_sum_s,
    Real* __restrict__ block_sum_s_sq,
    Real* __restrict__ block_sum_hs)
{
    extern __shared__ char shared_mem[];
    Real* s_h  = reinterpret_cast<Real*>(shared_mem);
    Real* s_h2 = s_h  + blockDim.x;
    Real* s_s  = s_h2 + blockDim.x;
    Real* s_s2 = s_s  + blockDim.x;
    Real* s_hs = s_s2 + blockDim.x;

    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;

    Real h  = static_cast<Real>(0);
    Real s  = static_cast<Real>(0);

    if (tid < num_paths) {
        s = terminal_values[tid];
        h = is_call ? ((s > strike) ? (s - strike) : static_cast<Real>(0))
                    : ((strike > s) ? (strike - s) : static_cast<Real>(0));
    }

    s_h[lid]  = h;
    s_h2[lid] = h * h;
    s_s[lid]  = s;
    s_s2[lid] = s * s;
    s_hs[lid] = h * s;
    __syncthreads();

    // Tree reduction across the five arrays in lock-step.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            s_h[lid]  += s_h[lid  + stride];
            s_h2[lid] += s_h2[lid + stride];
            s_s[lid]  += s_s[lid  + stride];
            s_s2[lid] += s_s2[lid + stride];
            s_hs[lid] += s_hs[lid + stride];
        }
        __syncthreads();
    }

    if (lid == 0) {
        block_sum_h[blockIdx.x]    = s_h[0];
        block_sum_h_sq[blockIdx.x] = s_h2[0];
        block_sum_s[blockIdx.x]    = s_s[0];
        block_sum_s_sq[blockIdx.x] = s_s2[0];
        block_sum_hs[blockIdx.x]   = s_hs[0];
    }
}

template <typename Real>
void launch_payoff_cv_reduction(
    const Real* d_terminal_values,
    std::size_t num_paths,
    Real strike,
    bool is_call,
    CVReductionResult<Real>& result)
{
    int num_blocks = static_cast<int>((num_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);

    Real* d_sum_h    = nullptr;
    Real* d_sum_h_sq = nullptr;
    Real* d_sum_s    = nullptr;
    Real* d_sum_s_sq = nullptr;
    Real* d_sum_hs   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_sum_h,    num_blocks * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sum_h_sq, num_blocks * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sum_s,    num_blocks * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sum_s_sq, num_blocks * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_sum_hs,   num_blocks * sizeof(Real)));

    // Five arrays of BLOCK_SIZE Reals in shared memory.
    std::size_t shared_mem_size = 5 * BLOCK_SIZE * sizeof(Real);

    payoff_cv_reduce_kernel<Real><<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        d_terminal_values,
        num_paths,
        strike,
        is_call,
        d_sum_h,
        d_sum_h_sq,
        d_sum_s,
        d_sum_s_sq,
        d_sum_hs);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy per-block results to host.
    std::vector<Real> h_h(num_blocks),  h_h2(num_blocks),
                      h_s(num_blocks),  h_s2(num_blocks), h_hs(num_blocks);

    CUDA_CHECK(cudaMemcpy(h_h.data(),  d_sum_h,    num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_h2.data(), d_sum_h_sq, num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_s.data(),  d_sum_s,    num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_s2.data(), d_sum_s_sq, num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hs.data(), d_sum_hs,   num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));

    cudaFree(d_sum_h);
    cudaFree(d_sum_h_sq);
    cudaFree(d_sum_s);
    cudaFree(d_sum_s_sq);
    cudaFree(d_sum_hs);

    // Host-side final sum across blocks.
    Real tot_h = 0, tot_h2 = 0, tot_s = 0, tot_s2 = 0, tot_hs = 0;
    for (int i = 0; i < num_blocks; ++i) {
        tot_h  += h_h[i];
        tot_h2 += h_h2[i];
        tot_s  += h_s[i];
        tot_s2 += h_s2[i];
        tot_hs += h_hs[i];
    }

    result.sum_h    = tot_h;
    result.sum_h_sq = tot_h2;
    result.sum_s    = tot_s;
    result.sum_s_sq = tot_s2;
    result.sum_hs   = tot_hs;
}

// Explicit instantiations.
template void launch_payoff_cv_reduction<float>(
    const float*, std::size_t, float, bool, CVReductionResult<float>&);

template void launch_payoff_cv_reduction<double>(
    const double*, std::size_t, double, bool, CVReductionResult<double>&);
