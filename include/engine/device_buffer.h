#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

/// CUDA error checking macro. Throws std::runtime_error on failure.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            std::string("CUDA error at ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + " — " + cudaGetErrorString(err)); \
    } \
} while(0)

/// RAII wrapper for CUDA device memory.
///
/// Owns a device-side allocation of T[count] elements. Move-only.
/// Automatically frees memory in destructor. Provides host<->device
/// copy helpers.
template <typename T>
class DeviceBuffer {
public:
    /// Allocate device memory for @p count elements.
    explicit DeviceBuffer(std::size_t count)
        : ptr_(nullptr), count_(count)
    {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);  // No throw in destructor.
        }
    }

    // Move construction.
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    // Move assignment.
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // Non-copyable.
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /// Raw device pointer.
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    /// Number of elements.
    std::size_t size() const { return count_; }

    /// Size in bytes.
    std::size_t bytes() const { return count_ * sizeof(T); }

    /// Copy from host vector to device. Vector size must match count_.
    void copy_from_host(const std::vector<T>& host_data) {
        if (host_data.size() != count_) {
            throw std::invalid_argument("Host vector size does not match device buffer size");
        }
        CUDA_CHECK(cudaMemcpy(ptr_, host_data.data(), bytes(), cudaMemcpyHostToDevice));
    }

    /// Copy from device to host vector. Resizes vector to count_.
    void copy_to_host(std::vector<T>& host_data) const {
        host_data.resize(count_);
        CUDA_CHECK(cudaMemcpy(host_data.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
    }

    /// Copy from raw host pointer (caller ensures at least count_ elements).
    void copy_from_host(const T* host_ptr) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, bytes(), cudaMemcpyHostToDevice));
    }

    /// Copy to raw host pointer (caller ensures at least count_ elements).
    void copy_to_host(T* host_ptr) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, bytes(), cudaMemcpyDeviceToHost));
    }

    /// Zero the device memory.
    void zero() {
        CUDA_CHECK(cudaMemset(ptr_, 0, bytes()));
    }

private:
    T* ptr_;
    std::size_t count_;
};
