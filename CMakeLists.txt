cmake_minimum_required(VERSION 3.18)
project(mc_options_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GPU architectures: 75 (Turing/T500 local), 80/86/90 (Ampere/Hopper remote)
# Note: compute_70 dropped in CUDA 13.x
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)

# Compiler warnings: must compile cleanly with -Wall -Wextra -Wpedantic
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)

# --------------------------------------------------------------------------
# Core library — CPU sources
# --------------------------------------------------------------------------
add_library(mc_engine STATIC
    src/models/gbm.cpp
    src/models/heston.cpp
    src/engine/cpu_engine.cpp
    src/pricing/european.cpp
    src/utils/black_scholes.cpp
    src/utils/heston_analytical.cpp
    src/utils/implied_vol.cpp
    kernels/gbm_kernel.cu
    kernels/heston_kernel.cu
    kernels/gbm_antithetic_kernel.cu
    kernels/asian_kernel.cu
    kernels/asian_heston_kernel.cu
    kernels/barrier_kernel.cu
    kernels/barrier_heston_kernel.cu
    kernels/greeks_kernel.cu
    kernels/reduction.cu
    kernels/reduction_cv.cu
    src/engine/gpu_engine.cu
    src/greeks/greeks.cpp
    src/greeks/greeks_gpu.cu
)

target_include_directories(mc_engine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)

# Allow .cu files to find project headers via the same include paths.
# POSITION_INDEPENDENT_CODE is required when mc_engine is linked into the
# pybind11 shared module (.so).
set_target_properties(mc_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# --------------------------------------------------------------------------
# Google Test via FetchContent
# --------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
add_executable(test_european tests/test_european.cpp)
target_link_libraries(test_european PRIVATE mc_engine GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_european)

add_executable(test_gpu tests/test_gpu.cpp)
target_link_libraries(test_gpu PRIVATE mc_engine GTest::gtest_main)
gtest_discover_tests(test_gpu)

add_executable(test_variance_reduction tests/test_variance_reduction.cpp)
target_link_libraries(test_variance_reduction PRIVATE mc_engine GTest::gtest_main)
gtest_discover_tests(test_variance_reduction)

add_executable(test_heston tests/test_heston.cpp)
target_link_libraries(test_heston PRIVATE mc_engine GTest::gtest_main)
gtest_discover_tests(test_heston)

add_executable(test_asian tests/test_asian.cpp)
target_link_libraries(test_asian PRIVATE mc_engine GTest::gtest_main)
gtest_discover_tests(test_asian)

add_executable(test_barrier tests/test_barrier.cpp)
target_link_libraries(test_barrier PRIVATE mc_engine GTest::gtest_main)
gtest_discover_tests(test_barrier)

add_executable(test_greeks tests/test_greeks.cpp)
target_link_libraries(test_greeks PRIVATE mc_engine GTest::gtest_main)
gtest_discover_tests(test_greeks)

# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
add_executable(benchmark_cpu_vs_gpu benchmarks/benchmark_cpu_vs_gpu.cpp)
target_link_libraries(benchmark_cpu_vs_gpu PRIVATE mc_engine)

add_executable(benchmark_convergence benchmarks/benchmark_convergence.cpp)
target_link_libraries(benchmark_convergence PRIVATE mc_engine)

add_executable(benchmark_heston_smile benchmarks/benchmark_heston_smile.cpp)
target_link_libraries(benchmark_heston_smile PRIVATE mc_engine)

add_executable(benchmark_all_options benchmarks/benchmark_all_options.cpp)
target_link_libraries(benchmark_all_options PRIVATE mc_engine)

# --------------------------------------------------------------------------
# Python bindings (pybind11)
# --------------------------------------------------------------------------
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_cmake_result
)
if(NOT pybind11_cmake_result EQUAL 0)
    message(WARNING "pybind11 not found via Python — skipping mc_engine Python module.")
else()
    find_package(pybind11 REQUIRED)
    pybind11_add_module(mc_engine_py python/bindings.cpp)
    target_link_libraries(mc_engine_py PRIVATE mc_engine)
    set_target_properties(mc_engine_py PROPERTIES OUTPUT_NAME mc_engine)
endif()
