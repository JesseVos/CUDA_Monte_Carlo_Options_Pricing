# Contributing

## Build requirements

- CMake ≥ 3.18
- CUDA Toolkit ≥ 11.0 (tested on 12.x)
- A C++17-capable compiler (GCC 9+, Clang 10+)
- Python ≥ 3.8 with pybind11 (for the Python bindings)
- Google Test is fetched automatically via CMake FetchContent

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The build produces:
- `mc_engine_tests` — the test binary
- `mc_engine*.so` — Python extension module (importable from `build/`)

## Running the tests

```bash
cd build
./mc_engine_tests
```

All 43 tests should pass. The GPU tests require a CUDA-capable device.

## Python bindings

```python
import sys
sys.path.insert(0, "build")
import mc_engine

pricer = mc_engine.GBMPricer(spot=100, strike=105, rate=0.05,
                              volatility=0.2, maturity=1.0,
                              n_paths=1_000_000, n_steps=252, use_gpu=True)
result = pricer.price_european_call()
print(result.price, result.standard_error)
```

## Notebooks

Requires Jupyter and matplotlib:

```bash
pip install jupyter matplotlib scipy
jupyter notebook notebooks/
```

## Validation script

```bash
python3 scripts/validate_engine.py
```

Runs 9 numerical checks (Black-Scholes convergence, put-call parity, Greeks,
exotic option properties, Heston smile). All should report PASS.
