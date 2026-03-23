/// pybind11 bindings for the MC options pricing engine.
///
/// Exposes GBMPricer and HestonPricer Python classes that wrap the C++
/// CPUEngine / GPUEngine internally, providing a clean 2-3 line API:
///
///   from mc_engine import GBMPricer, HestonPricer
///
///   pricer = GBMPricer(spot=100, strike=105, rate=0.05,
///                      volatility=0.2, maturity=1.0,
///                      n_paths=1_000_000, n_steps=252, use_gpu=True)
///   result = pricer.price_european_call()   # PricingResult(.price, .standard_error)
///   g      = pricer.compute_greeks()        # Greeks(.delta, .gamma, .vega, .theta, .rho)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine/cpu_engine.h"
#include "engine/gpu_engine.h"
#include "models/gbm.h"
#include "models/heston.h"
#include "pricing/european.h"
#include "pricing/barrier.h"
#include "greeks/greeks.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Thin C++ wrapper classes (not exposed to user directly, but make the
// binding code cleaner).
// ---------------------------------------------------------------------------

class GBMPricer {
public:
    GBMPricer(double spot, double strike, double rate,
              double volatility, double maturity,
              std::size_t n_paths, std::size_t n_steps,
              bool use_gpu, unsigned long long seed)
        : spot_(spot), strike_(strike), rate_(rate), vol_(volatility),
          maturity_(maturity), n_paths_(n_paths), n_steps_(n_steps),
          use_gpu_(use_gpu), seed_(seed)
    {}

    PricingResult price_european_call() const {
        return price_european(OptionType::Call);
    }
    PricingResult price_european_put() const {
        return price_european(OptionType::Put);
    }

    PricingResult price_asian_call() const { return price_asian(OptionType::Call); }
    PricingResult price_asian_put()  const { return price_asian(OptionType::Put);  }

    PricingResult price_barrier_call(double barrier, BarrierType bt) const {
        return price_barrier(barrier, bt, OptionType::Call);
    }
    PricingResult price_barrier_put(double barrier, BarrierType bt) const {
        return price_barrier(barrier, bt, OptionType::Put);
    }

    Greeks compute_greeks() const {
        if (use_gpu_) {
            return compute_greeks_gbm_gpu(
                spot_, strike_, rate_, vol_, maturity_,
                OptionType::Call, n_paths_, n_steps_, seed_);
        } else {
            return compute_greeks_gbm_cpu(
                spot_, strike_, rate_, vol_, maturity_,
                OptionType::Call, n_paths_, n_steps_,
                static_cast<unsigned long>(seed_));
        }
    }

private:
    PricingResult price_european(OptionType ot) const {
        if (use_gpu_) {
            GPUEngine eng(n_paths_, n_steps_, seed_);
            return eng.price_european_gbm(spot_, strike_, rate_, vol_, maturity_, ot);
        } else {
            GBM<double> model(rate_, vol_);
            CPUEngine eng(n_paths_, n_steps_, static_cast<unsigned long>(seed_));
            return eng.price_european(model, spot_, strike_, maturity_, ot);
        }
    }

    PricingResult price_asian(OptionType ot) const {
        if (use_gpu_) {
            GPUEngine eng(n_paths_, n_steps_, seed_);
            return eng.price_asian_gbm(spot_, strike_, rate_, vol_, maturity_, ot);
        } else {
            GBM<double> model(rate_, vol_);
            CPUEngine eng(n_paths_, n_steps_, static_cast<unsigned long>(seed_));
            return eng.price_asian(model, spot_, strike_, maturity_, ot);
        }
    }

    PricingResult price_barrier(double barrier, BarrierType bt, OptionType ot) const {
        if (use_gpu_) {
            GPUEngine eng(n_paths_, n_steps_, seed_);
            return eng.price_barrier_gbm(
                spot_, strike_, rate_, vol_, maturity_, barrier, bt, ot);
        } else {
            GBM<double> model(rate_, vol_);
            CPUEngine eng(n_paths_, n_steps_, static_cast<unsigned long>(seed_));
            return eng.price_barrier(model, spot_, strike_, maturity_, barrier, bt, ot);
        }
    }

    double spot_, strike_, rate_, vol_, maturity_;
    std::size_t n_paths_, n_steps_;
    bool use_gpu_;
    unsigned long long seed_;
};

// ---------------------------------------------------------------------------

class HestonPricer {
public:
    HestonPricer(double spot, double strike, double rate,
                 double v0, double kappa, double theta,
                 double xi, double rho, double maturity,
                 std::size_t n_paths, std::size_t n_steps,
                 bool use_gpu, unsigned long long seed)
        : spot_(spot), strike_(strike), rate_(rate),
          v0_(v0), kappa_(kappa), theta_(theta), xi_(xi), rho_(rho),
          maturity_(maturity), n_paths_(n_paths), n_steps_(n_steps),
          use_gpu_(use_gpu), seed_(seed)
    {}

    PricingResult price_european_call() const { return price_european(OptionType::Call); }
    PricingResult price_european_put()  const { return price_european(OptionType::Put);  }

    PricingResult price_asian_call() const { return price_asian(OptionType::Call); }
    PricingResult price_asian_put()  const { return price_asian(OptionType::Put);  }

    PricingResult price_barrier_call(double barrier, BarrierType bt) const {
        return price_barrier(barrier, bt, OptionType::Call);
    }
    PricingResult price_barrier_put(double barrier, BarrierType bt) const {
        return price_barrier(barrier, bt, OptionType::Put);
    }

private:
    PricingResult price_european(OptionType ot) const {
        if (use_gpu_) {
            GPUEngine eng(n_paths_, n_steps_, seed_);
            return eng.price_european_heston(
                spot_, strike_, rate_, maturity_,
                v0_, kappa_, theta_, xi_, rho_, ot);
        } else {
            Heston<double> model(rate_, v0_, kappa_, theta_, xi_, rho_);
            CPUEngine eng(n_paths_, n_steps_, static_cast<unsigned long>(seed_));
            return eng.price_european(model, spot_, strike_, maturity_, ot);
        }
    }

    PricingResult price_asian(OptionType ot) const {
        if (use_gpu_) {
            GPUEngine eng(n_paths_, n_steps_, seed_);
            return eng.price_asian_heston(
                spot_, strike_, rate_, maturity_,
                v0_, kappa_, theta_, xi_, rho_, ot);
        } else {
            Heston<double> model(rate_, v0_, kappa_, theta_, xi_, rho_);
            CPUEngine eng(n_paths_, n_steps_, static_cast<unsigned long>(seed_));
            return eng.price_asian(model, spot_, strike_, maturity_, ot);
        }
    }

    PricingResult price_barrier(double barrier, BarrierType bt, OptionType ot) const {
        if (use_gpu_) {
            GPUEngine eng(n_paths_, n_steps_, seed_);
            return eng.price_barrier_heston(
                spot_, strike_, rate_, maturity_,
                v0_, kappa_, theta_, xi_, rho_,
                barrier, bt, ot);
        } else {
            Heston<double> model(rate_, v0_, kappa_, theta_, xi_, rho_);
            CPUEngine eng(n_paths_, n_steps_, static_cast<unsigned long>(seed_));
            return eng.price_barrier(model, spot_, strike_, maturity_, barrier, bt, ot);
        }
    }

    double spot_, strike_, rate_;
    double v0_, kappa_, theta_, xi_, rho_;
    double maturity_;
    std::size_t n_paths_, n_steps_;
    bool use_gpu_;
    unsigned long long seed_;
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(mc_engine, m) {
    m.doc() = "GPU-accelerated Monte Carlo options pricing engine.\n\n"
              "Price European, Asian, and barrier options under Black-Scholes\n"
              "(GBM) and Heston stochastic volatility models, with optional\n"
              "GPU acceleration via CUDA.";

    // --- PricingResult ---
    py::class_<PricingResult>(m, "PricingResult",
        "Price estimate and Monte Carlo standard error.")
        .def_readonly("price",          &PricingResult::price,
            "Discounted expected payoff.")
        .def_readonly("standard_error", &PricingResult::standard_error,
            "Standard error of the Monte Carlo estimate.")
        .def("__repr__", [](const PricingResult& r) {
            return "PricingResult(price=" + std::to_string(r.price)
                 + ", standard_error=" + std::to_string(r.standard_error) + ")";
        });

    // --- Greeks ---
    py::class_<Greeks>(m, "Greeks", "Option risk sensitivities (finite-difference).")
        .def_readonly("delta", &Greeks::delta, "∂V/∂S")
        .def_readonly("gamma", &Greeks::gamma, "∂²V/∂S²")
        .def_readonly("vega",  &Greeks::vega,  "∂V/∂σ")
        .def_readonly("theta", &Greeks::theta, "∂V/∂t  (negative for long options)")
        .def_readonly("rho",   &Greeks::rho,   "∂V/∂r")
        .def("__repr__", [](const Greeks& g) {
            return "Greeks(delta=" + std::to_string(g.delta)
                 + ", gamma="      + std::to_string(g.gamma)
                 + ", vega="       + std::to_string(g.vega)
                 + ", theta="      + std::to_string(g.theta)
                 + ", rho="        + std::to_string(g.rho) + ")";
        });

    // --- BarrierType ---
    py::enum_<BarrierType>(m, "BarrierType",
        "Barrier option classification.")
        .value("UP_AND_OUT",  BarrierType::UpAndOut,
            "Knocked out when spot rises above barrier.")
        .value("DOWN_AND_OUT", BarrierType::DownAndOut,
            "Knocked out when spot falls below barrier.")
        .value("UP_AND_IN",   BarrierType::UpAndIn,
            "Activated when spot rises above barrier.")
        .value("DOWN_AND_IN", BarrierType::DownAndIn,
            "Activated when spot falls below barrier.")
        .export_values();

    // --- GBMPricer ---
    py::class_<GBMPricer>(m, "GBMPricer",
        "Monte Carlo pricer under Black-Scholes (GBM) dynamics.\n\n"
        "Example::\n\n"
        "    pricer = GBMPricer(spot=100, strike=105, rate=0.05,\n"
        "                       volatility=0.2, maturity=1.0,\n"
        "                       n_paths=1_000_000, n_steps=252, use_gpu=True)\n"
        "    result = pricer.price_european_call()\n"
        "    print(result.price, result.standard_error)\n")
        .def(py::init<double, double, double, double, double,
                      std::size_t, std::size_t, bool, unsigned long long>(),
             py::arg("spot"),
             py::arg("strike"),
             py::arg("rate"),
             py::arg("volatility"),
             py::arg("maturity"),
             py::arg("n_paths")  = static_cast<std::size_t>(100'000),
             py::arg("n_steps")  = static_cast<std::size_t>(252),
             py::arg("use_gpu")  = true,
             py::arg("seed")     = static_cast<unsigned long long>(42))
        .def("price_european_call", &GBMPricer::price_european_call,
             "Price a European call option.")
        .def("price_european_put", &GBMPricer::price_european_put,
             "Price a European put option.")
        .def("price_asian_call", &GBMPricer::price_asian_call,
             "Price an arithmetic-average Asian call option.")
        .def("price_asian_put", &GBMPricer::price_asian_put,
             "Price an arithmetic-average Asian put option.")
        .def("price_barrier_call", &GBMPricer::price_barrier_call,
             py::arg("barrier"),
             py::arg("barrier_type") = BarrierType::UpAndOut,
             "Price a barrier call option with Brownian bridge correction.")
        .def("price_barrier_put", &GBMPricer::price_barrier_put,
             py::arg("barrier"),
             py::arg("barrier_type") = BarrierType::UpAndOut,
             "Price a barrier put option with Brownian bridge correction.")
        .def("compute_greeks", &GBMPricer::compute_greeks,
             "Compute finite-difference Greeks (delta, gamma, vega, theta, rho)\n"
             "using common random numbers for variance reduction.");

    // --- HestonPricer ---
    py::class_<HestonPricer>(m, "HestonPricer",
        "Monte Carlo pricer under Heston stochastic volatility dynamics.\n\n"
        "Example::\n\n"
        "    heston = HestonPricer(spot=100, strike=100, rate=0.05,\n"
        "                          v0=0.04, kappa=2.0, theta=0.04,\n"
        "                          xi=0.3, rho=-0.7, maturity=1.0,\n"
        "                          n_paths=1_000_000, n_steps=252, use_gpu=True)\n"
        "    result = heston.price_european_call()\n")
        .def(py::init<double, double, double,
                      double, double, double, double, double,
                      double, std::size_t, std::size_t, bool, unsigned long long>(),
             py::arg("spot"),
             py::arg("strike"),
             py::arg("rate"),
             py::arg("v0"),
             py::arg("kappa"),
             py::arg("theta"),
             py::arg("xi"),
             py::arg("rho"),
             py::arg("maturity"),
             py::arg("n_paths")  = static_cast<std::size_t>(100'000),
             py::arg("n_steps")  = static_cast<std::size_t>(252),
             py::arg("use_gpu")  = true,
             py::arg("seed")     = static_cast<unsigned long long>(42))
        .def("price_european_call", &HestonPricer::price_european_call,
             "Price a European call option.")
        .def("price_european_put", &HestonPricer::price_european_put,
             "Price a European put option.")
        .def("price_asian_call", &HestonPricer::price_asian_call,
             "Price an arithmetic-average Asian call option.")
        .def("price_asian_put", &HestonPricer::price_asian_put,
             "Price an arithmetic-average Asian put option.")
        .def("price_barrier_call", &HestonPricer::price_barrier_call,
             py::arg("barrier"),
             py::arg("barrier_type") = BarrierType::UpAndOut,
             "Price a barrier call option.")
        .def("price_barrier_put", &HestonPricer::price_barrier_put,
             py::arg("barrier"),
             py::arg("barrier_type") = BarrierType::UpAndOut,
             "Price a barrier put option.");
}
