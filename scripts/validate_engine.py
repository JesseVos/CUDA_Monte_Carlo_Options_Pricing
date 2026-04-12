#!/usr/bin/env python3
"""
Engine validation script — runs 9 numerical checks and reports PASS/FAIL.

Usage (from repo root):
    source ~/.venvs/dev/bin/activate
    python3 scripts/validate_engine.py
"""

import sys
import os
import math

import numpy as np
from scipy.stats import norm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "build"))
import mc_engine

# ── Black-Scholes helpers ──────────────────────────────────────────────────────

def bs_call(S, K, r, sigma, T):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, sigma, T):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, r, sigma, T):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * math.sqrt(T) * norm.pdf(d1)
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm.cdf(d2))
    rho   = K * T * math.exp(-r * T) * norm.cdf(d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

def implied_vol_nr(price, S, K, r, T, sigma0=0.2, max_iter=100, tol=1e-8):
    sigma = sigma0
    for _ in range(max_iter):
        c    = bs_call(S, K, r, sigma, T)
        d1   = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * math.sqrt(T) * norm.pdf(d1)
        if vega < 1e-12:
            return None
        sigma -= (c - price) / vega
        if sigma <= 0:
            return None
        if abs(c - price) < tol:
            return sigma
    return None

# ── Test runner ────────────────────────────────────────────────────────────────

passed = 0
total  = 9

def run_test(name, fn):
    global passed
    try:
        result, detail = fn()
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {detail}")
        return result
    except Exception as e:
        print(f"  [FAIL] {name}  — exception: {e}")
        return False

# ── Common parameters ──────────────────────────────────────────────────────────
S, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0
N = 1_000_000
STEPS = 252

# ── Test 1: European call vs Black-Scholes ─────────────────────────────────────
def test_european_call():
    pricer = mc_engine.GBMPricer(
        spot=S, strike=K, rate=r, volatility=sigma, maturity=T,
        n_paths=N, n_steps=STEPS, use_gpu=True)
    res = pricer.price_european_call()
    analytical = bs_call(S, K, r, sigma, T)
    err = abs(res.price - analytical)
    nsigmas = err / res.standard_error
    ok = nsigmas < 3.0
    detail = (f"MC={res.price:.4f}, BS={analytical:.4f}, "
              f"|err|={err:.4f}, {nsigmas:.1f}σ")
    return ok, detail

# ── Test 2: European put vs Black-Scholes ──────────────────────────────────────
def test_european_put():
    pricer = mc_engine.GBMPricer(
        spot=S, strike=K, rate=r, volatility=sigma, maturity=T,
        n_paths=N, n_steps=STEPS, use_gpu=True)
    res = pricer.price_european_put()
    analytical = bs_put(S, K, r, sigma, T)
    err = abs(res.price - analytical)
    nsigmas = err / res.standard_error
    ok = nsigmas < 3.0
    detail = (f"MC={res.price:.4f}, BS={analytical:.4f}, "
              f"|err|={err:.4f}, {nsigmas:.1f}σ")
    return ok, detail

# ── Test 3: Put-call parity ────────────────────────────────────────────────────
def test_put_call_parity():
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    all_ok = True
    lines = []
    for Ki in strikes:
        pricer = mc_engine.GBMPricer(
            spot=S, strike=Ki, rate=r, volatility=sigma, maturity=T,
            n_paths=N, n_steps=STEPS, use_gpu=True)
        rc = pricer.price_european_call()
        rp = pricer.price_european_put()
        parity = S - Ki * math.exp(-r * T)
        diff   = abs(rc.price - rp.price - parity)
        tol    = 3.0 * (rc.standard_error + rp.standard_error)
        ok_i   = diff < tol
        if not ok_i:
            all_ok = False
        lines.append(f"K={Ki:.0f}: C-P={rc.price-rp.price:.4f}, "
                     f"S-Ke^(-rT)={parity:.4f}, diff={diff:.4f} "
                     f"({'OK' if ok_i else 'FAIL'})")
    for line in lines:
        print(f"         {line}")
    return all_ok, None

# ── Test 4: Greeks vs Black-Scholes analytical ─────────────────────────────────
def test_greeks():
    Katm = 100.0  # ATM for clearest results
    pricer = mc_engine.GBMPricer(
        spot=S, strike=Katm, rate=r, volatility=sigma, maturity=T,
        n_paths=2_000_000, n_steps=STEPS, use_gpu=True)
    mc_g = pricer.compute_greeks()
    bs_g = bs_greeks(S, Katm, r, sigma, T)

    greek_map = [
        ("delta", mc_g.delta,  bs_g["delta"]),
        ("gamma", mc_g.gamma,  bs_g["gamma"]),
        ("vega",  mc_g.vega,   bs_g["vega"]),
        ("theta", mc_g.theta,  bs_g["theta"]),
        ("rho",   mc_g.rho,    bs_g["rho"]),
    ]
    all_ok = True
    for name, mc_val, bs_val in greek_map:
        if abs(bs_val) > 1e-10:
            rel = abs(mc_val - bs_val) / abs(bs_val)
        else:
            rel = abs(mc_val - bs_val)
        ok_i = rel < 0.05
        if not ok_i:
            all_ok = False
        print(f"         {name:<6}: MC={mc_val:+.4f}, BS={bs_val:+.4f}, "
              f"rel err={rel:.2%} ({'OK' if ok_i else 'FAIL'})")
    return all_ok, None

# ── Test 5: Asian call ≤ European call ────────────────────────────────────────
def test_asian_le_european():
    Katm = 100.0
    p_eur = mc_engine.GBMPricer(
        spot=S, strike=Katm, rate=r, volatility=sigma, maturity=T,
        n_paths=N, n_steps=STEPS, use_gpu=True).price_european_call()
    p_asn = mc_engine.GBMPricer(
        spot=S, strike=Katm, rate=r, volatility=sigma, maturity=T,
        n_paths=N, n_steps=STEPS, use_gpu=True).price_asian_call()
    ok = p_asn.price < p_eur.price
    detail = (f"Asian={p_asn.price:.4f}, European={p_eur.price:.4f}, "
              f"diff={p_eur.price - p_asn.price:.4f}")
    return ok, detail

# ── Test 6: Barrier B=500 ≈ European ──────────────────────────────────────────
def test_barrier_convergence():
    Katm = 100.0
    p_eur = mc_engine.GBMPricer(
        spot=S, strike=Katm, rate=r, volatility=sigma, maturity=T,
        n_paths=N, n_steps=STEPS, use_gpu=True).price_european_call()
    p_bar = mc_engine.GBMPricer(
        spot=S, strike=Katm, rate=r, volatility=sigma, maturity=T,
        n_paths=N, n_steps=STEPS, use_gpu=True).price_barrier_call(
            barrier=500.0, barrier_type=mc_engine.BarrierType.UP_AND_OUT)
    diff = abs(p_bar.price - p_eur.price)
    ok = diff < 1.0
    detail = (f"Barrier(B=500)={p_bar.price:.4f}, European={p_eur.price:.4f}, "
              f"diff={diff:.4f}")
    return ok, detail

# ── Test 7: Heston prices positive with reasonable SE ─────────────────────────
def test_heston_basic():
    Katm = 100.0
    h = mc_engine.HestonPricer(
        spot=S, strike=Katm, rate=r,
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        maturity=T, n_paths=N, n_steps=STEPS, use_gpu=True)
    res = h.price_european_call()
    ok = res.price > 0.0 and res.standard_error < 0.5
    detail = f"price={res.price:.4f}, SE={res.standard_error:.5f}"
    return ok, detail

# ── Test 8: Heston smile — IV(K=80) > IV(K=120) when rho=-0.7 ─────────────────
def test_heston_smile():
    for Ki, attr in [(80.0, "iv_low"), (120.0, "iv_high")]:
        pass  # built below

    results = {}
    for Ki in [80.0, 120.0]:
        h = mc_engine.HestonPricer(
            spot=S, strike=Ki, rate=r,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            maturity=T, n_paths=N, n_steps=STEPS, use_gpu=True)
        res = h.price_european_call()
        iv = implied_vol_nr(res.price, S, Ki, r, T, sigma0=0.2)
        results[Ki] = iv

    iv_80, iv_120 = results[80.0], results[120.0]
    if iv_80 is None or iv_120 is None:
        return False, "IV solver failed"
    ok = iv_80 > iv_120
    detail = f"IV(K=80)={iv_80:.4f}, IV(K=120)={iv_120:.4f}"
    return ok, detail

# ── Test 9: Convergence — error at 1M < error at 10K ─────────────────────────
def test_convergence():
    analytical = bs_call(S, K, r, sigma, T)
    errors = {}
    for n in [10_000, 1_000_000]:
        pricer = mc_engine.GBMPricer(
            spot=S, strike=K, rate=r, volatility=sigma, maturity=T,
            n_paths=n, n_steps=STEPS, use_gpu=True)
        res = pricer.price_european_call()
        errors[n] = abs(res.price - analytical)
    ok = errors[1_000_000] < errors[10_000]
    detail = (f"|err|@10K={errors[10_000]:.4f}, "
              f"|err|@1M={errors[1_000_000]:.4f}")
    return ok, detail

# ── Run all tests ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Monte Carlo Engine Validation")
print("=" * 60)
run_test("European call vs Black-Scholes",          test_european_call)
run_test("European put vs Black-Scholes",           test_european_put)
run_test("Put-call parity (5 strikes)",             test_put_call_parity)
run_test("Greeks vs analytical (ATM, 2M paths)",   test_greeks)
run_test("Asian call ≤ European call",              test_asian_le_european)
run_test("Barrier (B=500) ≈ European call",         test_barrier_convergence)
run_test("Heston: price > 0 and SE < 0.5",         test_heston_basic)
run_test("Heston smile: IV(K=80) > IV(K=120)",     test_heston_smile)
run_test("Convergence: error@1M < error@10K",       test_convergence)
print("=" * 60)
print(f"  {passed}/{total} tests passed")
print("=" * 60)
sys.exit(0 if passed == total else 1)
