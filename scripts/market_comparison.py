#!/usr/bin/env python3
"""
AEX market options vs Monte Carlo pricing comparison.

Loads AEX June 2026 European call option data, inverts to implied vols,
prices under GBM (flat vol) and Heston SV, and compares.

Usage (from repo root):
    source ~/.venvs/dev/bin/activate
    python3 scripts/market_comparison.py
"""

import sys
import os
import math
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm

# ── Python bindings ────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "build"))
import mc_engine

# ── Load market data ───────────────────────────────────────────────────────────
spot_path = os.path.join(REPO_ROOT, "data", "aex_spot.txt")
spot_data = {}
with open(spot_path) as f:
    for line in f:
        line = line.strip()
        if line:
            k, v = line.split("=", 1)
            spot_data[k.strip()] = v.strip()

SPOT      = float(spot_data["spot"])
RATE      = float(spot_data["rate"])
SNAP_DATE = datetime.date.fromisoformat(spot_data["date"])

csv_path = os.path.join(REPO_ROOT, "data", "aex_options.csv")
df = pd.read_csv(csv_path)
df["expiry_dt"]  = pd.to_datetime(df["expiry"]).dt.date
df["T"]          = df["expiry_dt"].apply(lambda e: (e - SNAP_DATE).days / 365.0)
df["call_mid"]   = (df["call_bid"] + df["call_ask"]) / 2.0
df["call_spread"] = df["call_ask"] - df["call_bid"]

# Filter: drop zero-bid rows and wide-spread rows
before = len(df)
df = df[df["call_bid"] > 0]
df = df[df["call_spread"] / df["call_mid"] <= 0.50]
df = df.reset_index(drop=True)
print(f"Loaded {before} option rows, {len(df)} passed quality filter.")

T        = float(df["T"].iloc[0])          # all rows share the same expiry
N_STEPS  = 252                              # fixed, consistent with all other engine calls
N_PATHS  = 1_000_000

print(f"Spot={SPOT}, Rate={RATE:.1%}, Snapshot={SNAP_DATE}, "
      f"Expiry={df['expiry'].iloc[0]}, T={T:.4f} yr, n_steps={N_STEPS}")

# ── Black-Scholes helpers ──────────────────────────────────────────────────────
def _d1d2(S, K, r, sigma, T):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)

def bs_call(S, K, r, sigma, T):
    d1, d2 = _d1d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def bs_vega(S, K, r, sigma, T):
    d1, _ = _d1d2(S, K, r, sigma, T)
    return S * math.sqrt(T) * norm.pdf(d1)

def implied_vol_nr(market_price, S, K, r, T,
                   sigma0=0.2, max_iter=100, tol=1e-8):
    """Newton-Raphson implied vol inversion.  Returns None on failure."""
    sigma = sigma0
    for _ in range(max_iter):
        try:
            price = bs_call(S, K, r, sigma, T)
            vega  = bs_vega(S, K, r, sigma, T)
        except (ValueError, ZeroDivisionError):
            return None
        if abs(vega) < 1e-12:
            return None
        sigma -= (price - market_price) / vega
        if sigma <= 1e-6:
            return None
        if abs(price - market_price) < tol:
            return sigma
    return None   # did not converge

# ── Step 1: Market implied vols ────────────────────────────────────────────────
market_ivs = []
valid_mask = []
for _, row in df.iterrows():
    iv = implied_vol_nr(row["call_mid"], SPOT, row["strike"], RATE, T)
    if iv is None:
        print(f"  Warning: IV solver failed for K={row['strike']:.0f}, skipping.")
    market_ivs.append(iv)
    valid_mask.append(iv is not None)

df["market_iv"] = market_ivs
df = df[valid_mask].reset_index(drop=True)
print(f"{len(df)} strikes with valid market IVs: "
      f"{df['strike'].min():.0f}–{df['strike'].max():.0f}")

# ── Step 2: ATM implied vol ────────────────────────────────────────────────────
atm_idx = (df["strike"] - SPOT).abs().idxmin()
ATM_IV  = df.loc[atm_idx, "market_iv"]
print(f"ATM strike={df.loc[atm_idx,'strike']:.0f}, ATM IV={ATM_IV:.2%}")

# ── Step 3: Monte Carlo pricing loop ──────────────────────────────────────────
print(f"\nPricing {len(df)} strikes with MC ({N_PATHS:,} paths, {N_STEPS} steps)...")

gbm_prices    = []
heston_prices = []

for i, row in df.iterrows():
    K = float(row["strike"])

    # GBM: flat vol = ATM IV
    gbm = mc_engine.GBMPricer(
        spot=SPOT, strike=K, rate=RATE, volatility=ATM_IV,
        maturity=T, n_paths=N_PATHS, n_steps=N_STEPS, use_gpu=True
    )
    gbm_prices.append(gbm.price_european_call().price)

    # Heston: v0 = theta = ATM_IV², kappa=2, xi=0.3, rho=-0.7
    v0 = ATM_IV ** 2
    heston = mc_engine.HestonPricer(
        spot=SPOT, strike=K, rate=RATE,
        v0=v0, kappa=2.0, theta=v0, xi=0.3, rho=-0.7,
        maturity=T, n_paths=N_PATHS, n_steps=N_STEPS, use_gpu=True
    )
    heston_prices.append(heston.price_european_call().price)

    print(f"  K={K:.0f}: mkt={row['call_mid']:.3f}  "
          f"GBM={gbm_prices[-1]:.3f}  Heston={heston_prices[-1]:.3f}")

df["gbm_price"]    = gbm_prices
df["heston_price"] = heston_prices

# ── Step 4: Invert MC prices to IVs ───────────────────────────────────────────
gbm_ivs    = []
heston_ivs = []
for _, row in df.iterrows():
    K = float(row["strike"])
    gbm_iv = implied_vol_nr(row["gbm_price"], SPOT, K, RATE, T,
                            sigma0=ATM_IV)
    heston_iv = implied_vol_nr(row["heston_price"], SPOT, K, RATE, T,
                               sigma0=ATM_IV)
    if gbm_iv is None:
        print(f"  Warning: GBM IV inversion failed for K={K:.0f}")
    if heston_iv is None:
        print(f"  Warning: Heston IV inversion failed for K={K:.0f}")
    gbm_ivs.append(gbm_iv)
    heston_ivs.append(heston_iv)

df["gbm_iv"]    = gbm_ivs
df["heston_iv"] = heston_ivs

# ── Step 5: Summary table ─────────────────────────────────────────────────────
print()
header = (f"{'Strike':>7} | {'Mkt Price':>9} | {'GBM Price':>9} | "
          f"{'Heston Price':>12} | {'Mkt IV':>7} | {'GBM IV':>7} | {'Heston IV':>9}")
sep = "-" * len(header)
print(sep)
print(header)
print(sep)
for _, row in df.iterrows():
    mkt_iv_s = f"{row['market_iv']:.4f}" if row['market_iv'] else " N/A  "
    gbm_iv_s = f"{row['gbm_iv']:.4f}"   if row['gbm_iv']    else " N/A  "
    hes_iv_s = f"{row['heston_iv']:.4f}" if row['heston_iv'] else " N/A  "
    print(f"{row['strike']:>7.0f} | {row['call_mid']:>9.2f} | "
          f"{row['gbm_price']:>9.2f} | {row['heston_price']:>12.2f} | "
          f"{mkt_iv_s:>7} | {gbm_iv_s:>7} | {hes_iv_s:>9}")
print(sep)

# ── Step 6: Plot 1 — Volatility Smile ─────────────────────────────────────────
fig_dir = os.path.join(REPO_ROOT, "benchmarks", "figures")
os.makedirs(fig_dir, exist_ok=True)

moneyness = df["strike"] / SPOT

fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(moneyness, df["market_iv"] * 100,
         "bo-", linewidth=1.5, markersize=5, label="AEX Market")
ax1.plot(moneyness, df["gbm_iv"] * 100,
         "--", color="gray", linewidth=1.5, label="Black-Scholes (flat vol)")
ax1.plot(moneyness, df["heston_iv"] * 100,
         "r-", linewidth=1.5, label="Heston SV")

ax1.set_xlabel("Moneyness  (K / S)", fontsize=12)
ax1.set_ylabel("Implied Volatility (%)", fontsize=12)
ax1.set_title("AEX Volatility Smile: Market vs Models\n"
              f"Snapshot {SNAP_DATE}, Expiry {df['expiry'].iloc[0]}, "
              f"S={SPOT:.0f}, r={RATE:.1%}",
              fontsize=11)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.35)
fig1.tight_layout()
smile_path = os.path.join(fig_dir, "volatility_smile.png")
fig1.savefig(smile_path, dpi=150)
print(f"\nSaved {smile_path}")

# ── Step 7: Plot 2 — Price Comparison ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
strikes = df["strike"]
ax2.plot(strikes, df["call_mid"],
         "bo-", linewidth=1.5, markersize=5, label="AEX Market Mid")
ax2.plot(strikes, df["gbm_price"],
         "--", color="gray", linewidth=1.5, label="GBM MC (flat vol)")
ax2.plot(strikes, df["heston_price"],
         "r-", linewidth=1.5, label="Heston SV MC")

ax2.set_xlabel("Strike (EUR)", fontsize=12)
ax2.set_ylabel("Call Price (EUR)", fontsize=12)
ax2.set_title("AEX Option Prices: Market vs Monte Carlo\n"
              f"1M paths, {N_STEPS} steps, T={T:.3f} yr",
              fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.35)
fig2.tight_layout()
price_path = os.path.join(fig_dir, "price_comparison.png")
fig2.savefig(price_path, dpi=150)
print(f"Saved {price_path}")
