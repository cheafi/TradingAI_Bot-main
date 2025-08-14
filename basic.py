#!/usr/bin/env python3
# coding: utf-8
"""
Basic Kelly formula demo
 - safe interactive input (or defaults)
 - shows fractional Kelly and saves Kelly vs win rate plot
"""

import math
import matplotlib.pyplot as plt

capital = 1_000_000.0
default_win_rate = 0.55
default_rr = 2.5
default_cap = 0.02

print(f"Default capital: ${capital:,.0f}")
print(f"Default max fraction cap: {default_cap*100:.2f}%\n")

def kelly(p=0.6, rr=2.0, cap=0.01):
    if rr <= 0:
        return 0.0
    f = (p * rr - (1 - p)) / rr
    return max(0.0, min(f, cap))

def prompt_user():
    try:
        print("Provide Kelly parameters (press Enter for defaults):")
        p_in = input(f"Win probability [default {default_win_rate}]: ") or str(default_win_rate)
        rr_in = input(f"Reward/risk ratio [default {default_rr}]: ") or str(default_rr)
        cap_in = input(f"Max fraction cap [default {default_cap}]: ") or str(default_cap)
        p = float(p_in)
        rr = float(rr_in)
        cap = float(cap_in)
        if not (0 < p < 1):
            print("Win probability must be (0,1). Using default.")
            p = default_win_rate
        if rr <= 0:
            print("RR must be > 0. Using default.")
            rr = default_rr
        if not (0 < cap <= 1):
            print("Cap must be in (0,1]. Using default.")
            cap = default_cap
        return p, rr, cap
    except Exception:
        print("Invalid input — using defaults.")
        return default_win_rate, default_rr, default_cap

if __name__ == "__main__":
    win_rate, reward_risk, kelly_cap = prompt_user()
    kelly_frac = kelly(win_rate, reward_risk, kelly_cap)
    trade_risk = capital * kelly_frac
    print(f"\nKelly fraction = {kelly_frac:.6f}")
    print(f"Optimal risk per trade = ${trade_risk:,.2f}")
    print("Note: Fractional Kelly (e.g., 0.5 * Kelly) reduces volatility and drawdown.\n")

    # Plot Kelly vs Win rate
    win_rates = [i / 100 for i in range(1, 100)]
    fractions = [kelly(p=w, rr=reward_risk, cap=kelly_cap) for w in win_rates]

    plt.figure(figsize=(8, 5))
    plt.plot(win_rates, fractions, label="Kelly fraction")
    plt.xlabel("Win Rate")
    plt.ylabel("Kelly Fraction")
    plt.title(f"Kelly Fraction vs Win Rate (RR={reward_risk}, Cap={kelly_cap})")
    plt.grid(True)
    plt.legend()
    fname = "kelly_vs_winrate.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")
