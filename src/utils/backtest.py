from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from .risk import sharpe, var_95, monte_carlo

def vectorized_backtest(
    df: pd.DataFrame,
    entries: pd.Series,
    ml_prob: pd.Series,
    atr: pd.Series,
    cfg,
    fee_pct: float = 0.0002,
    slip_pct: float = 0.0005,
    start_capital: float = 50_000.0
) -> Dict[str, float]:
    """Simplified single-entry-at-a-time backtest (long-only, ATR stops & TP)."""
    df = df.copy()
    entries = entries.fillna(False)
    P = df["close"].values
    O = df["open"].shift(-1).fillna(df["close"]).values
    H = df["high"].values
    L = df["low"].values
    ATR = atr.fillna(method="ffill").values
    prob = ml_prob.fillna(0.5).values
    n = len(df)

    capital = start_capital
    equity_curve = [capital]
    in_pos = False
    qty = 0
    entry_px = 0.0
    stop = 0.0
    entry_i = 0

    rets = []

    for i in range(n-1):
        if in_pos:
            # update trailing
            hwm = max(H[entry_i:i+1]) if i > entry_i else O[i]
            gap = cfg.k_trail * ATR[i]
            run_atr = (hwm - entry_px) / max(ATR[i], 1e-9)
            if run_atr >= cfg.r1: gap = cfg.k_fast * ATR[i]
            if run_atr >= cfg.r2: gap = cfg.k_ultra * ATR[i]
            stop = max(stop, hwm - gap)

            # check TP
            tp = entry_px + cfg.take_profit_atr_mult * ATR[i]
            exit_px = None
            reason = None
            if H[i] >= tp:
                exit_px = tp * (1 - slip_pct)
                reason = "TP"
            elif L[i] <= stop:
                exit_px = stop * (1 - slip_pct)
                reason = "SL"
            elif i - entry_i >= cfg.max_hold_bars:
                exit_px = P[i] * (1 - slip_pct)
                reason = "TIME"

            if exit_px:
                gross = (exit_px - entry_px) * qty
                fee = exit_px * qty * fee_pct
                pnl = gross - fee
                capital += (exit_px * qty) - fee  # cash returns
                in_pos = False
                qty = 0
                rets.append(pnl / max(equity_curve[-1], 1e-9))
                equity_curve.append(capital)
                continue

        if (not in_pos) and entries.iloc[i]:
            entry_px = O[i] * (1 + slip_pct)
            atrv = ATR[i]
            if atrv <= 0:
                continue
            # simple sizing
            rr = 1.5
            f_kelly = (prob[i]*rr - (1-prob[i]))/rr
            f_kelly = np.clip(f_kelly, cfg.base_risk_pct, cfg.max_risk_pct)
            risk_dollars = capital * f_kelly
            qty = int(risk_dollars // max(cfg.k_init * atrv, 1e-9))
            if qty <= 0: 
                continue
            fee = entry_px * qty * fee_pct
            capital -= entry_px * qty + fee
            stop = entry_px - cfg.k_init * atrv
            in_pos = True
            entry_i = i
            equity_curve.append(capital)
        else:
            equity_curve.append(capital)

    ret = np.array(rets, dtype=float)
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-12)
    metrics = {
        "final_capital": float(eq[-1]),
        "roi": float((eq[-1] - start_capital) / start_capital),
        "max_dd": float(dd.max() if dd.size else 0.0),
        "sharpe": sharpe(ret, rf=0.0, periods_per_year=252),
        "var95": var_95(ret),
    }
    mc_med, mc_p95 = monte_carlo(ret, paths=2000, horizon=252)
    metrics["mc_dd_med"] = mc_med
    metrics["mc_dd_95"] = mc_p95
    return metrics