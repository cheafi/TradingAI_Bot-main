# src/utils/risk.py
from __future__ import annotations
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def kelly_fraction(p: float, b: float, cap: float = 0.01) -> float:
    if b <= 0:
        return 0.0
    f = (p * b - (1 - p)) / b
    return float(max(0.0, min(f, cap)))

def sharpe(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio:
    returns: array of per-period returns (decimal)
    rf: risk-free rate (decimal)
    """
    returns = np.asarray(returns)
    if returns.size == 0:
        return 0.0
    mu = np.mean(returns) * periods_per_year
    sigma = np.std(returns) * np.sqrt(periods_per_year)
    if sigma == 0:
        return 0.0
    return float((mu - rf) / sigma)

def var_95(returns: np.ndarray) -> float:
    r = np.asarray(returns)
    if r.size == 0:
        return 0.0
    return float(-np.percentile(r, 5))

def monte_carlo_drawdown(returns: np.ndarray, paths: int = 2000, horizon: int = None) -> Tuple[float, float]:
    r = np.asarray(returns)
    if r.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    n = len(r) if horizon is None else horizon
    sims = rng.choice(r, size=(paths, n), replace=True)
    equity = np.cumprod(1 + sims, axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    dd = (peaks - equity) / peaks
    max_dd = dd.max(axis=1)
    return float(np.median(max_dd)), float(np.percentile(max_dd, 95))

# GARCH volatility hook (simple stub)
def garch_volatility_forecast(returns: np.ndarray) -> float:
    """
    Very simple heuristic for volatility forecast.
    For production, use arch/GARCH libs. This function returns estimated vol (sigma).
    """
    r = np.asarray(returns)
    if r.size < 10:
        return float(np.std(r) if r.size > 0 else 0.0)
    # Exponential-weighted volatility as a proxy
    weights = np.exp(-0.05 * np.arange(r.size)[::-1])
    weights = weights / weights.sum()
    vol = np.sqrt(np.sum(weights * (r - r.mean()) ** 2))
    return float(vol)