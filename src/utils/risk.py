# src/utils/risk.py
"""Risk tools: Kelly, Sharpe, VaR, Monte Carlo."""
from __future__ import annotations
import numpy as np
from typing import Tuple


def kelly_fraction(p: float, b: float, cap: float = 0.01) -> float:
    """
    Kelly formula: f = (p*b - (1-p)) / b
    p: probability of win (0..1)
    b: reward/risk (e.g., 1.5)
    cap: cap fraction (safety)
    """
    if b <= 0:
        return 0.0
    f = (p * b - (1 - p)) / b
    return float(max(0.0, min(f, cap)))


def sharpe(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio. returns: per-bar returns in decimal."""
    if returns.size == 0:
        return 0.0
    mu = float(np.mean(returns)) * periods
    sigma = float(np.std(returns)) * np.sqrt(periods)
    if sigma == 0:
        return 0.0
    return (mu - rf) / sigma


def var_95(returns: np.ndarray) -> float:
    """Historical VaR(95) (loss as positive)."""
    if returns.size == 0:
        return 0.0
    return float(-np.percentile(returns, 5))


def monte_carlo_drawdown(returns: np.ndarray, paths: int = 2000, horizon: int = None) -> Tuple[float, float]:
    """
    Bootstrapped Monte Carlo drawdown:
    - returns: historical per-period returns
    - returns median and 95th percentile of max drawdown (positive fraction)
    """
    if returns.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    if horizon is None:
        horizon = len(returns)
    sims = rng.choice(returns, size=(paths, horizon), replace=True)
    equity = np.cumprod(1 + sims, axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    dd = (peaks - equity) / peaks
    max_dd = dd.max(axis=1)
    return float(np.median(max_dd)), float(np.percentile(max_dd, 95))
