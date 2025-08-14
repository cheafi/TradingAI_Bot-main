from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

def kelly_fraction(p: float, b: float, cap: float = 0.02) -> float:
    """Kelly f = (p*b - (1-p))/b, capped to avoid overbetting."""
    f = (p * b - (1 - p)) / max(b, 1e-9)
    return float(np.clip(f, 0.0, cap))

def var_95(returns: np.ndarray) -> float:
    """Historical VaR at 95% (positive number = loss magnitude)."""
    if returns.size == 0:
        return 0.0
    return float(np.percentile(-returns, 95))

def monte_carlo(returns: np.ndarray, paths: int = 2000, horizon: int = 252) -> Tuple[float, float]:
    """Simulate equity paths; return (max_drawdown_median, max_drawdown_95pct)."""
    if returns.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(123)
    rets = rng.choice(returns, size=(paths, horizon), replace=True)
    equity = 1.0 * np.cumprod(1.0 + rets, axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    dd = 1.0 - equity / peak
    return float(np.median(dd.max(axis=1))), float(np.percentile(dd.max(axis=1), 95))

def sharpe(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    mu = np.mean(returns) * periods_per_year
    sd = np.std(returns) * np.sqrt(periods_per_year)
    return float(0.0 if sd == 0 else (mu - rf) / sd)