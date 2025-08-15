# src/utils/risk.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass
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

@dataclass
class RiskMetrics:
    var_95: float
    cvar_95: float
    max_drawdown: float
    beta: float
    correlation_matrix: pd.DataFrame

class RiskManager:
    def __init__(self, max_position_size: float = 0.1, max_portfolio_var: float = 0.02):
        self.max_position_size = max_position_size
        self.max_portfolio_var = max_portfolio_var

    def calculate_position_size(self, capital: float, price: float, volatility: float) -> int:
        max_capital_at_risk = capital * self.max_position_size
        position_size = max_capital_at_risk / (price * volatility)
        return int(position_size)

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        var = np.percentile(returns, 100 * (1 - confidence_level))
        cvar = returns[returns <= var].mean()
        return cvar

    def stress_test(self, returns: pd.Series, shock: float = -0.10) -> float:
        """Apply a shock to the returns and calculate the resulting loss."""
        shocked_returns = returns + shock
        cumulative_return = (1 + shocked_returns).cumprod()
        final_return = cumulative_return.iloc[-1] - 1
        return final_return

    def calculate_risk_metrics(self, returns: pd.DataFrame) -> RiskMetrics:
        """Calculate risk metrics including VaR, CVaR, max drawdown, beta, and correlation matrix."""
        try:
            var_95 = np.percentile(returns, 5)
            cvar_95 = self.calculate_cvar(returns)
            cumulative = (1 + returns).cumprod()
            drawdown = 1 - cumulative / cumulative.cummax()
            beta = returns.cov() / returns.var()
            correlation_matrix = returns.corr()

            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=drawdown.max(),
                beta=beta,
                correlation_matrix=correlation_matrix
            )
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                var_95=0.0,
                cvar_95=0.0,
                max_drawdown=0.0,
                beta=0.0,
                correlation_matrix=pd.DataFrame()
            )