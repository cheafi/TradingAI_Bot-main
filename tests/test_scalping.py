# tests/test_scalping.py

import numpy as np
import pandas as pd
from src.strategies.scalping import ScalpingConfig, enrich, signal
from src.utils.backtest import vectorized_backtest

def test_strategy_pipeline_runs():
    idx = pd.date_range("2024-01-01", periods=300, freq="15min")
    df = pd.DataFrame({
        "open": np.linspace(100, 110, 300) + np.random.randn(300),
        "high": np.linspace(101, 111, 300) + np.random.randn(300),
        "low":  np.linspace( 99, 109, 300) + np.random.randn(300),
        "close":np.linspace(100, 110, 300) + np.random.randn(300),
        "volume": np.random.randint(100, 500, 300)
    }, index=idx)

    cfg = ScalpingConfig()
    edf = enrich(df, cfg)
    ml_prob = pd.Series(0.6, index=edf.index)
    entries = signal(edf, ml_prob, cfg)
    metrics = vectorized_backtest(edf, entries, ml_prob, edf["atr"], cfg, start_capital=10_000.0)
    assert "sharpe" in metrics and np.isfinite(metrics["sharpe"])
