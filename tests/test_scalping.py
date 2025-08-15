# tests/test_scalping.py
import pandas as pd
import numpy as np
from src.strategies.scalping import enrich, signals

def make_df(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({"open": price, "high": price+1, "low": price-1, "close": price, "volume": rng.integers(10,100, n)}, index=idx)

def test_signals_runs():
    df = make_df()
    df2 = enrich(df, 20, 14, 1.5)
    sig = signals(df, None)
    assert sig.shape[0] == df.shape[0]
