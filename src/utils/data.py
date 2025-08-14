from __future__ import annotations
from typing import Optional, Dict
import os, logging, time
import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except ModuleNotFoundError:
    ccxt = None  # allow tests/offline

def _synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="15min")
    price = 100 + rng.standard_normal(n).cumsum()
    high = price + np.abs(rng.normal(0, 0.5, n))
    low = price - np.abs(rng.normal(0, 0.5, n))
    open_ = price + rng.normal(0, 0.2, n)
    close = price + rng.normal(0, 0.2, n)
    vol = rng.integers(100, 1000, n)
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol}, index=ts)

def fetch_ohlcv(symbol: str, exchange_id: str = "binance", timeframe: str = "15m", limit: int = 1000) -> pd.DataFrame:
    if ccxt is None:
        logging.warning("ccxt not installed, using synthetic data.")
        return _synthetic_ohlcv(limit)
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=30)).timestamp() * 1000)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            return _synthetic_ohlcv(limit)
        cols = ["timestamp","open","high","low","close","volume"]
        df = pd.DataFrame(ohlcv, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open","high","low","close","volume"]].astype(float)
        return df.dropna()
    except Exception as e:
        logging.exception(f"fetch_ohlcv failed: {e}")
        return _synthetic_ohlcv(limit)