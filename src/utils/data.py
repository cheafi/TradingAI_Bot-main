# src/utils/data.py
"""Market data adapters: try ccxt (live); fallback to synthetic for offline demos."""
from __future__ import annotations
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def synthetic_ohlcv(symbol: str = "X", limit: int = 1000) -> pd.DataFrame:
    """Create synthetic OHLCV data (safe demo, no API keys)."""
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    n = int(limit)
    shocks = rng.normal(loc=0, scale=0.0015, size=n)
    price = 100.0 * np.exp(np.cumsum(shocks))
    close = price
    high = close * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0008, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    volume = rng.integers(50, 500, n)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="15min")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    df = df.astype(float)
    return df


def fetch_ohlcv_ccxt(symbol: str, exchange_id: str = "binance", timeframe: str = "15m", limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV using ccxt; if ccxt missing or fetch fails, return synthetic data."""
    try:
        import ccxt  # type: ignore
    except Exception:
        logger.warning("ccxt not installed; returning synthetic data.")
        return synthetic_ohlcv(symbol, limit)
    try:
        ex_cls = getattr(ccxt, exchange_id)
        ex = ex_cls({"enableRateLimit": True})
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").astype(float)
        return df
    except Exception as exc:
        logger.exception("ccxt fetch failed: %s", exc)
        return synthetic_ohlcv(symbol, limit)
