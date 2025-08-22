# src/utils/data.py
from __future__ import annotations
import logging
from typing import Optional
import pandas as pd
import numpy as np
import os
import requests

logger = logging.getLogger(__name__)

def synthetic_ohlcv(symbol: str = "X", limit: int = 1000) -> pd.DataFrame:
    # Ensure symbol is a string for hashing
    symbol_str = str(symbol) if not isinstance(symbol, str) else symbol
    rng = np.random.default_rng(abs(hash(symbol_str)) % (2**32))
    n = int(limit)
    shocks = rng.normal(0, 0.0015, n)
    price = 100.0 * np.exp(np.cumsum(shocks))
    close = price
    high = close * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0008, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    volume = rng.integers(50, 500, n)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

def fetch_ohlcv_ccxt(symbol: str, exchange_id: str = "binance", timeframe: str = "15m", limit: int = 1500) -> pd.DataFrame:
    try:
        import ccxt  # type: ignore
    except Exception:
        logger.warning("ccxt not available, returning synthetic data.")
        return synthetic_ohlcv(symbol, limit)
    try:
        ex_cls = getattr(ccxt, exchange_id)
        ex = ex_cls({"enableRateLimit": True})
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp").astype(float)
    except Exception as exc:
        logger.exception("ccxt fetch failed, using synthetic: %s", exc)
        return synthetic_ohlcv(symbol, limit)

# FUTU adapter stub: try to import futu, otherwise fallback
def fetch_ohlcv_futu(symbol: str, limit: int = 1500) -> pd.DataFrame:
    """
    Placeholder for Futu API adapter. If futu SDK not available, returns synthetic.
    To enable real Futu: pip install futu-api and implement auth + fetch.
    """
    try:
        import futu  # type: ignore
    except Exception:
        logger.info("futu SDK not installed, returning synthetic data.")
        return synthetic_ohlcv(symbol, limit)
    try:
        # Example: real implementation requires broker connection and subscription.
        # This is a simplified placeholder.
        # TODO: implement real futu client using futu.OpenQuoteContext / OpenSecTradeContext
        return synthetic_ohlcv(symbol, limit)
    except Exception as exc:
        logger.exception("Futu fetch failed: %s", exc)
        return synthetic_ohlcv(symbol, limit)

def fetch_ohlcv_iexcloud(symbol: str, api_key: str, period: str = "1m") -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from IEX Cloud."""
    try:
        url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/{period}?token={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        else:
            logger.error(f"IEX Cloud request failed with status {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching data from IEX Cloud: {e}")
        return None