# src/strategies/scalping.py
from __future__ import annotations
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from src.config import Config

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    emaup = up.ewm(alpha=1/period, adjust=False).mean()
    emadown = down.ewm(alpha=1/period, adjust=False).mean()
    rs = emaup / (emadown + 1e-12)
    return 100 - (100 / (1 + rs))

def keltner_channels(df: pd.DataFrame, ema_period: int, atr_period: int, mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(df["close"], ema_period)
    rng = atr(df, atr_period)
    up = mid + mult * rng
    low = mid - mult * rng
    return mid, up, low

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Average True Range (ATR) for the DataFrame."""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

# -------------------------
# enrich: robust/backwards-compatible
# -------------------------
def enrich(df: pd.DataFrame, config: Optional[Config] = None) -> pd.DataFrame:
    """Enrich price data with technical indicators."""
    cfg = config or Config()
    result = df.copy()
    
    # Use config values with fallbacks
    result['ema'] = df['close'].ewm(span=cfg.EMA_PERIOD).mean()
    result['atr'] = calculate_atr(df, cfg.ATR_PERIOD)
    mid, up, low = keltner_channels(result, cfg.EMA_PERIOD, cfg.ATR_PERIOD, cfg.KELTNER_MULT)
    result["KC_MID"], result["KC_UP"], result["KC_LOW"] = mid, up, low
    result["RSI3"] = rsi(result["close"], 3)
    result["ATR_PCT"] = (result["ATR"] / (result["close"] + 1e-12)).clip(lower=0)
    return result

# -------------------------
# ML stub
# -------------------------
def ml_probability_stub(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].pct_change().fillna(0.0)
    rolling = ret.rolling(5).mean().fillna(0)
    denom = (rolling.abs().max() + 1e-12)
    p = 0.5 + 0.5 * (rolling / denom).clip(-0.5, 0.5)
    return p.clip(0.01, 0.99)

# -------------------------
# signals (safe)
# -------------------------
def signals(df: pd.DataFrame, config: Optional[Config] = None) -> pd.Series:
    """Generate trading signals."""
    cfg = config or Config()
    out = enrich(df, config=cfg)
    prob = ml_probability_stub(out)
    long_signal = (out["close"] > out["KC_UP"]) & (out["RSI3"] > 50) & (out["ATR_PCT"].between(0.0005, 0.03))
    long_signal = long_signal & (prob >= 0.55)
    return long_signal.reindex(out.index).fillna(False).astype(bool)
        cfg = global_cfg
    out = enrich(df, cfg=cfg)
    prob = ml_probability_stub(out)
    long_signal = (out["close"] > out["KC_UP"]) & (out["RSI3"] > 50) & (out["ATR_PCT"].between(0.0005, 0.03))
    long_signal = long_signal & (prob >= 0.55)
    return long_signal.reindex(out.index).fillna(False).astype(bool)
