# src/strategies/scalping.py
from __future__ import annotations
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from src.config import cfg as global_cfg

# --- Indicator helpers ---
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

# --- Enrichment function ---
def enrich(df: pd.DataFrame, cfg: Optional[object] = None) -> pd.DataFrame:
    """
    Add indicators. If cfg is None, use global defaults from src.config.cfg.
    """
    if cfg is None:
        cfg = global_cfg
    out = df.copy()
    out["EMA"] = ema(out["close"], cfg.EMA_PERIOD)
    out["ATR"] = atr(out, cfg.ATR_PERIOD)
    mid, up, low = keltner_channels(out, cfg.EMA_PERIOD, cfg.ATR_PERIOD, cfg.KELTNER_MULT)
    out["KC_MID"], out["KC_UP"], out["KC_LOW"] = mid, up, low
    out["RSI3"] = rsi(out["close"], cfg.RSI_FAST)
    out["ATR_PCT"] = (out["ATR"] / (out["close"] + 1e-12)).clip(lower=0)
    return out

# --- ML stub: replace with walk-forward models later ---
def ml_probability_stub(df: pd.DataFrame) -> pd.Series:
    # lightweight momentum-based probability between 0.01 and 0.99
    ret = df["close"].pct_change().fillna(0.0)
    rolling = ret.rolling(5).mean().fillna(0)
    p = 0.5 + 0.5 * (rolling / (rolling.abs().max() + 1e-12)).clip(-0.5, 0.5)
    return p.clip(0.01, 0.99)

# --- Signals (safe) ---
def signals(df: pd.DataFrame, cfg: Optional[object] = None) -> pd.Series:
    """
    Returns a boolean Series indicating long entry opportunities.
    Accepts cfg or uses global defaults. Safe for tests/UI passing None.
    """
    if cfg is None:
        cfg = global_cfg
    out = enrich(df, cfg)
    prob = ml_probability_stub(out)
    long_signal = (out["close"] > out["KC_UP"]) & (out["RSI3"] > 50) & (out["ATR_PCT"].between(0.0005, 0.03))
    long_signal = long_signal & (prob >= 0.55)
    # ensure same index and dtype
    return long_signal.reindex(out.index).fillna(False).astype(bool)