# src/strategies/scalping.py
"""Scalping strategy with Keltner + RSI + ML-probability stub."""
from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()

def keltner_channels(df: pd.DataFrame, ema_period: int, atr_period: int, mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(df["close"], ema_period)
    a = atr(df, atr_period)
    return m, m + mult * a, m - mult * a

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    emaup = up.ewm(alpha=1/period, adjust=False).mean()
    emadown = down.ewm(alpha=1/period, adjust=False).mean()
    rs = emaup / (emadown + 1e-12)
    return 100 - (100 / (1 + rs))

def enrich(df: pd.DataFrame, ema_p: int, atr_p: int, mult: float) -> pd.DataFrame:
    df = df.copy()
    df["EMA"] = ema(df["close"], ema_p)
    df["ATR"] = atr(df, atr_p)
    m, up, low = keltner_channels(df, ema_p, atr_p, mult)
    df["KC_MID"], df["KC_UP"], df["KC_LOW"] = m, up, low
    df["RSI3"] = rsi(df["close"], 3)
    df["ATR_PCT"] = (df["ATR"] / (df["close"] + 1e-12)).clip(lower=0)
    return df

def ml_probability_stub(df: pd.DataFrame) -> pd.Series:
    """
    Placeholder probability estimator: slight tilt toward momentum.
    Replace with walk-forward RF/LSTM later.
    """
    returns = df["close"].pct_change().fillna(0)
    p = 0.5 + 0.5 * (returns.rolling(5).mean().fillna(0).clip(-0.02, 0.02) / 0.02)
    return p.clip(0.01, 0.99)

def signals(df: pd.DataFrame, cfg) -> pd.Series:
    """Return boolean LongSignal series."""
    df = enrich(df, cfg.EMA_PERIOD, cfg.ATR_PERIOD, cfg.KELTNER_MULT)
    prob = ml_probability_stub(df)
    long = (df["close"] > df["KC_UP"]) & (df["RSI3"] > 50) & (df["ATR_PCT"].between(cfg.ATR_PERIOD * 0.0001, cfg.ATR_PERIOD * 0.01))
    long &= prob >= 0.55
    return long
