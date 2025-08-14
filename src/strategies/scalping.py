from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..utils.risk import kelly_fraction

@dataclass
class ScalpingConfig:
    ema_period: int = 20
    atr_period: int = 14
    keltner_mult: float = 1.5
    rsi_fast: int = 3
    ml_min_prob: float = 0.60
    k_init: float = 1.0
    take_profit_atr_mult: float = 1.5
    p_be: float = 0.6
    k_trail: float = 0.8
    k_fast: float = 0.6
    r1: float = 1.0
    k_ultra: float = 0.4
    r2: float = 2.0
    max_hold_bars: int = 24
    base_risk_pct: float = 0.005
    max_risk_pct: float = 0.01
    kelly_cap: float = 0.01

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ema_u = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    ema_d = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = ema_u / (ema_d + 1e-12)
    return 100 - (100 / (1 + rs))

def enrich(df: pd.DataFrame, cfg: ScalpingConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema"] = _ema(out["close"], cfg.ema_period)
    out["atr"] = _atr(out, cfg.atr_period)
    out["kc_mid"] = out["ema"]
    out["kc_upper"] = out["ema"] + cfg.keltner_mult * out["atr"]
    out["kc_lower"] = out["ema"] - cfg.keltner_mult * out["atr"]
    out["kc_width"] = (out["kc_upper"] - out["kc_lower"]) / (out["close"] + 1e-12)
    out["rsi3"] = _rsi(out["close"], cfg.rsi_fast)
    return out

def signal(df: pd.DataFrame, ml_prob: pd.Series, cfg: ScalpingConfig) -> pd.Series:
    cross_up = (df["close"] > df["kc_upper"]) & (df["close"].shift(1) <= df["kc_upper"].shift(1))
    rsi_slope = (df["rsi3"] > 50) & (df["rsi3"] > df["rsi3"].shift(1))
    ml_ok = ml_prob.fillna(0.5) >= cfg.ml_min_prob
    return cross_up & rsi_slope & ml_ok

def position_size(equity: float, entry: float, atr_val: float, prob: float, cfg: ScalpingConfig) -> int:
    if entry <= 0 or atr_val <= 0:
        return 0
    rr = 1.5
    f_kelly = kelly_fraction(prob, rr, cfg.kelly_cap)
    risk_pct = float(np.clip(f_kelly, cfg.base_risk_pct, cfg.max_risk_pct))
    risk_dollars = equity * risk_pct
    stop_distance = cfg.k_init * atr_val
    return max(0, int(risk_dollars // max(stop_distance, 1e-9)))