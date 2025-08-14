# src/main.py
from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd
from src.utils.data import fetch_ohlcv
from src.strategies.scalping import ScalpingConfig, enrich, signal
from src.utils.backtest import vectorized_backtest

@dataclass
class RunConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    start_capital: float = 50_000.0

def ml_probability_stub(df: pd.DataFrame) -> pd.Series:
    """Placeholder ML prob — plug in walk-forward RF/LSTM later."""
    # slightly pro-trend probability
    prob = 0.55 + 0.05*(df["close"].pct_change().rolling(5).mean().fillna(0.0).clip(-0.05,0.05))
    return prob.clip(0.4, 0.7)

def run_once(cfg: RunConfig):
    df = fetch_ohlcv(cfg.symbol, timeframe=cfg.timeframe, limit=1200)
    scfg = ScalpingConfig()
    edf = enrich(df, scfg)
    ml_prob = ml_probability_stub(edf)
    entries = signal(edf, ml_prob, scfg)
    metrics = vectorized_backtest(edf, entries, ml_prob, edf["atr"], scfg, start_capital=cfg.start_capital)
    logging.info(f"Run metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    run_once(RunConfig())
