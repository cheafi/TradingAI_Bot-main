# src/main.py
"""Main runner. Modes: demo (safe), backtest, paper, live (stub)."""
import argparse
import logging
import asyncio
from src.config import cfg
from src.utils.data import fetch_ohlcv_ccxt, synthetic_ohlcv
from src.strategies.scalping import signals, enrich
from src.utils.execution import PaperTrader
from src.utils.risk import kelly_fraction, sharpe, monte_carlo_drawdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_run(symbol: str = "BTC/USDT"):
    # get safe data
    df = fetch_ohlcv_ccxt(symbol, limit=cfg.DEFAULT_LIMIT)
    trader = PaperTrader(cfg.INITIAL_CAPITAL)
    df2 = enrich(df, cfg.EMA_PERIOD, cfg.ATR_PERIOD, cfg.KELTNER_MULT)
    long_signals = signals(df, cfg)
    # Simple loop: buy at next open and use ATR stop
    for i in range(1, len(df2)-1):
        ts = df2.index[i]
        if long_signals.iloc[i] and trader.positions.get(symbol, None) is None:
            entry = float(df2.iloc[i+1]["open"])
            atr = float(df2.iloc[i]["ATR"])
            stop = entry - cfg.K_INIT * atr
            qty = int((cfg.INITIAL_CAPITAL * cfg.KELLY_CAP) // max(stop - entry if stop != entry else 1e-6, 1e-6))
            if qty > 0:
                trader.open_long(symbol, entry, qty, stop, ts)
        # check stops
        pos = trader.positions.get(symbol)
        if pos and pos.qty > 0:
            low = float(df2.iloc[i]["low"])
            if low <= pos.stop:
                trader.close_long(symbol, pos.stop, ts, reason="stop")
    # summary
    trades = trader.trades
    returns = [t.get("pnl", 0.0) for t in trades if t.get("pnl") is not None]
    import numpy as np
    r = np.array(returns)
    med_dd, p95_dd = monte_carlo_drawdown(r, paths=500)
    logger.info("Demo finished. Trades: %d, Final capital: %.2f", len(trades), trader.capital)
    logger.info("Monte Carlo median DD: %.3f, 95pct DD: %.3f", med_dd, p95_dd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="demo", choices=["demo", "backtest", "paper", "live"])
    parser.add_argument("--symbol", default="BTC/USDT")
    args = parser.parse_args()
    if args.mode == "demo":
        demo_run(args.symbol)
    else:
        print("Mode not fully implemented in demo. Use demo mode for safe run.")

if __name__ == "__main__":
    main()
