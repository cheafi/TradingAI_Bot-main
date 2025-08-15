# src/main.py
import argparse
import logging
import numpy as np
from src.config import cfg
from src.utils.data import fetch_ohlcv_ccxt, fetch_ohlcv_futu, synthetic_ohlcv
from src.strategies.scalping import signals, enrich
from src.utils.execution import PaperTrader
from src.utils.risk import sharpe, monte_carlo_drawdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_run(symbol: str = "BTC/USDT"):
    # prefer ccxt fetch but fallback to synthetic
    df = fetch_ohlcv_ccxt(symbol, limit=cfg.DEFAULT_LIMIT)
    if df.empty:
        df = synthetic_ohlcv(symbol, limit=cfg.DEFAULT_LIMIT)
    df = df.dropna().astype(float)
    trader = PaperTrader(cfg.INITIAL_CAPITAL)
    enriched = enrich(df, cfg)
    long_mask = signals(df, cfg)
    # Run over bars
    for i in range(1, len(enriched) - 1):
        ts = enriched.index[i]
        if long_mask.iloc[i] and (trader.positions.get(symbol, None) is None or trader.positions[symbol].qty == 0):
            entry = float(enriched.iloc[i + 1]["open"])
            atr = float(enriched.iloc[i]["ATR"])
            stop = entry - cfg.K_INIT * atr
            # conservative sizing: use Kelly fraction capped
            qty = int((cfg.INITIAL_CAPITAL * cfg.KELLY_CAP) // max(1e-8, (entry - stop)))
            if qty > 0:
                trader.open_long(symbol, entry, qty, stop, ts)
        # check stop exit
        pos = trader.positions.get(symbol)
        if pos and pos.qty > 0:
            low = float(enriched.iloc[i]["low"])
            if low <= pos.stop:
                trader.close_long(symbol, pos.stop, ts, reason="stop")
            # time-based exit
            if pos.entry_time is not None and (i - enriched.index.get_loc(pos.entry_time) >= cfg.MAX_HOLD_BARS):
                trader.close_long(symbol, float(enriched.iloc[i]["close"]), ts, reason="time")
    # compute simple metrics from trades
    trades = trader.trades
    returns = []
    for t in trades:
        if "pnl" in t and t["pnl"] is not None:
            returns.append(t["pnl"] / max(1.0, abs(t.get("price", 1.0) * t.get("qty", 1))))
    r = np.array(returns) if returns else np.array([])
    mc_med, mc_p95 = monte_carlo_drawdown(r, paths=500)
    logger.info("Demo finished. Trades: %d, Final capital: %.2f", len(trades), trader.capital)
    logger.info("MC median DD: %.3f, MC 95%% DD: %.3f", mc_med, mc_p95)
    return {"trades": trades, "final_capital": trader.capital, "mc_med_dd": mc_med, "mc_p95_dd": mc_p95}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="demo", choices=["demo", "backtest", "paper", "live"])
    parser.add_argument("--symbol", default="BTC/USDT")
    args = parser.parse_args()
    if args.mode == "demo":
        demo_run(args.symbol)
    else:
        logger.info("Only demo mode is fully implemented in this starter. Use demo for safe testing.")

if __name__ == "__main__":
    main()