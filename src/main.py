import os
import sys
import argparse
import logging
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.data import synthetic_ohlcv, fetch_ohlcv_ccxt
from src.strategies.scalping import enrich, signals
from src.utils.risk import sharpe, max_drawdown
from src.config import cfg


@dataclass
class RunConfig:
    mode: str = "demo"
    symbol: str = "BTC/USDT"
    interval: str = "1m"
    limit: int = 1200
    dry_run: bool = True


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("main")


def demo_run(run_cfg: RunConfig):
    """Demo run function that returns a dictionary with results"""
    symbol = run_cfg.symbol
    limit = run_cfg.limit
    try:
        df = synthetic_ohlcv(symbol, limit=limit)
        df.columns = [c.lower() for c in df.columns]
        # Use global cfg for indicators since RunConfig doesn't have EMA_PERIOD
        enriched = enrich(df, cfg=cfg)
        sig = signals(enriched, cfg=cfg)
        ret = enriched["close"].pct_change().dropna()
        eq = (1+ret).cumprod()
        sharpe_ratio = sharpe(ret)
        mdd = max_drawdown(eq)
        
        log.info("Demo completed: %s rows, Sharpe=%.2f, MDD=%.2f%%",
                 len(enriched), sharpe_ratio, mdd*100)
        
        return {
            "status": "ok",
            "symbol": symbol,
            "mode": run_cfg.mode,
            "rows": len(enriched),
            "sharpe": sharpe_ratio,
            "max_drawdown": mdd
        }
    except Exception as e:
        log.error("Demo run failed: %s", e)
        return {
            "status": "error",
            "symbol": symbol,
            "mode": run_cfg.mode,
            "error": str(e)
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "ccxt"], default="demo")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--limit", type=int, default=1200)
    args = p.parse_args()

    run_cfg = RunConfig(
        mode=args.mode,
        symbol=args.symbol,
        limit=args.limit
    )

    if args.mode == "demo":
        result = demo_run(run_cfg)
        log.info("Demo result: %s", result)
    else:
        df = fetch_ohlcv_ccxt(args.symbol, limit=args.limit)
        df.columns = [c.lower() for c in df.columns]
        enriched = enrich(df, cfg=cfg)
        sig = signals(enriched, cfg=cfg)
        log.info("CCXT run: rows=%d, last signal=%s", 
                len(enriched), bool(sig.iloc[-1]))


if __name__ == "__main__":
    main()
