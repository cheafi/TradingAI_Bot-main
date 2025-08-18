import os, sys, argparse, logging
import pandas as pd
from datetime import datetime
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.data import synthetic_ohlcv, fetch_ohlcv_ccxt
from src.strategies.scalping import enrich, signals
from src.utils.risk import sharpe, max_drawdown
from src.config import cfg

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("main")

def demo_run(symbol: str="BTC/USDT", limit:int=1200):
    df = synthetic_ohlcv(symbol, limit=limit)
    df.columns=[c.lower() for c in df.columns]
    enriched = enrich(df, cfg=cfg)
    sig = signals(enriched, cfg=cfg)
    ret = enriched["close"].pct_change().dropna()
    eq = (1+ret).cumprod()
    log.info("Demo completed: %s rows, Sharpe=%.2f, MDD=%.2f%%", len(enriched), sharpe(ret), max_drawdown(eq)*100)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo","ccxt"], default="demo")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--limit", type=int, default=1200)
    args = p.parse_args()

    if args.mode == "demo":
        demo_run(args.symbol, args.limit)
    else:
        df = fetch_ohlcv_ccxt(args.symbol, limit=args.limit)
        df.columns=[c.lower() for c in df.columns]
        enriched = enrich(df, cfg=cfg)
        sig = signals(enriched, cfg=cfg)
        log.info("CCXT run: rows=%d, last signal=%s", len(enriched), bool(sig.iloc[-1]))

if __name__ == "__main__":
    main()
