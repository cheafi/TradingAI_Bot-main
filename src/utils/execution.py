# src/utils/execution.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import os
import json
import time
import pandas as pd

logger = logging.getLogger(__name__)

# Optional Redis persistence
REDIS_URL = os.getenv("REDIS_URL", "")

try:
    import redis
    _redis_client = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    _redis_client = None
    logger.debug("redis not installed or REDIS_URL not set; running without persistence.")

@dataclass
class Position:
    qty: int = 0
    avg_price: float = 0.0
    stop: float = 0.0
    entry_time: Optional[pd.Timestamp] = None

class PaperTrader:
    def __init__(self, capital: float):
        self.initial_capital = float(capital)
        self.capital = float(capital)
        self.positions: Dict[str, Position] = {}
        self.trades: List[dict] = []
        # restore state
        if _redis_client:
            self._restore()

    # persistence helpers
    def _persist(self):
        if not _redis_client:
            return
        try:
            state = {"capital": self.capital, "positions": {k: v.__dict__ for k, v in self.positions.items()}, "trades": self.trades}
            _redis_client.set("paper_trader_state", json.dumps(state))
        except Exception as exc:
            logger.exception("Redis persist failed: %s", exc)

    def _restore(self):
        try:
            payload = _redis_client.get("paper_trader_state")
            if not payload:
                return
            state = json.loads(payload)
            self.capital = state.get("capital", self.capital)
            posdict = state.get("positions", {})
            for k, v in posdict.items():
                self.positions[k] = Position(**v)
            self.trades = state.get("trades", [])
            logger.info("Restored trader state from Redis.")
        except Exception as exc:
            logger.exception("Redis restore failed: %s", exc)

    def _ensure(self, symbol: str):
        if symbol not in self.positions:
            self.positions[symbol] = Position()

    def open_long(self, symbol: str, price: float, qty: int, stop: float, ts):
        self._ensure(symbol)
        cost = price * qty
        if qty <= 0 or cost > self.capital:
            logger.info("Open rejected: qty=%s cost=%.2f capital=%.2f", qty, cost, self.capital)
            return False
        pos = self.positions[symbol]
        # average in
        if pos.qty == 0:
            pos.avg_price = price
            pos.entry_time = ts
        else:
            pos.avg_price = (pos.avg_price * pos.qty + price * qty) / (pos.qty + qty)
        pos.qty += qty
        pos.stop = stop
        self.capital -= cost
        trade = {"time": str(ts), "symbol": symbol, "side": "buy", "qty": qty, "price": price}
        self.trades.append(trade)
        logger.info("Opened %s %d @ %.6f", symbol, qty, price)
        self._persist()
        return True

    def close_long(self, symbol: str, price: float, ts, reason: str = ""):
        pos = self.positions.get(symbol)
        if not pos or pos.qty == 0:
            return False
        proceeds = price * pos.qty
        pnl = proceeds - pos.avg_price * pos.qty
        self.capital += proceeds
        trade = {"time": str(ts), "symbol": symbol, "side": "sell", "qty": pos.qty, "price": price, "pnl": pnl, "reason": reason}
        self.trades.append(trade)
        logger.info("Closed %s %d @ %.6f PnL=%.2f", symbol, pos.qty, price, pnl)
        self.positions[symbol] = Position()
        self._persist()
        return True

# --- Live execution using ccxt (if available) ---
def live_place_order_ccxt(exchange_id: str, symbol: str, side: str, price: float, qty: float, api_key: str, api_secret: str):
    """
    Place limit order using ccxt. If ccxt not installed or fails, raise exception.
    """
    try:
        import ccxt
    except Exception as exc:
        raise RuntimeError("ccxt not available for live orders") from exc
    try:
        ex_cls = getattr(ccxt, exchange_id)
        ex = ex_cls({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
        # For simplicity: create a limit order
        if side.lower() == "buy":
            order = ex.create_limit_buy_order(symbol, qty, price)
        else:
            order = ex.create_limit_sell_order(symbol, qty, price)
        return order
    except Exception as exc:
        logger.exception("Live order failed: %s", exc)
        raise