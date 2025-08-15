# src/utils/execution.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import os
import json
import time
import pandas as pd
import datetime
from pathlib import Path

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
    """
    Simulates paper trades, tracks positions, logs to CSV, sends Telegram alerts.
    Why: Tests strategy safely, logs for SEC compliance, monitors drawdown daily (Zipline-inspired).
    """
    def __init__(self, initial_capital: float, cfg: dict):
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.trades: List[float] = []
        self.daily_pnl: List[float] = []
        self.cfg = cfg
        self.daily_start_equity: Optional[float] = None
        self.last_date: Optional[datetime.date] = None
        self.trade_log_file = cfg.get("trade_log_csv", "trade_log.csv")
        self.state_file = Path("trader_state.json")  # Define state file path
        self.load_state()  # Load state during initialization
        with open(self.trade_log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "symbol", "side", "qty", "price", "type", "pnl", "capital_after", "reason"])

    def load_state(self):
        """Load trader state from JSON file."""
        if self.state_file.exists():
            try:
                with self.state_file.open("r") as f:
                    state = json.load(f)
                    self.capital = state.get("capital", self.initial_capital)
                    self.positions = {
                        symbol: Position(**pos_data) for symbol, pos_data in state.get("positions", {}).items()
                    }
                    self.trades = state.get("trades", [])
                    self.daily_pnl = state.get("daily_pnl", [])
                    self.daily_start_equity = state.get("daily_start_equity")
                    self.last_date = datetime.datetime.strptime(state.get("last_date"), "%Y-%m-%d").date() if state.get("last_date") else None
                logger.info("Trader state loaded from %s", self.state_file)
            except Exception as e:
                logger.warning("Failed to load trader state: %s", e)

    def save_state(self):
        """Save trader state to JSON file."""
        try:
            state = {
                "capital": self.capital,
                "positions": {symbol: pos.__dict__ for symbol, pos in self.positions.items()},
                "trades": self.trades,
                "daily_pnl": self.daily_pnl,
                "daily_start_equity": self.daily_start_equity,
                "last_date": self.last_date.strftime("%Y-%m-%d") if self.last_date else None,
            }
            with self.state_file.open("w") as f:
                json.dump(state, f)
            logger.info("Trader state saved to %s", self.state_file)
        except Exception as e:
            logger.error("Failed to save trader state: %s", e)

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

    def open_long(self, symbol: str, price: float, qty: int, ts: pd.Timestamp, idx: int, reason: str=""):
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
        self.capital -= cost
        trade = {"time": str(ts), "symbol": symbol, "side": "buy", "qty": qty, "price": price}
        self.trades.append(trade)
        logger.info("Opened %s %d @ %.6f", symbol, qty, price)
        self.save_state()  # Save state after opening position
        return True

    def close_position(self, symbol: str, price: float, reason: str=""):
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
        self.save_state()  # Save state after closing position
        return True

    def check_stop_exit(self, symbol: str, bar_low: float, ts: pd.Timestamp) -> bool:
        pos = self.positions.get(symbol)
        if not pos or pos.qty == 0:
            return False
        # Check if we need to exit based on the stop price
        if bar_low <= pos.stop:
            logger.info("Stop loss hit for %s: selling %d @ %.6f", symbol, pos.qty, bar_low)
            self.close_position(symbol, bar_low, reason="stop_loss")
            return True
        self.save_state()  # Save state after stop exit
        return False

    def check_daily_drawdown(self, ts: pd.Timestamp) -> bool:
        if not self.last_date or ts.date() != self.last_date:
            # New day, check drawdown
            self.last_date = ts.date()
            # Calculate daily PnL
            if len(self.trades) > 0:
                daily_pnl = sum(
                    trade["pnl"] for trade in self.trades if trade["time"].startswith(str(ts.date()))
                )
                self.daily_pnl.append(daily_pnl)
                # Check if we exceed the maximum allowed drawdown
                max_drawdown = self.cfg.get("max_drawdown", -1000)
                if daily_pnl < max_drawdown:
                    logger.warning("Daily drawdown limit hit: PnL=%.2f", daily_pnl)
                    # Close all positions on drawdown limit hit
                    for symbol in list(self.positions.keys()):
                        self.close_position(symbol, self.positions[symbol].avg_price, reason="drawdown_limit")
                    self.save_state()  # Save state after drawdown check
                    return True
        return False

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