# src/utils/execution.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import logging
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Position:
    qty: int = 0
    avg_price: float = 0.0
    stop: float = 0.0

class PaperTrader:
    """Simplified paper trader: buys at next-open, closes on stop or time-exit."""
    def __init__(self, capital: float):
        self.initial_capital = capital
        self.capital = capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []

    def _ensure(self, symbol: str):
        if symbol not in self.positions:
            self.positions[symbol] = Position()

    def open_long(self, symbol: str, price: float, qty: int, stop: float, ts):
        self._ensure(symbol)
        cost = price * qty
        if cost > self.capital or qty <= 0:
            return False
        pos = self.positions[symbol]
        pos.qty += qty
        pos.avg_price = price if pos.avg_price == 0 else (pos.avg_price * (pos.qty - qty) + price * qty) / pos.qty
        pos.stop = stop
        self.capital -= cost
        self.trades.append({"time": ts, "symbol": symbol, "side": "buy", "qty": qty, "price": price})
        logger.info("Opened %s %d @ %.4f", symbol, qty, price)
        return True

    def close_long(self, symbol: str, price: float, ts, reason: str = ""):
        pos = self.positions.get(symbol)
        if not pos or pos.qty <= 0:
            return False
        proceeds = price * pos.qty
        pnl = proceeds - pos.avg_price * pos.qty
        self.capital += proceeds
        self.trades.append({"time": ts, "symbol": symbol, "side": "sell", "qty": pos.qty, "price": price, "pnl": pnl, "reason": reason})
        logger.info("Closed %s %d @ %.4f PnL %.2f", symbol, pos.qty, price, pnl)
        self.positions[symbol] = Position()
        return True
