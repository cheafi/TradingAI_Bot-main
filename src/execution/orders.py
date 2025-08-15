from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import datetime

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"

@dataclass
class Order:
    symbol: str
    side: str
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    take_profit_price: Optional[float] = None
    time_in_force: str = "GTC"

class OrderManager:
    def __init__(self):
        self.active_orders: List[Order] = []

    def create_trailing_stop_order(self, symbol: str, side: str, quantity: float, trail_percent: float) -> Order:
        """Create a trailing stop order."""
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.TRAILING_STOP,
            trail_percent=trail_percent
        )
        self.active_orders.append(order)
        return order

    def create_bracket_order(self, symbol: str, side: str, quantity: float, price: float, take_profit_price: float, stop_loss_price: float) -> List[Order]:
        """Create a bracket order."""
        orders = [
            Order(symbol=symbol, side=side, quantity=quantity, order_type=OrderType.LIMIT, price=take_profit_price),
            Order(symbol=symbol, side=side, quantity=quantity, order_type=OrderType.STOP, stop_price=stop_loss_price)
        ]
        self.active_orders.extend(orders)
        return orders
