"""
Event-driven pipeline architecture for institutional trading.
Market data → Features → Signals → Intents → Orders → Fills → PnL/Risk
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events in the pipeline."""
    MARKET_DATA = "market_data"
    FEATURE = "feature"
    SIGNAL = "signal"
    INTENT = "intent"
    ORDER = "order"
    FILL = "fill"
    PNL = "pnl"
    RISK = "risk"
    SYSTEM = "system"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Event:
    """Base event class."""
    event_id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.SYSTEM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataEvent(Event):
    """Market data event."""
    event_type: EventType = EventType.MARKET_DATA
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    def __post_init__(self):
        self.data.update({
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask
        })


@dataclass 
class FeatureEvent(Event):
    """Feature computation event."""
    event_type: EventType = EventType.FEATURE
    symbol: str = ""
    features: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.data.update({
            "symbol": self.symbol,
            "features": self.features
        })


@dataclass
class SignalEvent(Event):
    """Trading signal event."""
    event_type: EventType = EventType.SIGNAL
    symbol: str = ""
    signal_strength: float = 0.0  # -1 to 1
    confidence: float = 0.0       # 0 to 1
    agent_name: str = ""
    reasoning: str = ""
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    horizon_days: Optional[int] = None
    
    def __post_init__(self):
        self.data.update({
            "symbol": self.symbol,
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
            "agent_name": self.agent_name,
            "reasoning": self.reasoning,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "horizon_days": self.horizon_days
        })


@dataclass
class IntentEvent(Event):
    """Portfolio intent event."""
    event_type: EventType = EventType.INTENT
    symbol: str = ""
    target_weight: float = 0.0    # Target portfolio weight
    current_weight: float = 0.0   # Current portfolio weight
    dollar_amount: Optional[float] = None
    urgency: float = 0.5          # 0 to 1
    max_participation: float = 0.1  # Max % of volume
    
    def __post_init__(self):
        self.data.update({
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "dollar_amount": self.dollar_amount,
            "urgency": self.urgency,
            "max_participation": self.max_participation
        })


@dataclass
class OrderEvent(Event):
    """Order event."""
    event_type: EventType = EventType.ORDER
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    broker: str = ""
    
    def __post_init__(self):
        self.data.update({
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "broker": self.broker
        })


@dataclass
class FillEvent(Event):
    """Fill execution event."""
    event_type: EventType = EventType.FILL
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    implementation_shortfall: float = 0.0
    
    def __post_init__(self):
        self.data.update({
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "slippage": self.slippage,
            "implementation_shortfall": self.implementation_shortfall
        })


@dataclass
class PnLEvent(Event):
    """PnL computation event."""
    event_type: EventType = EventType.PNL
    symbol: Optional[str] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    position_value: float = 0.0
    portfolio_value: float = 0.0
    
    def __post_init__(self):
        self.data.update({
            "symbol": self.symbol,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "position_value": self.position_value,
            "portfolio_value": self.portfolio_value
        })


@dataclass
class RiskEvent(Event):
    """Risk monitoring event."""
    event_type: EventType = EventType.RISK
    risk_type: str = ""
    severity: str = ""  # info, warning, critical
    message: str = ""
    symbol: Optional[str] = None
    value: Optional[float] = None
    limit: Optional[float] = None
    
    def __post_init__(self):
        self.data.update({
            "risk_type": self.risk_type,
            "severity": self.severity,
            "message": self.message,
            "symbol": self.symbol,
            "value": self.value,
            "limit": self.limit
        })


class EventHandler(Protocol):
    """Protocol for event handlers."""
    
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle an event and optionally return new events."""
        ...


class EventBus:
    """Central event bus for the trading system."""
    
    def __init__(self):
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.event_queue = asyncio.Queue()
        self.running = False
        self.event_history: List[Event] = []
        self.max_history = 10000
    
    def subscribe(self, event_type: EventType, handler: EventHandler):
        """Subscribe a handler to an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Subscribed {handler.__class__.__name__} to {event_type}")
    
    async def publish(self, event: Event):
        """Publish an event to the bus."""
        await self.event_queue.put(event)
    
    async def start(self):
        """Start the event processing loop."""
        self.running = True
        logger.info("Event bus started")
        
        while self.running:
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def stop(self):
        """Stop the event bus."""
        self.running = False
        logger.info("Event bus stopped")
    
    async def _process_event(self, event: Event):
        """Process a single event."""
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Process with registered handlers
        handlers = self.handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                new_events = await handler.handle(event)
                if new_events:
                    for new_event in new_events:
                        await self.publish(new_event)
            except Exception as e:
                logger.error(f"Handler {handler.__class__.__name__} failed: {e}")
    
    def get_recent_events(
        self, 
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get recent events, optionally filtered by type."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]


class EventProcessor(ABC):
    """Base class for event processors."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.setup_subscriptions()
    
    @abstractmethod
    def setup_subscriptions(self):
        """Setup event subscriptions."""
        pass
    
    @abstractmethod
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle an event."""
        pass


# Global event bus instance
event_bus = EventBus()
