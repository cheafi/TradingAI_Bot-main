from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List
import asyncio
import logging

@dataclass
class Event:
    type: str
    payload: Dict[str, Any]

class EventBus:
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[Callable[[Event], Any]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        self.subscribers.setdefault(event_type, []).append(handler)

    async def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        evt = Event(event_type, payload)
        for handler in self.subscribers.get(event_type, []):
            try:
                res = handler(evt)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logging.exception(f"[EventBus] handler failed: {e}")