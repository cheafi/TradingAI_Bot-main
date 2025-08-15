# src/core/events.py
"""Simple async EventBus used to decouple components."""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Event:
    name: str
    payload: Dict[str, Any]


class EventBus:
    def __init__(self) -> None:
        self.handlers: Dict[str, List[Callable[[Event], Any]]] = {}

    def subscribe(self, name: str, handler: Callable[[Event], Any]) -> None:
        self.handlers.setdefault(name, []).append(handler)

    async def publish(self, name: str, payload: Dict[str, Any]) -> None:
        evt = Event(name=name, payload=payload)
        for fn in self.handlers.get(name, []):
            try:
                res = fn(evt)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.exception("Event handler failed: %s", exc)
