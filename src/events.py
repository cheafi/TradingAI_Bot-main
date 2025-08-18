import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Any, Awaitable

@dataclass
class Event:
    type: str
    payload: Dict[str, Any]

class EventBus:
    def __init__(self):
        self._handlers: Dict[str, Callable[[Event], Awaitable[None]]] = {}

    def on(self, type_: str):
        def deco(fn):
            self._handlers[type_] = fn
            return fn
        return deco

    async def emit(self, type_: str, payload: Dict[str, Any]):
        if type_ in self._handlers:
            await self._handlers[type_](Event(type_, payload))
