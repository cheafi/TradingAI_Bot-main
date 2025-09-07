from __future__ import annotations
import time
import threading
from typing import Any, Callable


class TTLCache:
    def __init__(self, ttl_seconds: int = 60, max_items: int = 512):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            ts, val = item
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return val

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.max_items:
                oldest = min(self._store.items(), key=lambda x: x[1][0])[0]
                self._store.pop(oldest, None)
            self._store[key] = (time.time(), value)

    def cached(self, key: str, supplier: Callable[[], Any]) -> Any:
        val = self.get(key)
        if val is not None:
            return val
        val = supplier()
        if val is not None:
            self.set(key, val)
        return val


GLOBAL_CACHE = TTLCache(ttl_seconds=120)
