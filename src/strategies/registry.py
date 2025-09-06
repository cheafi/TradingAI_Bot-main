"""Strategy registry and interface.

Each strategy entry:
    name: {
        'config': ConfigClass/Callable,
        'enrich': callable(df, context) -> df
        'signal': callable(df, context) -> dict or Series (entries/exits)
    }
"""
from __future__ import annotations

from . import scalping

REGISTRY = {
    "scalping_ml": {
        "config": getattr(scalping, "ScalpingConfig", object),
        "enrich": scalping.enrich,
        "signal": getattr(scalping, "signals"),
    },
}


def list_strategies() -> list[str]:
    return sorted(REGISTRY.keys())


def get_strategy(name: str):
    return REGISTRY.get(name)