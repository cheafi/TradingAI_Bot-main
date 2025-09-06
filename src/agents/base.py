"""Agent interface for signal generation.
Each agent must implement generate_signals(data: pd.DataFrame) -> pd.DataFrame
with columns: ['signal', 'expected_alpha_bps', 'confidence', 'horizon_bars', 'reason'].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any
import pandas as pd

class Agent(Protocol):  # pragma: no cover
    name: str
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class SignalRecord:
    signal: int
    expected_alpha_bps: float
    confidence: float
    horizon_bars: int
    reason: str

@dataclass
class AgentMetadata:
    name: str
    version: str = "0.1.0"
    author: str = "system"

