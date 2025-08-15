```python
# src/config.py
"""Central configuration for the bot. Adjust conservatively for live trading."""

from dataclasses import dataclass

@dataclass
class Config:
    # account & sizing
    INITIAL_CAPITAL: float = 50_000.0
    KELLY_P: float = 0.60          # assumed win probability
    KELLY_B: float = 1.5           # assumed reward:risk
    KELLY_CAP: float = 0.01        # cap Kelly fraction (1%)

    # indicators
    EMA_PERIOD: int = 20
    ATR_PERIOD: int = 14
    KELTNER_MULT: float = 1.5
    RSI_FAST: int = 3

    # trading / stops
    K_INIT: float = 1.0
    P_BE: float = 0.6
    K_TRAIL: float = 0.8
    K_FAST: float = 0.6
    R1: float = 1.0
    R2: float = 2.0
    MAX_HOLD_BARS: int = 24

    # data
    DEFAULT_TIMEFRAME: str = "15m"
    DEFAULT_LIMIT: int = 1500

    # environment
    TELEGRAM_ENABLED: bool = False

cfg = Config()