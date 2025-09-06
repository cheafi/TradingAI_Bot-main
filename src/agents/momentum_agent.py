"""Simple momentum agent using moving average crossover and RSI filter."""
from __future__ import annotations
import pandas as pd
import numpy as np
from .base import Agent

class MomentumAgent:
    name = "momentum_basic"

    def __init__(self, fast: int = 20, slow: int = 50, rsi_len: int = 14, rsi_upper: int = 70, rsi_lower: int = 30):
        self.fast = fast
        self.slow = slow
        self.rsi_len = rsi_len
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

    def _rsi(self, close: pd.Series, length: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(length).mean()
        loss = -delta.clip(upper=0).rolling(length).mean()
        rs = gain / (loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['ma_fast'] = df['close'].rolling(self.fast).mean()
        df['ma_slow'] = df['close'].rolling(self.slow).mean()
        df['rsi'] = self._rsi(df['close'], self.rsi_len)
        df['crossover'] = np.where(df['ma_fast'] > df['ma_slow'], 1, -1)
        # Signal only when RSI not extreme
        df['signal'] = np.where((df['rsi'] < self.rsi_upper) & (df['rsi'] > self.rsi_lower), df['crossover'], 0)
        # Expected alpha: scaled momentum strength
        momentum_strength = (df['ma_fast'] / (df['ma_slow'] + 1e-9) - 1) * 10_000
        df['expected_alpha_bps'] = momentum_strength.clip(-50, 50)
        df['confidence'] = df['expected_alpha_bps'].abs() / 50
        df['horizon_bars'] = np.where(df['signal'] != 0, self.slow // 2, 0)
        df['reason'] = np.where(df['signal'] != 0, 'MA crossover with neutral RSI', '')
        return df[['signal', 'expected_alpha_bps', 'confidence', 'horizon_bars', 'reason']].dropna().iloc[self.slow:]

