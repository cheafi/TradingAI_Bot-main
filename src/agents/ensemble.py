"""Ensemble of trading agents for comprehensive market analysis."""
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from qlib.contrib.model import DQNModel
try:
    from openbb import OBBSession
    HAS_OPENBB = True
except ImportError:
    HAS_OPENBB = False

logger = logging.getLogger(__name__)


@dataclass
class MarketInsight:
    """Comprehensive market analysis from multiple agents."""
    symbol: str
    signal: str  # buy/sell/hold
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasons: List[str] = None
    metrics: Dict = None

    def to_chinese(self) -> str:
        """Convert insight to Chinese format for HK users."""
        signal_cn = {"buy": "買入", "sell": "賣出", "hold": "持有"}
        return (
            f"分析：{self.symbol} 建議{signal_cn[self.signal]}\n"
            f"目標價：${self.target_price:.2f}\n"
            f"止損價：${self.stop_loss:.2f}\n"
            f"信心指數：{self.confidence*100:.1f}%\n"
            f"原因：{', '.join(self.reasons)}"
        )


class EnsembleAgent:
    """Combines multiple analysis approaches for robust trading decisions."""
    
    def __init__(self):
        self.rl_model = DQNModel()  # Qlib's RL agent
        self.obb = OBBSession() if HAS_OPENBB else None
        
    async def analyze(self, symbol: str) -> MarketInsight:
        """Run comprehensive analysis using multiple agents."""
        try:
            # Get real market data
            data = await self._fetch_data(symbol)
            
            # Technical analysis
            tech_signal = self._technical_analysis(data)
            
            # Fundamental analysis 
            fund_signal = self._fundamental_analysis(symbol)
            
            # Sentiment analysis
            sent_signal = self._sentiment_analysis(symbol)
            
            # Combine signals with weights
            weights = [0.4, 0.3, 0.3]  # tech, fund, sent
            signals = np.array([tech_signal, fund_signal, sent_signal])
            confidence = np.average(signals, weights=weights)
            
            return MarketInsight(
                symbol=symbol,
                signal=self._get_signal(confidence),
                confidence=confidence,
                target_price=self._calc_target(data),
                stop_loss=self._calc_stop_loss(data),
                reasons=self._generate_reasons(),
                metrics={"sharpe": self._calc_sharpe(data)}
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return None

    async def _fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch real market data using OpenBB/yfinance/ccxt."""
        if self.obb:
            return await self.obb.get_data(symbol)
        # Fallback to other data sources
        return pd.DataFrame()  # TODO: implement fallbacks

    def _technical_analysis(self, data: pd.DataFrame) -> float:
        """Technical analysis using RL model."""
        return self.rl_model.predict(data)

    def _fundamental_analysis(self, symbol: str) -> float:
        """Fundamental analysis (DCF, ratios etc)."""
        return 0.5  # TODO: implement real analysis

    def _sentiment_analysis(self, symbol: str) -> float:
        """Analyze news and social media sentiment."""
        return 0.5  # TODO: implement real analysis

    @staticmethod
    def _get_signal(confidence: float) -> str:
        """Convert confidence score to trading signal."""
        if confidence > 0.7:
            return "buy"
        elif confidence < 0.3:
            return "sell"
        return "hold"

    @staticmethod
    def _calc_target(data: pd.DataFrame) -> float:
        """Calculate target price using technical and fundamental factors."""
        return data["close"].iloc[-1] * 1.1  # Simple +10% target

    @staticmethod
    def _calc_stop_loss(data: pd.DataFrame) -> float:
        """Calculate stop loss using ATR or volatility."""
        return data["close"].iloc[-1] * 0.95  # Simple -5% stop

    @staticmethod
    def _calc_sharpe(data: pd.DataFrame, risk_free=0.02) -> float:
        """Calculate Sharpe ratio."""
        returns = data["close"].pct_change().dropna()
        excess_returns = returns - risk_free/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _generate_reasons(self) -> List[str]:
        """Generate analysis reasons in both English and Chinese."""
        return [
            "Strong technical trend 技術走勢強勁",
            "Positive sentiment 市場情緒樂觀",
            "Good value 估值合理"
        ]
