"""
Multi-Agent Alpha Factory - Institutional Investment Styles
Implementation of famous investor strategies as autonomous agents.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from platform.events import (
    Event, EventProcessor, SignalEvent, FeatureEvent, 
    event_bus, EventType
)
from platform.config import AgentConfig, AgentType


logger = logging.getLogger(__name__)


@dataclass
class AgentSignal:
    """Standardized agent signal output."""
    
    symbol: str
    signal_strength: float  # -1 to 1 (sell to buy)
    confidence: float       # 0 to 1
    reasoning: str
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    horizon_days: Optional[int] = None
    risk_factors: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "horizon_days": self.horizon_days,
            "risk_factors": self.risk_factors or []
        }


class BaseAgent(EventProcessor, ABC):
    """Base class for all trading agents."""
    
    def __init__(self, config: AgentConfig, event_bus):
        self.config = config
        self.name = config.name
        self.agent_type = config.agent_type
        self.enabled = config.enabled
        self.weight = config.weight
        self.parameters = config.parameters
        
        # State
        self.last_signals: Dict[str, AgentSignal] = {}
        self.feature_cache: Dict[str, Dict] = {}
        
        super().__init__(event_bus)
    
    def setup_subscriptions(self):
        """Subscribe to feature events."""
        self.event_bus.subscribe(EventType.FEATURE, self)
    
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle feature events and generate signals."""
        if not self.enabled:
            return None
            
        if isinstance(event, FeatureEvent):
            # Cache features
            self.feature_cache[event.symbol] = event.features
            
            # Generate signal
            signal = await self.generate_signal(event.symbol, event.features)
            
            if signal and signal.confidence >= self.parameters.get("min_conviction", 0.6):
                self.last_signals[event.symbol] = signal
                
                # Create signal event
                signal_event = SignalEvent(
                    symbol=signal.symbol,
                    signal_strength=signal.signal_strength,
                    confidence=signal.confidence,
                    agent_name=self.name,
                    reasoning=signal.reasoning,
                    target_price=signal.target_price,
                    stop_price=signal.stop_price,
                    horizon_days=signal.horizon_days
                )
                
                return [signal_event]
        
        return None
    
    @abstractmethod
    async def generate_signal(
        self, 
        symbol: str, 
        features: Dict[str, float]
    ) -> Optional[AgentSignal]:
        """Generate trading signal for a symbol."""
        pass
    
    def _calculate_confidence(self, *factors: float) -> float:
        """Calculate confidence from multiple factors."""
        # Geometric mean for conservative confidence
        factors = [max(0.01, min(0.99, f)) for f in factors]
        confidence = np.prod(factors) ** (1 / len(factors))
        return confidence


class BenGrahamAgent(BaseAgent):
    """Ben Graham Deep Value Agent - Net-nets and statistical cheapness."""
    
    async def generate_signal(
        self, 
        symbol: str, 
        features: Dict[str, float]
    ) -> Optional[AgentSignal]:
        """
        Ben Graham criteria:
        - P/B < 1.5 (preferably < 1.0)
        - P/E < 15
        - Current ratio > 1.5
        - Debt/Equity < 30%
        - FCF yield > 10%
        """
        try:
            pb_ratio = features.get("pb_ratio", float("inf"))
            pe_ratio = features.get("pe_ratio", float("inf"))
            current_ratio = features.get("current_ratio", 0)
            debt_equity = features.get("debt_equity", float("inf"))
            fcf_yield = features.get("fcf_yield", 0)
            
            # Graham scoring
            score = 0
            reasons = []
            
            # P/B criterion (most important)
            if pb_ratio < 1.0:
                score += 0.4
                reasons.append(f"Excellent P/B: {pb_ratio:.2f}")
            elif pb_ratio < 1.5:
                score += 0.2
                reasons.append(f"Good P/B: {pb_ratio:.2f}")
            
            # P/E criterion
            if pe_ratio < 10:
                score += 0.3
                reasons.append(f"Very cheap P/E: {pe_ratio:.2f}")
            elif pe_ratio < 15:
                score += 0.15
                reasons.append(f"Reasonable P/E: {pe_ratio:.2f}")
            
            # Financial strength
            if current_ratio > 2.0:
                score += 0.15
                reasons.append(f"Strong liquidity: {current_ratio:.2f}")
            elif current_ratio > 1.5:
                score += 0.1
                reasons.append(f"Adequate liquidity: {current_ratio:.2f}")
            
            # Low debt
            if debt_equity < 0.3:
                score += 0.15
                reasons.append(f"Conservative debt: {debt_equity:.2f}")
            
            # Cash generation
            if fcf_yield > 0.15:
                score += 0.2
                reasons.append(f"Excellent FCF yield: {fcf_yield:.2%}")
            elif fcf_yield > 0.1:
                score += 0.1
                reasons.append(f"Good FCF yield: {fcf_yield:.2%}")
            
            if score < 0.3:  # Minimum threshold
                return None
            
            # Calculate target price (conservative)
            book_value = features.get("book_value_per_share", 0)
            target_price = book_value * 1.2  # 20% premium to book
            
            # Conservative stop loss
            current_price = features.get("price", 0)
            stop_price = current_price * 0.85  # 15% stop loss
            
            signal_strength = min(score, 0.8)  # Cap at 0.8 for conservatism
            confidence = self._calculate_confidence(score, min(1.0, 1/pb_ratio))
            
            return AgentSignal(
                symbol=symbol,
                signal_strength=signal_strength,
                confidence=confidence,
                reasoning=f"Ben Graham criteria met: {'; '.join(reasons)}",
                target_price=target_price,
                stop_price=stop_price,
                horizon_days=365,  # Long-term value play
                risk_factors=["Market downturn", "Value trap", "Fundamental deterioration"]
            )
            
        except Exception as e:
            logger.error(f"BenGrahamAgent error for {symbol}: {e}")
            return None


class WarrenBuffettAgent(BaseAgent):
    """Warren Buffett Quality Agent - Wonderful businesses at fair prices."""
    
    async def generate_signal(
        self, 
        symbol: str, 
        features: Dict[str, float]
    ) -> Optional[AgentSignal]:
        """
        Buffett criteria:
        - High ROIC (>15%)
        - Consistent revenue growth
        - Strong margins and moat
        - Reasonable price
        """
        try:
            roic = features.get("roic", 0)
            roe = features.get("roe", 0)
            revenue_growth_3y = features.get("revenue_growth_3y", 0)
            gross_margin = features.get("gross_margin", 0)
            net_margin = features.get("net_margin", 0)
            pe_ratio = features.get("pe_ratio", float("inf"))
            debt_equity = features.get("debt_equity", 1.0)
            
            score = 0
            reasons = []
            
            # Return on invested capital (key metric)
            if roic > 0.25:
                score += 0.3
                reasons.append(f"Exceptional ROIC: {roic:.1%}")
            elif roic > 0.15:
                score += 0.2
                reasons.append(f"Strong ROIC: {roic:.1%}")
            
            # Return on equity
            if roe > 0.20:
                score += 0.2
                reasons.append(f"Excellent ROE: {roe:.1%}")
            elif roe > 0.15:
                score += 0.1
                reasons.append(f"Good ROE: {roe:.1%}")
            
            # Growth consistency
            if revenue_growth_3y > 0.10:
                score += 0.15
                reasons.append(f"Strong growth: {revenue_growth_3y:.1%}")
            elif revenue_growth_3y > 0.05:
                score += 0.1
                reasons.append(f"Steady growth: {revenue_growth_3y:.1%}")
            
            # Margins (economic moat indicator)
            if gross_margin > 0.4 and net_margin > 0.15:
                score += 0.2
                reasons.append(f"Wide moat margins: {gross_margin:.1%}/{net_margin:.1%}")
            elif gross_margin > 0.3 and net_margin > 0.1:
                score += 0.1
                reasons.append(f"Good margins: {gross_margin:.1%}/{net_margin:.1%}")
            
            # Reasonable valuation
            if pe_ratio < 20:
                score += 0.15
                reasons.append(f"Fair valuation: {pe_ratio:.1f} P/E")
            elif pe_ratio < 30:
                score += 0.05
                reasons.append(f"Acceptable valuation: {pe_ratio:.1f} P/E")
            
            # Conservative debt
            if debt_equity < 0.5:
                score += 0.1
                reasons.append(f"Conservative debt: {debt_equity:.2f}")
            
            if score < 0.4:  # High quality bar
                return None
            
            # DCF-based target (simplified)
            earnings_per_share = features.get("eps", 0)
            growth_rate = min(revenue_growth_3y, 0.15)  # Cap growth assumptions
            target_pe = min(pe_ratio * 1.2, 25)  # Reasonable multiple
            target_price = earnings_per_share * target_pe * (1 + growth_rate)
            
            current_price = features.get("price", 0)
            stop_price = current_price * 0.8  # 20% stop for quality names
            
            signal_strength = min(score * 0.9, 0.9)  # Strong but not maximum
            confidence = self._calculate_confidence(
                score, 
                min(1.0, roic * 3),  # Weight ROIC heavily
                min(1.0, (1 - debt_equity))
            )
            
            return AgentSignal(
                symbol=symbol,
                signal_strength=signal_strength,
                confidence=confidence,
                reasoning=f"Buffett quality criteria: {'; '.join(reasons)}",
                target_price=target_price,
                stop_price=stop_price,
                horizon_days=1095,  # 3 years - long term
                risk_factors=["Multiple compression", "Competitive pressure", "Regulatory risk"]
            )
            
        except Exception as e:
            logger.error(f"WarrenBuffettAgent error for {symbol}: {e}")
            return None


class TechnicalAgent(BaseAgent):
    """Technical Analysis Agent - Momentum and mean reversion signals."""
    
    async def generate_signal(
        self, 
        symbol: str, 
        features: Dict[str, float]
    ) -> Optional[AgentSignal]:
        """
        Technical indicators:
        - RSI for momentum
        - Moving averages for trend
        - Volume confirmation
        - Support/resistance levels
        """
        try:
            rsi = features.get("rsi_14", 50)
            price = features.get("price", 0)
            sma_20 = features.get("sma_20", price)
            sma_50 = features.get("sma_50", price)
            volume_ratio = features.get("volume_ratio_20", 1.0)
            bollinger_position = features.get("bollinger_position", 0.5)
            
            signal_strength = 0
            reasons = []
            
            # RSI momentum
            if rsi < 30:
                signal_strength += 0.4
                reasons.append(f"Oversold RSI: {rsi:.1f}")
            elif rsi > 70:
                signal_strength -= 0.4
                reasons.append(f"Overbought RSI: {rsi:.1f}")
            
            # Moving average trend
            if price > sma_20 > sma_50:
                signal_strength += 0.3
                reasons.append("Bullish MA alignment")
            elif price < sma_20 < sma_50:
                signal_strength -= 0.3
                reasons.append("Bearish MA alignment")
            
            # Volume confirmation
            if volume_ratio > 1.5:
                signal_strength *= 1.2  # Amplify with volume
                reasons.append(f"Strong volume: {volume_ratio:.1f}x")
            
            # Bollinger bands mean reversion
            if bollinger_position < 0.2:
                signal_strength += 0.2
                reasons.append("Near lower Bollinger band")
            elif bollinger_position > 0.8:
                signal_strength -= 0.2
                reasons.append("Near upper Bollinger band")
            
            # Normalize signal strength
            signal_strength = max(-1.0, min(1.0, signal_strength))
            
            if abs(signal_strength) < 0.3:
                return None
            
            # Technical targets
            if signal_strength > 0:
                target_price = price * 1.1  # 10% upside target
                stop_price = price * 0.95   # 5% stop loss
            else:
                target_price = price * 0.9  # 10% downside target
                stop_price = price * 1.05   # 5% stop loss (short)
            
            confidence = self._calculate_confidence(
                abs(signal_strength),
                min(1.0, volume_ratio / 2),
                0.7  # Technical analysis inherent uncertainty
            )
            
            return AgentSignal(
                symbol=symbol,
                signal_strength=signal_strength,
                confidence=confidence,
                reasoning=f"Technical setup: {'; '.join(reasons)}",
                target_price=target_price,
                stop_price=stop_price,
                horizon_days=30,  # Short to medium term
                risk_factors=["Whipsaws", "False breakouts", "Market noise"]
            )
            
        except Exception as e:
            logger.error(f"TechnicalAgent error for {symbol}: {e}")
            return None


class SentimentAgent(BaseAgent):
    """Sentiment Analysis Agent - News and social sentiment signals."""
    
    def __init__(self, config: AgentConfig, event_bus):
        super().__init__(config, event_bus)
        self.sentiment_cache: Dict[str, float] = {}
        
    async def generate_signal(
        self, 
        symbol: str, 
        features: Dict[str, float]
    ) -> Optional[AgentSignal]:
        """
        Sentiment indicators:
        - News sentiment score
        - Social media buzz
        - Analyst revisions
        - Earnings call tone
        """
        try:
            news_sentiment = features.get("news_sentiment_7d", 0)
            social_sentiment = features.get("social_sentiment", 0)
            analyst_revisions = features.get("analyst_revisions_30d", 0)
            earnings_sentiment = features.get("earnings_call_sentiment", 0)
            sentiment_momentum = features.get("sentiment_momentum", 0)
            
            # Weighted sentiment score
            weights = [0.3, 0.2, 0.3, 0.2]  # News, social, analysts, earnings
            sentiments = [news_sentiment, social_sentiment, analyst_revisions, earnings_sentiment]
            
            weighted_sentiment = sum(w * s for w, s in zip(weights, sentiments) if s != 0)
            weight_sum = sum(w for w, s in zip(weights, sentiments) if s != 0)
            
            if weight_sum == 0:
                return None
                
            avg_sentiment = weighted_sentiment / weight_sum
            
            # Apply momentum factor
            final_sentiment = avg_sentiment * (1 + sentiment_momentum * 0.5)
            
            # Convert to signal strength
            signal_strength = np.tanh(final_sentiment * 2)  # Smooth bounded function
            
            if abs(signal_strength) < 0.2:
                return None
            
            reasons = []
            if news_sentiment > 0.3:
                reasons.append(f"Positive news sentiment: {news_sentiment:.2f}")
            elif news_sentiment < -0.3:
                reasons.append(f"Negative news sentiment: {news_sentiment:.2f}")
                
            if analyst_revisions > 0.2:
                reasons.append(f"Positive analyst revisions: {analyst_revisions:.2f}")
            elif analyst_revisions < -0.2:
                reasons.append(f"Negative analyst revisions: {analyst_revisions:.2f}")
            
            # Sentiment-based targets (less precise)
            current_price = features.get("price", 0)
            if signal_strength > 0:
                target_price = current_price * (1 + abs(signal_strength) * 0.15)
                stop_price = current_price * 0.92
            else:
                target_price = current_price * (1 - abs(signal_strength) * 0.15)
                stop_price = current_price * 1.08
            
            confidence = self._calculate_confidence(
                abs(avg_sentiment),
                min(1.0, abs(sentiment_momentum) + 0.5),
                0.6  # Sentiment can be noisy
            )
            
            return AgentSignal(
                symbol=symbol,
                signal_strength=signal_strength,
                confidence=confidence,
                reasoning=f"Sentiment analysis: {'; '.join(reasons)}",
                target_price=target_price,
                stop_price=stop_price,
                horizon_days=14,  # Short term sentiment plays
                risk_factors=["Sentiment reversal", "News fatigue", "Contrarian moves"]
            )
            
        except Exception as e:
            logger.error(f"SentimentAgent error for {symbol}: {e}")
            return None


class AgentOrchestrator:
    """Orchestrates multiple agents and combines their signals."""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.agents: Dict[str, BaseAgent] = {}
        self.signal_aggregator = BayesianSignalAggregator()
        
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the orchestrator."""
        self.agents[agent.name] = agent
        logger.info(f"Added agent: {agent.name}")
    
    def get_combined_signals(self, symbol: str) -> Optional[AgentSignal]:
        """Get combined signal for a symbol from all agents."""
        agent_signals = []
        
        for agent in self.agents.values():
            if symbol in agent.last_signals:
                signal = agent.last_signals[symbol]
                agent_signals.append((signal, agent.weight))
        
        if not agent_signals:
            return None
            
        return self.signal_aggregator.combine_signals(agent_signals)
    
    def get_agent_consensus(self, symbol: str) -> Dict[str, Any]:
        """Get detailed consensus information for a symbol."""
        signals = {}
        total_weight = 0
        weighted_signal = 0
        weighted_confidence = 0
        
        for agent in self.agents.values():
            if symbol in agent.last_signals:
                signal = agent.last_signals[symbol]
                signals[agent.name] = signal.to_dict()
                
                weight = agent.weight
                total_weight += weight
                weighted_signal += signal.signal_strength * weight
                weighted_confidence += signal.confidence * weight
        
        if total_weight == 0:
            return {}
            
        return {
            "symbol": symbol,
            "agent_signals": signals,
            "consensus_signal": weighted_signal / total_weight,
            "consensus_confidence": weighted_confidence / total_weight,
            "num_agents": len(signals),
            "total_weight": total_weight
        }


class BayesianSignalAggregator:
    """Bayesian approach to combining agent signals."""
    
    def combine_signals(
        self, 
        agent_signals: List[Tuple[AgentSignal, float]]
    ) -> AgentSignal:
        """Combine multiple agent signals using Bayesian model averaging."""
        
        if not agent_signals:
            raise ValueError("No signals to combine")
        
        if len(agent_signals) == 1:
            return agent_signals[0][0]
        
        # Extract signals and weights
        signals = [signal for signal, _ in agent_signals]
        weights = np.array([weight for _, weight in agent_signals])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted average of signal strengths
        signal_strengths = np.array([s.signal_strength for s in signals])
        combined_signal = np.average(signal_strengths, weights=weights)
        
        # Uncertainty reduction through combination
        confidences = np.array([s.confidence for s in signals])
        combined_confidence = 1 - np.prod(1 - confidences * weights)
        
        # Combine reasoning
        reasoning_parts = []
        for i, (signal, weight) in enumerate(agent_signals):
            reasoning_parts.append(f"{signals[i].reasoning} (weight: {weight:.2f})")
        
        combined_reasoning = "Combined analysis: " + "; ".join(reasoning_parts)
        
        # Use the most conservative targets and stops
        target_prices = [s.target_price for s in signals if s.target_price is not None]
        stop_prices = [s.stop_price for s in signals if s.stop_price is not None]
        
        combined_target = np.median(target_prices) if target_prices else None
        combined_stop = np.median(stop_prices) if stop_prices else None
        
        # Average horizon
        horizons = [s.horizon_days for s in signals if s.horizon_days is not None]
        combined_horizon = int(np.median(horizons)) if horizons else None
        
        # Combine risk factors
        all_risk_factors = []
        for signal in signals:
            if signal.risk_factors:
                all_risk_factors.extend(signal.risk_factors)
        
        unique_risks = list(set(all_risk_factors))
        
        return AgentSignal(
            symbol=signals[0].symbol,
            signal_strength=combined_signal,
            confidence=combined_confidence,
            reasoning=combined_reasoning,
            target_price=combined_target,
            stop_price=combined_stop,
            horizon_days=combined_horizon,
            risk_factors=unique_risks
        )
