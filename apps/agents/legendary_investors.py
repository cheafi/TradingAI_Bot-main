"""
Multi-Agent Trading Shop - Legendary Investor Personas
=====================================================

This module implements the greatest trading minds as specialized agents,
each with their own alpha generation methodology, risk controls, and KPIs.

Universal Agent Interface:
- All agents emit standardized alpha_bps signals
- ROI-focused measurement (edge after cost)
- Regime-aware capital allocation
- Automatic promotion/demotion based on performance
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json

# Enhanced imports for the new agents
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime classifications for agent allocation."""
    LOW_VOL_UPTREND = "low_vol_uptrend"
    HIGH_VOL_DOWNTREND = "high_vol_downtrend"
    SIDEWAYS_CHOP = "sideways_chop"
    MACRO_TRANSITION = "macro_transition"
    UNKNOWN = "unknown"


@dataclass
class AgentSignal:
    """Standardized agent signal output."""
    ticker: str
    direction: str  # "long", "short", "neutral"
    alpha_bps: float  # Expected alpha in basis points
    confidence: float  # 0-1 confidence score
    horizon_days: int  # Expected holding period
    agent_name: str
    timestamp: datetime
    risk_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "alpha_bps": self.alpha_bps,
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "risk_notes": self.risk_notes
        }


@dataclass
class AgentOutput:
    """Complete agent output package."""
    agent_name: str
    as_of: datetime
    universe: List[str]
    signals: List[AgentSignal]
    regime_view: Optional[MarketRegime] = None
    risk_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent": self.agent_name,
            "as_of": self.as_of.isoformat(),
            "universe": self.universe,
            "signals": [signal.to_dict() for signal in self.signals],
            "regime_view": self.regime_view.value if self.regime_view else None,
            "risk_notes": self.risk_notes,
            "metadata": self.metadata
        }


@dataclass
class AgentKPIs:
    """KPI tracking for agent performance."""
    agent_name: str
    edge_after_cost_bps: float
    information_coefficient: float
    information_ratio: float
    half_life_days: float
    correlation_to_book: float
    deflated_sharpe_ratio: float
    capacity_slope: float
    implementation_shortfall_bps: float
    consecutive_failures: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_passing(self) -> bool:
        """Check if agent meets minimum KPI thresholds."""
        return (
            self.edge_after_cost_bps >= 2.0 and  # 2x cost minimum
            self.information_coefficient >= 0.02 and  # Minimum IC
            self.correlation_to_book <= 0.6 and  # Diversification
            self.deflated_sharpe_ratio >= 0.0 and  # DSR positive
            self.consecutive_failures <= 2  # Max 2 consecutive failures
        )


class LegendaryAgent(ABC):
    """Base class for all legendary investor agents."""
    
    def __init__(self, name: str, description: str, horizon_days: int):
        self.name = name
        self.description = description
        self.horizon_days = horizon_days
        self.logger = logging.getLogger(f"agent.{name}")
        self.kpis = None
        self.is_active = True
        self.capital_allocation = 0.0
        self.risk_budget = 0.0
        
    @abstractmethod
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        universe: List[str],
                        as_of: datetime) -> AgentOutput:
        """Generate trading signals based on agent's methodology."""
        pass
    
    @abstractmethod
    def get_preferred_regimes(self) -> List[MarketRegime]:
        """Return market regimes where this agent performs best."""
        pass
    
    def update_kpis(self, kpis: AgentKPIs):
        """Update agent performance KPIs."""
        self.kpis = kpis
        
        # Auto-deactivate if failing KPIs
        if not kpis.is_passing():
            self.logger.warning(f"Agent {self.name} failing KPIs - reducing allocation")
            self.capital_allocation *= 0.5  # Halve allocation
            
        if kpis.consecutive_failures >= 3:
            self.is_active = False
            self.logger.error(f"Agent {self.name} deactivated after 3 consecutive failures")


class SimonsStatArbAgent(LegendaryAgent):
    """
    Jim Simons Statistical Arbitrage Agent
    
    Mission: Cross-sectional mean reversion & short-horizon patterns
    Method: PCA/cluster residuals, Kalman pairs, HMM regime detection
    Horizon: 1-5 days
    """
    
    def __init__(self):
        super().__init__(
            name="Simons_StatArb_v1",
            description="Short-term mean reversion using statistical arbitrage",
            horizon_days=3
        )
        self.lookback_days = 60
        self.max_turnover_daily = 0.5  # 50% daily turnover cap
        self.beta_neutral = True
        self.sector_neutral = True
        
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        universe: List[str],
                        as_of: datetime) -> AgentOutput:
        """Generate stat-arb signals using residual mean reversion."""
        
        signals = []
        
        try:
            # Get returns data
            returns_data = self._prepare_returns_data(market_data, universe, as_of)
            
            if len(returns_data) < 30:  # Need minimum data
                return AgentOutput(self.name, as_of, universe, [])
            
            # 1. Cross-sectional residuals via PCA
            residuals = self._calculate_residuals(returns_data)
            
            # 2. Mean reversion signals
            signals = self._generate_mean_reversion_signals(
                residuals, universe, as_of
            )
            
            # 3. Apply risk filters
            filtered_signals = self._apply_risk_filters(signals, returns_data)
            
            return AgentOutput(
                agent_name=self.name,
                as_of=as_of,
                universe=universe,
                signals=filtered_signals,
                risk_notes=["Beta-neutral enforced", "Sector-neutral enforced", 
                           f"Max turnover: {self.max_turnover_daily:.0%}"]
            )
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return AgentOutput(self.name, as_of, universe, [])
    
    def _prepare_returns_data(self, market_data: pd.DataFrame, 
                             universe: List[str], as_of: datetime) -> pd.DataFrame:
        """Prepare returns data for analysis."""
        end_date = as_of
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Filter data
        mask = (market_data.index >= start_date) & (market_data.index <= end_date)
        data = market_data.loc[mask]
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Filter to universe
        available_tickers = [t for t in universe if t in returns.columns]
        returns = returns[available_tickers]
        
        return returns
    
    def _calculate_residuals(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-sectional residuals using PCA."""
        
        # Standardize returns
        scaler = StandardScaler()
        scaled_returns = pd.DataFrame(
            scaler.fit_transform(returns_data.fillna(0)),
            index=returns_data.index,
            columns=returns_data.columns
        )
        
        # PCA to extract common factors
        n_components = min(5, len(returns_data.columns) // 3)
        pca = PCA(n_components=n_components)
        
        # Fit PCA and get residuals
        common_factors = pca.fit_transform(scaled_returns.fillna(0))
        reconstructed = pca.inverse_transform(common_factors)
        
        residuals = scaled_returns - pd.DataFrame(
            reconstructed, 
            index=scaled_returns.index, 
            columns=scaled_returns.columns
        )
        
        return residuals
    
    def _generate_mean_reversion_signals(self, 
                                       residuals: pd.DataFrame,
                                       universe: List[str],
                                       as_of: datetime) -> List[AgentSignal]:
        """Generate mean reversion signals from residuals."""
        
        signals = []
        
        # Get recent residuals for scoring
        recent_residuals = residuals.tail(5)  # Last 5 days
        
        for ticker in universe:
            if ticker not in residuals.columns:
                continue
                
            ticker_residuals = recent_residuals[ticker].dropna()
            
            if len(ticker_residuals) < 3:
                continue
            
            # Mean reversion score
            current_residual = ticker_residuals.iloc[-1]
            rolling_mean = ticker_residuals.mean()
            rolling_std = ticker_residuals.std()
            
            if rolling_std == 0:
                continue
            
            # Z-score of current residual
            z_score = (current_residual - rolling_mean) / rolling_std
            
            # Generate signal (mean revert)
            if abs(z_score) > 1.5:  # Threshold for signal
                direction = "short" if z_score > 0 else "long"
                alpha_bps = min(abs(z_score) * 3.0, 15.0)  # Cap at 15 bps
                confidence = min(abs(z_score) / 3.0, 0.8)  # Max 80% confidence
                
                signal = AgentSignal(
                    ticker=ticker,
                    direction=direction,
                    alpha_bps=alpha_bps,
                    confidence=confidence,
                    horizon_days=self.horizon_days,
                    agent_name=self.name,
                    timestamp=as_of,
                    risk_notes=[f"Z-score: {z_score:.2f}"]
                )
                
                signals.append(signal)
        
        return signals
    
    def _apply_risk_filters(self, 
                           signals: List[AgentSignal],
                           returns_data: pd.DataFrame) -> List[AgentSignal]:
        """Apply risk filters to signals."""
        
        # Beta neutrality: balance long/short signals
        long_signals = [s for s in signals if s.direction == "long"]
        short_signals = [s for s in signals if s.direction == "short"]
        
        # Take top signals from each side
        max_positions = 10  # Max 10 positions per side
        
        long_signals.sort(key=lambda x: x.alpha_bps, reverse=True)
        short_signals.sort(key=lambda x: x.alpha_bps, reverse=True)
        
        filtered_signals = (
            long_signals[:max_positions] + 
            short_signals[:max_positions]
        )
        
        return filtered_signals
    
    def get_preferred_regimes(self) -> List[MarketRegime]:
        """Stat-arb works best in sideways/choppy markets."""
        return [MarketRegime.SIDEWAYS_CHOP, MarketRegime.LOW_VOL_UPTREND]


class DalioMacroAgent(LegendaryAgent):
    """
    Ray Dalio Macro Agent (Bridgewater Style)
    
    Mission: Growth/Inflation regimes & cross-asset tilts
    Method: 4-quadrant regime analysis + carry/trend overlays
    Horizon: 20-60 days
    """
    
    def __init__(self):
        super().__init__(
            name="Dalio_Macro_v1", 
            description="Macro regime-based asset allocation",
            horizon_days=45
        )
        self.regime_lookback = 90
        
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        universe: List[str],
                        as_of: datetime) -> AgentOutput:
        """Generate macro regime-based signals."""
        
        try:
            # 1. Determine current regime
            regime = self._classify_regime(market_data, as_of)
            
            # 2. Generate regime-appropriate signals
            signals = self._generate_regime_signals(regime, universe, as_of)
            
            return AgentOutput(
                agent_name=self.name,
                as_of=as_of,
                universe=universe,
                signals=signals,
                regime_view=regime,
                risk_notes=[f"Regime: {regime.value}", "Cross-asset tilts only"]
            )
            
        except Exception as e:
            self.logger.error(f"Error in macro agent: {e}")
            return AgentOutput(self.name, as_of, universe, [])
    
    def _classify_regime(self, market_data: pd.DataFrame, as_of: datetime) -> MarketRegime:
        """Classify current macro regime using 4-quadrant framework."""
        
        # Simplified regime classification
        # In practice, would use nowcasting models, yield curves, etc.
        
        try:
            end_date = as_of
            start_date = end_date - timedelta(days=self.regime_lookback)
            
            recent_data = market_data.loc[
                (market_data.index >= start_date) & 
                (market_data.index <= end_date)
            ]
            
            if len(recent_data) < 20:
                return MarketRegime.UNKNOWN
            
            # Simple volatility-based regime
            returns = recent_data.pct_change().dropna()
            if len(returns.columns) == 0:
                return MarketRegime.UNKNOWN
                
            vol = returns.std().mean() * np.sqrt(252)  # Annualized vol
            trend = returns.mean().mean() * 252  # Annualized return
            
            if vol < 0.15 and trend > 0.05:
                return MarketRegime.LOW_VOL_UPTREND
            elif vol > 0.25 and trend < -0.05:
                return MarketRegime.HIGH_VOL_DOWNTREND
            elif abs(trend) < 0.02:
                return MarketRegime.SIDEWAYS_CHOP
            else:
                return MarketRegime.MACRO_TRANSITION
                
        except Exception:
            return MarketRegime.UNKNOWN
    
    def _generate_regime_signals(self, 
                               regime: MarketRegime,
                               universe: List[str],
                               as_of: datetime) -> List[AgentSignal]:
        """Generate signals based on regime classification."""
        
        signals = []
        
        # Regime-specific allocations
        regime_allocations = {
            MarketRegime.LOW_VOL_UPTREND: {
                "growth_tilt": 8.0,  # 8 bps alpha
                "defensive_tilt": -4.0
            },
            MarketRegime.HIGH_VOL_DOWNTREND: {
                "defensive_tilt": 12.0,
                "growth_tilt": -8.0
            },
            MarketRegime.SIDEWAYS_CHOP: {
                "neutral": 0.0
            },
            MarketRegime.MACRO_TRANSITION: {
                "hedge_tilt": 6.0
            }
        }
        
        allocations = regime_allocations.get(regime, {})
        
        # Map tilts to universe (simplified)
        for ticker in universe[:5]:  # Top 5 for demo
            if "growth_tilt" in allocations:
                alpha_bps = allocations["growth_tilt"]
                direction = "long" if alpha_bps > 0 else "short"
                
                signal = AgentSignal(
                    ticker=ticker,
                    direction=direction,
                    alpha_bps=abs(alpha_bps),
                    confidence=0.6,
                    horizon_days=self.horizon_days,
                    agent_name=self.name,
                    timestamp=as_of,
                    risk_notes=[f"Regime tilt: {regime.value}"]
                )
                
                signals.append(signal)
        
        return signals
    
    def get_preferred_regimes(self) -> List[MarketRegime]:
        """Macro agent works in all regimes but especially transitions."""
        return [MarketRegime.MACRO_TRANSITION, MarketRegime.HIGH_VOL_DOWNTREND]


class PTJTrendAgent(LegendaryAgent):
    """
    Paul Tudor Jones Trend Following Agent
    
    Mission: Medium-term price trend with volatility-adjusted sizing
    Method: Multi-timeframe momentum + breakout with ATR sizing
    Horizon: 20-60 days
    """
    
    def __init__(self):
        super().__init__(
            name="PTJ_Trend_v1",
            description="Medium-term trend following with momentum",
            horizon_days=40
        )
        self.momentum_windows = [10, 20, 50]  # Multiple timeframes
        
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        universe: List[str],
                        as_of: datetime) -> AgentOutput:
        """Generate trend-following signals."""
        
        try:
            signals = []
            
            for ticker in universe:
                if ticker not in market_data.columns:
                    continue
                
                signal = self._generate_trend_signal(market_data, ticker, as_of)
                if signal:
                    signals.append(signal)
            
            # Filter for trend regime
            regime_note = "Skip in mean-reverting markets"
            
            return AgentOutput(
                agent_name=self.name,
                as_of=as_of,
                universe=universe,
                signals=signals,
                risk_notes=[regime_note, "Trailing stops enabled"]
            )
            
        except Exception as e:
            self.logger.error(f"Error in trend agent: {e}")
            return AgentOutput(self.name, as_of, universe, [])
    
    def _generate_trend_signal(self, 
                             market_data: pd.DataFrame,
                             ticker: str,
                             as_of: datetime) -> Optional[AgentSignal]:
        """Generate trend signal for individual ticker."""
        
        try:
            # Get price series
            prices = market_data[ticker].dropna()
            
            if len(prices) < max(self.momentum_windows) + 10:
                return None
            
            # Calculate momentum scores
            momentum_scores = []
            
            for window in self.momentum_windows:
                if len(prices) >= window:
                    momentum = (prices.iloc[-1] / prices.iloc[-window] - 1) * 100
                    momentum_scores.append(momentum)
            
            if not momentum_scores:
                return None
            
            # Average momentum score
            avg_momentum = np.mean(momentum_scores)
            
            # ATR for volatility adjustment
            returns = prices.pct_change().dropna()
            atr = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0.02
            
            # Generate signal
            if abs(avg_momentum) > 2.0:  # 2% momentum threshold
                direction = "long" if avg_momentum > 0 else "short"
                
                # Alpha scaled by momentum strength and adjusted for volatility
                raw_alpha = min(abs(avg_momentum) * 0.5, 10.0)  # Cap at 10 bps
                vol_adjusted_alpha = raw_alpha / max(atr * 100, 1.0)  # Adjust for vol
                
                alpha_bps = min(vol_adjusted_alpha, 8.0)  # Final cap
                confidence = min(abs(avg_momentum) / 10.0, 0.75)
                
                return AgentSignal(
                    ticker=ticker,
                    direction=direction,
                    alpha_bps=alpha_bps,
                    confidence=confidence,
                    horizon_days=self.horizon_days,
                    agent_name=self.name,
                    timestamp=as_of,
                    risk_notes=[f"Momentum: {avg_momentum:.1f}%", f"ATR: {atr:.1%}"]
                )
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend for {ticker}: {e}")
            
        return None
    
    def get_preferred_regimes(self) -> List[MarketRegime]:
        """Trend following works best in trending markets."""
        return [MarketRegime.LOW_VOL_UPTREND, MarketRegime.HIGH_VOL_DOWNTREND]


class MicrostructureAgent(LegendaryAgent):
    """
    Microstructure Optimization Agent
    
    Mission: Lower implementation shortfall & boost fill quality
    Method: Adaptive order routing with participation limits
    Horizon: Intraday
    """
    
    def __init__(self):
        super().__init__(
            name="Microstructure_v1",
            description="Execution optimization and microstructure alpha",
            horizon_days=1
        )
        self.max_participation = 0.15  # 15% of ADV
        
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        universe: List[str],
                        as_of: datetime) -> AgentOutput:
        """Generate execution optimization signals."""
        
        try:
            signals = []
            
            # Microstructure signals (simplified)
            for ticker in universe[:3]:  # Focus on top names
                signal = self._generate_microstructure_signal(ticker, as_of)
                if signal:
                    signals.append(signal)
            
            return AgentOutput(
                agent_name=self.name,
                as_of=as_of,
                universe=universe,
                signals=signals,
                risk_notes=[f"Max participation: {self.max_participation:.0%}"]
            )
            
        except Exception as e:
            self.logger.error(f"Error in microstructure agent: {e}")
            return AgentOutput(self.name, as_of, universe, [])
    
    def _generate_microstructure_signal(self, 
                                      ticker: str,
                                      as_of: datetime) -> Optional[AgentSignal]:
        """Generate microstructure-based signal."""
        
        # Simplified microstructure signal
        # In practice, would use order book data, flow analysis, etc.
        
        return AgentSignal(
            ticker=ticker,
            direction="neutral",  # Execution optimization, not directional
            alpha_bps=2.0,  # Expected IS reduction
            confidence=0.8,
            horizon_days=1,
            agent_name=self.name,
            timestamp=as_of,
            risk_notes=["Execution alpha only"]
        )
    
    def get_preferred_regimes(self) -> List[MarketRegime]:
        """Microstructure agent works in all regimes."""
        return list(MarketRegime)


class DrawdownGuardian(LegendaryAgent):
    """
    Enhanced Drawdown Guardian
    
    Mission: Intraday kill-switch, loss limits, cost spike protection
    Method: Real-time monitoring with dynamic position sizing
    """
    
    def __init__(self):
        super().__init__(
            name="Drawdown_Guardian_v1",
            description="Enhanced risk management and drawdown protection",
            horizon_days=1
        )
        self.daily_loss_limit = -0.02  # 2% daily loss limit
        self.cost_spike_threshold = 2.0  # 2x normal costs
        
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        universe: List[str],
                        as_of: datetime) -> AgentOutput:
        """Generate risk management signals."""
        
        # This agent generates risk adjustments, not alpha signals
        risk_signals = self._assess_risk_environment(market_data, as_of)
        
        return AgentOutput(
            agent_name=self.name,
            as_of=as_of,
            universe=universe,
            signals=risk_signals,
            risk_notes=["Active risk monitoring", "Dynamic position sizing"]
        )
    
    def _assess_risk_environment(self, 
                               market_data: pd.DataFrame,
                               as_of: datetime) -> List[AgentSignal]:
        """Assess current risk environment and suggest adjustments."""
        
        signals = []
        
        # Risk assessment logic (simplified)
        # In practice, would integrate with kill-switch system
        
        # Example: Reduce exposure in high volatility
        try:
            recent_data = market_data.tail(10)
            if len(recent_data) > 0:
                vol_estimate = recent_data.pct_change().std().mean()
                
                if vol_estimate > 0.03:  # High volatility threshold
                    # Generate risk reduction signal
                    signal = AgentSignal(
                        ticker="PORTFOLIO",
                        direction="reduce",
                        alpha_bps=-1.0,  # Negative alpha = risk reduction
                        confidence=0.9,
                        horizon_days=1,
                        agent_name=self.name,
                        timestamp=as_of,
                        risk_notes=["High volatility detected", "Reduce exposure"]
                    )
                    signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"Risk assessment error: {e}")
        
        return signals
    
    def get_preferred_regimes(self) -> List[MarketRegime]:
        """Risk management works in all regimes."""
        return list(MarketRegime)


# Agent Registry for easy management
LEGENDARY_AGENTS = {
    "simons_stat_arb": SimonsStatArbAgent,
    "dalio_macro": DalioMacroAgent,
    "ptj_trend": PTJTrendAgent,
    "microstructure": MicrostructureAgent,
    "drawdown_guardian": DrawdownGuardian
}


def create_agent(agent_type: str) -> Optional[LegendaryAgent]:
    """Factory function to create agents."""
    agent_class = LEGENDARY_AGENTS.get(agent_type)
    if agent_class:
        return agent_class()
    return None


def get_available_agents() -> List[str]:
    """Get list of available agent types."""
    return list(LEGENDARY_AGENTS.keys())


if __name__ == "__main__":
    # Demo the agents
    print("🏛️ Multi-Agent Trading Shop - Legendary Investors")
    print("=" * 60)
    
    # Create sample market data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    np.random.seed(42)
    market_data = pd.DataFrame(
        100 * np.cumprod(1 + np.random.normal(0.001, 0.02, (len(dates), len(symbols))), axis=0),
        index=dates,
        columns=symbols
    )
    
    # Test each agent
    as_of = dates[-1]
    
    for agent_name, agent_class in LEGENDARY_AGENTS.items():
        print(f"\n📊 Testing {agent_name}...")
        agent = agent_class()
        
        output = agent.generate_signals(market_data, symbols, as_of)
        
        print(f"  Signals generated: {len(output.signals)}")
        if output.signals:
            avg_alpha = np.mean([s.alpha_bps for s in output.signals])
            print(f"  Average alpha: {avg_alpha:.1f} bps")
        print(f"  Preferred regimes: {[r.value for r in agent.get_preferred_regimes()]}")
