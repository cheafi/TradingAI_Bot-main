"""
Decision Gateway - Multi-Agent Orchestration System
=================================================

This module implements the central decision-making system that:
- Validates and normalizes agent signals
- Applies ROI-focused filtering (alpha >= cost threshold)
- Manages regime-based capital allocation
- Tracks agent performance and auto-adjusts allocation
- Implements position limits and risk controls

Universal Agent Interface:
- All agents emit standardized alpha_bps signals
- Trade only if alpha_bps >= κ × (fees + slippage + borrow)
- Automatic agent promotion/demotion based on KPIs
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging
import json
from collections import defaultdict

# Import our agent system
from apps.agents.legendary_investors import (
    LegendaryAgent, AgentSignal, AgentOutput, AgentKPIs, MarketRegime,
    LEGENDARY_AGENTS, create_agent
)

# Import cost model for ROI calculations
try:
    from trading_platform.execution.cost_model import (
        RealisticCostModel, OrderContext, AssetClass, OrderType, MarketCondition
    )
except ImportError:
    # Fallback if cost model not available
    class MockCostModel:
        def calculate_costs(self, order_context):
            # Return simple cost estimate
            class MockCosts:
                def __init__(self):
                    self.total_cost = order_context.notional * 0.0010  # 10 bps
            return MockCosts()
    RealisticCostModel = MockCostModel


class EnsembleMethod(Enum):
    """Methods for combining agent signals."""
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VARIANCE = "inverse_variance"
    PERFORMANCE_WEIGHT = "performance_weight"
    BAYESIAN_MODEL_AVERAGE = "bayesian_model_average"


@dataclass
class PortfolioSignal:
    """Final portfolio signal after ensemble."""
    ticker: str
    direction: str
    target_weight: float
    expected_alpha_bps: float
    confidence: float
    contributing_agents: List[str]
    risk_budget: float
    horizon_days: int
    timestamp: datetime


@dataclass
class DecisionGatewayConfig:
    """Configuration for the decision gateway."""
    
    # ROI thresholds
    min_alpha_to_cost_ratio: float = 2.0  # Minimum 2x cost coverage
    cost_buffer_multiplier: float = 1.5   # Safety buffer on costs
    
    # Agent allocation limits
    max_agent_allocation: float = 0.35    # Max 35% to any single agent
    min_agent_allocation: float = 0.05    # Min 5% for active agents
    
    # Portfolio constraints
    max_portfolio_turnover_daily: float = 0.15  # 15% daily turnover
    max_single_position: float = 0.08     # 8% max position size
    max_sector_allocation: float = 0.25   # 25% max per sector
    
    # Risk management
    portfolio_vol_target: float = 0.12    # 12% annual volatility target
    max_drawdown_limit: float = -0.08     # 8% max drawdown
    
    # Ensemble settings
    ensemble_method: EnsembleMethod = EnsembleMethod.PERFORMANCE_WEIGHT
    min_agents_for_signal: int = 2        # Need 2+ agents for trade
    
    # Regime allocation matrix
    regime_allocations: Dict[MarketRegime, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize regime allocation matrix."""
        if not self.regime_allocations:
            self.regime_allocations = {
                MarketRegime.LOW_VOL_UPTREND: {
                    "ptj_trend": 0.25,
                    "simons_stat_arb": 0.20,
                    "dalio_macro": 0.15,
                    "microstructure": 0.10,
                    "drawdown_guardian": 0.30
                },
                MarketRegime.HIGH_VOL_DOWNTREND: {
                    "dalio_macro": 0.30,
                    "drawdown_guardian": 0.35,
                    "microstructure": 0.15,
                    "simons_stat_arb": 0.10,
                    "ptj_trend": 0.10
                },
                MarketRegime.SIDEWAYS_CHOP: {
                    "simons_stat_arb": 0.35,
                    "microstructure": 0.25,
                    "drawdown_guardian": 0.25,
                    "dalio_macro": 0.10,
                    "ptj_trend": 0.05
                },
                MarketRegime.MACRO_TRANSITION: {
                    "dalio_macro": 0.40,
                    "drawdown_guardian": 0.30,
                    "microstructure": 0.15,
                    "simons_stat_arb": 0.10,
                    "ptj_trend": 0.05
                }
            }


class RegimeClassifier:
    """Market regime classification system."""
    
    def __init__(self):
        self.logger = logging.getLogger("regime_classifier")
        
    def classify_regime(self, 
                       market_data: pd.DataFrame,
                       as_of: datetime) -> MarketRegime:
        """Classify current market regime."""
        
        try:
            # Get recent data for analysis
            end_date = as_of
            start_date = end_date - timedelta(days=60)
            
            recent_data = market_data.loc[
                (market_data.index >= start_date) & 
                (market_data.index <= end_date)
            ]
            
            if len(recent_data) < 20:
                return MarketRegime.UNKNOWN
            
            # Calculate regime indicators
            returns = recent_data.pct_change().dropna()
            
            # Average volatility across assets
            vol = returns.std().mean() * np.sqrt(252)
            
            # Average trend across assets
            trend = returns.mean().mean() * 252
            
            # Regime classification logic
            if vol < 0.15:  # Low volatility
                if trend > 0.05:
                    return MarketRegime.LOW_VOL_UPTREND
                else:
                    return MarketRegime.SIDEWAYS_CHOP
            else:  # High volatility
                if trend < -0.05:
                    return MarketRegime.HIGH_VOL_DOWNTREND
                else:
                    return MarketRegime.MACRO_TRANSITION
                    
        except Exception as e:
            self.logger.error(f"Regime classification error: {e}")
            return MarketRegime.UNKNOWN


class AgentPerformanceTracker:
    """Track and evaluate agent performance for KPI-based allocation."""
    
    def __init__(self):
        self.agent_history: Dict[str, List[Dict]] = defaultdict(list)
        self.performance_window = 30  # 30-day rolling window
        self.logger = logging.getLogger("performance_tracker")
        
    def record_agent_performance(self,
                                agent_name: str,
                                signals: List[AgentSignal],
                                actual_returns: Dict[str, float],
                                costs: Dict[str, float],
                                timestamp: datetime):
        """Record agent performance for KPI calculation."""
        
        # Calculate signal accuracy and profitability
        for signal in signals:
            ticker = signal.ticker
            predicted_return = signal.alpha_bps / 10000  # Convert to decimal
            actual_return = actual_returns.get(ticker, 0.0)
            cost = costs.get(ticker, 0.0)
            
            # Direction accuracy
            direction_correct = (
                (signal.direction == "long" and actual_return > 0) or
                (signal.direction == "short" and actual_return < 0)
            )
            
            # Net P&L after costs
            net_pnl = actual_return - cost if signal.direction == "long" else -actual_return - cost
            
            performance_record = {
                "timestamp": timestamp,
                "ticker": ticker,
                "predicted_alpha_bps": signal.alpha_bps,
                "actual_return": actual_return,
                "cost": cost,
                "net_pnl": net_pnl,
                "direction_correct": direction_correct,
                "confidence": signal.confidence,
                "horizon_days": signal.horizon_days
            }
            
            self.agent_history[agent_name].append(performance_record)
        
        # Keep only recent history
        cutoff_date = timestamp - timedelta(days=self.performance_window * 2)
        self.agent_history[agent_name] = [
            record for record in self.agent_history[agent_name]
            if record["timestamp"] > cutoff_date
        ]
    
    def calculate_agent_kpis(self, agent_name: str) -> Optional[AgentKPIs]:
        """Calculate comprehensive KPIs for an agent."""
        
        if agent_name not in self.agent_history:
            return None
            
        history = self.agent_history[agent_name]
        
        if len(history) < 10:  # Need minimum history
            return None
        
        try:
            # Extract metrics
            predicted_alphas = [h["predicted_alpha_bps"] for h in history]
            actual_returns = [h["actual_return"] * 10000 for h in history]  # To bps
            net_pnls = [h["net_pnl"] * 10000 for h in history]  # To bps
            costs = [h["cost"] * 10000 for h in history]  # To bps
            
            # Information Coefficient
            ic = np.corrcoef(predicted_alphas, actual_returns)[0, 1] if len(predicted_alphas) > 1 else 0.0
            ic = ic if not np.isnan(ic) else 0.0
            
            # Information Ratio
            alpha_errors = np.array(predicted_alphas) - np.array(actual_returns)
            ir = -np.mean(alpha_errors) / (np.std(alpha_errors) + 1e-8)
            
            # Edge after cost
            edge_after_cost = np.mean(net_pnls)
            
            # Half-life (simplified)
            half_life = np.mean([h["horizon_days"] for h in history])
            
            # Deflated Sharpe Ratio (simplified)
            if len(net_pnls) > 1:
                sharpe = np.mean(net_pnls) / (np.std(net_pnls) + 1e-8)
                # Apply deflation factor (simplified)
                dsr = sharpe * np.sqrt(max(1 - 0.1 * len(net_pnls) / 252, 0.1))
            else:
                dsr = 0.0
            
            # Implementation shortfall
            is_bps = np.mean(costs)
            
            return AgentKPIs(
                agent_name=agent_name,
                edge_after_cost_bps=edge_after_cost,
                information_coefficient=ic,
                information_ratio=ir,
                half_life_days=half_life,
                correlation_to_book=0.3,  # Placeholder - would calculate vs portfolio
                deflated_sharpe_ratio=dsr,
                capacity_slope=1.0,  # Placeholder
                implementation_shortfall_bps=is_bps,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating KPIs for {agent_name}: {e}")
            return None


class DecisionGateway:
    """
    Central decision-making system for multi-agent trading.
    
    Orchestrates all agents, applies ROI filtering, manages regime-based
    allocation, and ensures signals meet minimum profitability thresholds.
    """
    
    def __init__(self, config: Optional[DecisionGatewayConfig] = None):
        self.config = config or DecisionGatewayConfig()
        self.logger = logging.getLogger("decision_gateway")
        
        # Core systems
        self.regime_classifier = RegimeClassifier()
        self.performance_tracker = AgentPerformanceTracker()
        self.cost_model = RealisticCostModel()
        
        # Agent management
        self.active_agents: Dict[str, LegendaryAgent] = {}
        self.agent_allocations: Dict[str, float] = {}
        self.agent_kpis: Dict[str, AgentKPIs] = {}
        
        # Portfolio state
        self.current_positions: Dict[str, float] = {}
        self.current_regime = MarketRegime.UNKNOWN
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all available agents."""
        for agent_type in LEGENDARY_AGENTS.keys():
            try:
                agent = create_agent(agent_type)
                if agent:
                    self.active_agents[agent_type] = agent
                    self.agent_allocations[agent_type] = self.config.min_agent_allocation
                    self.logger.info(f"Initialized agent: {agent_type}")
            except Exception as e:
                self.logger.error(f"Failed to initialize agent {agent_type}: {e}")
    
    def generate_portfolio_signals(self,
                                 market_data: pd.DataFrame,
                                 universe: List[str],
                                 as_of: datetime) -> List[PortfolioSignal]:
        """
        Main method: Generate portfolio signals from all agents.
        
        Process:
        1. Classify market regime
        2. Adjust agent allocations based on regime
        3. Collect signals from all active agents
        4. Apply ROI filtering (alpha >= cost threshold)
        5. Ensemble signals using performance weighting
        6. Apply portfolio constraints
        """
        
        try:
            # 1. Classify market regime
            self.current_regime = self.regime_classifier.classify_regime(market_data, as_of)
            self.logger.info(f"Current regime: {self.current_regime.value}")
            
            # 2. Update agent allocations based on regime
            self._update_regime_allocations()
            
            # 3. Collect signals from all active agents
            all_agent_outputs = self._collect_agent_signals(market_data, universe, as_of)
            
            # 4. Apply ROI filtering
            filtered_signals = self._apply_roi_filter(all_agent_outputs, universe)
            
            # 5. Ensemble signals
            portfolio_signals = self._ensemble_signals(filtered_signals, universe, as_of)
            
            # 6. Apply portfolio constraints
            final_signals = self._apply_portfolio_constraints(portfolio_signals)
            
            self.logger.info(f"Generated {len(final_signals)} portfolio signals")
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio signals: {e}")
            return []
    
    def _update_regime_allocations(self):
        """Update agent allocations based on current regime."""
        regime_allocs = self.config.regime_allocations.get(self.current_regime, {})
        
        for agent_name in self.active_agents.keys():
            if agent_name in regime_allocs:
                self.agent_allocations[agent_name] = regime_allocs[agent_name]
            else:
                # Default allocation for agents not in regime matrix
                self.agent_allocations[agent_name] = self.config.min_agent_allocation
        
        # Normalize to ensure sum = 1
        total_allocation = sum(self.agent_allocations.values())
        if total_allocation > 0:
            for agent_name in self.agent_allocations:
                self.agent_allocations[agent_name] /= total_allocation
    
    def _collect_agent_signals(self,
                             market_data: pd.DataFrame,
                             universe: List[str],
                             as_of: datetime) -> Dict[str, AgentOutput]:
        """Collect signals from all active agents."""
        
        agent_outputs = {}
        
        for agent_name, agent in self.active_agents.items():
            if not agent.is_active:
                continue
                
            try:
                output = agent.generate_signals(market_data, universe, as_of)
                if output.signals:  # Only include if signals generated
                    agent_outputs[agent_name] = output
                    self.logger.debug(f"Agent {agent_name} generated {len(output.signals)} signals")
            except Exception as e:
                self.logger.error(f"Error getting signals from {agent_name}: {e}")
        
        return agent_outputs
    
    def _apply_roi_filter(self,
                         agent_outputs: Dict[str, AgentOutput],
                         universe: List[str]) -> Dict[str, List[AgentSignal]]:
        """Apply ROI filtering: trade only if alpha >= cost threshold."""
        
        filtered_signals = {}
        
        for agent_name, output in agent_outputs.items():
            agent_filtered = []
            
            for signal in output.signals:
                # Estimate trading costs
                estimated_cost = self._estimate_trading_cost(signal, universe)
                
                # Apply ROI threshold
                required_alpha = estimated_cost * self.config.min_alpha_to_cost_ratio
                required_alpha *= self.config.cost_buffer_multiplier  # Safety buffer
                
                if signal.alpha_bps >= required_alpha:
                    agent_filtered.append(signal)
                    self.logger.debug(
                        f"{signal.ticker}: alpha={signal.alpha_bps:.1f} >= "
                        f"threshold={required_alpha:.1f}"
                    )
                else:
                    self.logger.debug(
                        f"{signal.ticker}: rejected - alpha={signal.alpha_bps:.1f} < "
                        f"threshold={required_alpha:.1f}"
                    )
            
            if agent_filtered:
                filtered_signals[agent_name] = agent_filtered
        
        return filtered_signals
    
    def _estimate_trading_cost(self, signal: AgentSignal, universe: List[str]) -> float:
        """Estimate trading cost in basis points."""
        
        try:
            # Simplified cost estimation
            # In practice, would use full cost model with market data
            
            # Base cost estimates by signal characteristics
            base_cost = 5.0  # 5 bps base
            
            # Adjust for urgency/horizon
            if signal.horizon_days <= 1:
                base_cost *= 1.5  # Higher cost for urgent trades
            elif signal.horizon_days >= 20:
                base_cost *= 0.8  # Lower cost for patient trades
            
            # Adjust for direction
            if signal.direction == "short":
                base_cost *= 1.3  # Higher cost for shorts (borrow, etc.)
            
            return base_cost
            
        except Exception as e:
            self.logger.warning(f"Cost estimation error: {e}")
            return 8.0  # Default 8 bps cost
    
    def _ensemble_signals(self,
                         filtered_signals: Dict[str, List[AgentSignal]],
                         universe: List[str],
                         as_of: datetime) -> List[PortfolioSignal]:
        """Ensemble signals from multiple agents using performance weighting."""
        
        portfolio_signals = []
        
        # Group signals by ticker
        ticker_signals = defaultdict(list)
        for agent_name, signals in filtered_signals.items():
            for signal in signals:
                ticker_signals[signal.ticker].append((agent_name, signal))
        
        # Process each ticker
        for ticker, agent_signal_pairs in ticker_signals.items():
            if len(agent_signal_pairs) < self.config.min_agents_for_signal:
                continue  # Need minimum agents for consensus
            
            # Get agent weights
            agent_weights = self._get_agent_weights(
                [pair[0] for pair in agent_signal_pairs]
            )
            
            # Ensemble the signals
            portfolio_signal = self._create_ensemble_signal(
                agent_signal_pairs, agent_weights, ticker, as_of
            )
            
            if portfolio_signal:
                portfolio_signals.append(portfolio_signal)
        
        return portfolio_signals
    
    def _get_agent_weights(self, agent_names: List[str]) -> Dict[str, float]:
        """Get performance-based weights for agents."""
        
        weights = {}
        
        if self.config.ensemble_method == EnsembleMethod.EQUAL_WEIGHT:
            # Equal weighting
            weight = 1.0 / len(agent_names)
            for name in agent_names:
                weights[name] = weight
                
        elif self.config.ensemble_method == EnsembleMethod.PERFORMANCE_WEIGHT:
            # Performance-based weighting
            total_performance = 0.0
            performance_scores = {}
            
            for name in agent_names:
                kpis = self.agent_kpis.get(name)
                if kpis and kpis.is_passing():
                    # Performance score based on edge after cost and IC
                    score = max(kpis.edge_after_cost_bps * kpis.information_coefficient, 0.1)
                else:
                    score = 0.1  # Minimum weight for new/poor agents
                
                performance_scores[name] = score
                total_performance += score
            
            # Normalize weights
            for name in agent_names:
                weights[name] = performance_scores[name] / max(total_performance, 1e-8)
        
        else:
            # Fallback to equal weight
            weight = 1.0 / len(agent_names)
            for name in agent_names:
                weights[name] = weight
        
        return weights
    
    def _create_ensemble_signal(self,
                              agent_signal_pairs: List[Tuple[str, AgentSignal]],
                              agent_weights: Dict[str, float],
                              ticker: str,
                              as_of: datetime) -> Optional[PortfolioSignal]:
        """Create ensemble signal from multiple agent signals."""
        
        try:
            # Weighted combination of signals
            total_alpha = 0.0
            total_confidence = 0.0
            avg_horizon = 0.0
            contributing_agents = []
            
            long_weight = 0.0
            short_weight = 0.0
            
            for agent_name, signal in agent_signal_pairs:
                weight = agent_weights.get(agent_name, 0.0)
                
                # Aggregate alpha and confidence
                total_alpha += signal.alpha_bps * weight
                total_confidence += signal.confidence * weight
                avg_horizon += signal.horizon_days * weight
                contributing_agents.append(agent_name)
                
                # Track directional weights
                if signal.direction == "long":
                    long_weight += weight
                elif signal.direction == "short":
                    short_weight += weight
            
            # Determine final direction
            if long_weight > short_weight:
                direction = "long"
                net_weight = long_weight - short_weight
            elif short_weight > long_weight:
                direction = "short"
                net_weight = short_weight - long_weight
            else:
                return None  # No consensus
            
            # Calculate target weight (scaled by regime allocation)
            base_weight = min(net_weight * 0.05, self.config.max_single_position)  # 5% base
            
            return PortfolioSignal(
                ticker=ticker,
                direction=direction,
                target_weight=base_weight,
                expected_alpha_bps=total_alpha,
                confidence=total_confidence,
                contributing_agents=contributing_agents,
                risk_budget=base_weight * total_confidence,
                horizon_days=int(avg_horizon),
                timestamp=as_of
            )
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble signal for {ticker}: {e}")
            return None
    
    def _apply_portfolio_constraints(self,
                                   signals: List[PortfolioSignal]) -> List[PortfolioSignal]:
        """Apply portfolio-level constraints and risk limits."""
        
        if not signals:
            return signals
        
        # Sort by expected alpha (best first)
        signals.sort(key=lambda x: x.expected_alpha_bps, reverse=True)
        
        # Apply position size limits
        total_weight = 0.0
        constrained_signals = []
        
        for signal in signals:
            # Check if adding this position exceeds limits
            if total_weight + signal.target_weight <= 1.0:  # Don't exceed 100%
                constrained_signals.append(signal)
                total_weight += signal.target_weight
            else:
                # Reduce size to fit
                remaining_capacity = 1.0 - total_weight
                if remaining_capacity > 0.01:  # Minimum 1% position
                    signal.target_weight = remaining_capacity
                    constrained_signals.append(signal)
                    break
        
        # Apply turnover constraints
        # (Would compare against current positions in practice)
        
        return constrained_signals
    
    def update_agent_performance(self,
                               actual_returns: Dict[str, float],
                               costs: Dict[str, float],
                               timestamp: datetime):
        """Update agent performance tracking."""
        
        for agent_name, agent in self.active_agents.items():
            # Get recent signals for this agent
            # (In practice, would track signals by timestamp)
            
            # Update KPIs
            kpis = self.performance_tracker.calculate_agent_kpis(agent_name)
            if kpis:
                self.agent_kpis[agent_name] = kpis
                agent.update_kpis(kpis)
                
                # Log performance
                self.logger.info(
                    f"Agent {agent_name}: Edge={kpis.edge_after_cost_bps:.1f}bps, "
                    f"IC={kpis.information_coefficient:.3f}, "
                    f"Passing={kpis.is_passing()}"
                )
    
    def get_agent_status(self) -> Dict[str, Dict]:
        """Get status of all agents."""
        
        status = {}
        
        for agent_name, agent in self.active_agents.items():
            agent_status = {
                "is_active": agent.is_active,
                "allocation": self.agent_allocations.get(agent_name, 0.0),
                "preferred_regimes": [r.value for r in agent.get_preferred_regimes()],
                "description": agent.description
            }
            
            # Add KPIs if available
            kpis = self.agent_kpis.get(agent_name)
            if kpis:
                agent_status.update({
                    "edge_after_cost_bps": kpis.edge_after_cost_bps,
                    "information_coefficient": kpis.information_coefficient,
                    "is_passing_kpis": kpis.is_passing()
                })
            
            status[agent_name] = agent_status
        
        return status


if __name__ == "__main__":
    # Demo the Decision Gateway
    print("🎯 Decision Gateway - Multi-Agent Orchestration")
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
    
    # Initialize Decision Gateway
    gateway = DecisionGateway()
    
    # Generate portfolio signals
    as_of = dates[-1]
    portfolio_signals = gateway.generate_portfolio_signals(market_data, symbols, as_of)
    
    print(f"\n📊 Generated {len(portfolio_signals)} portfolio signals")
    print(f"Current regime: {gateway.current_regime.value}")
    
    # Show signals
    for signal in portfolio_signals:
        print(f"\n🎯 {signal.ticker}:")
        print(f"  Direction: {signal.direction}")
        print(f"  Target weight: {signal.target_weight:.1%}")
        print(f"  Expected alpha: {signal.expected_alpha_bps:.1f} bps")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Contributing agents: {', '.join(signal.contributing_agents)}")
    
    # Show agent status
    print(f"\n👥 Agent Status:")
    status = gateway.get_agent_status()
    for agent_name, info in status.items():
        print(f"  {agent_name}: Active={info['is_active']}, Allocation={info['allocation']:.1%}")
