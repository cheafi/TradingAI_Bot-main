"""
Agent Onboarding & Promotion System
===================================

This module implements the 2-week onboarding process for new agents:
- Sandbox testing with paper money
- KPI monitoring and evaluation
- Automatic promotion/demotion based on performance
- Risk-adjusted capacity scaling

Process:
1. New Agent → 2-week sandbox with minimal allocation
2. Daily KPI evaluation against benchmarks
3. Passing agents → Promotion to higher allocation
4. Failing agents → Demotion or deactivation
5. Continuous monitoring and rebalancing
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging
import json
from collections import defaultdict, deque

from apps.agents.legendary_investors import (
    LegendaryAgent, AgentKPIs, MarketRegime, LEGENDARY_AGENTS, create_agent
)


class AgentStatus(Enum):
    """Agent lifecycle status."""
    CANDIDATE = "candidate"           # New agent, not yet tested
    SANDBOX = "sandbox"              # 2-week evaluation period
    PROBATION = "probation"          # Failed first evaluation, second chance
    ACTIVE = "active"                # Passed evaluation, production ready
    VETERAN = "veteran"              # Long-term successful agent
    SUSPENDED = "suspended"          # Temporarily deactivated
    RETIRED = "retired"              # Permanently deactivated


@dataclass
class AgentEvaluation:
    """Agent evaluation results."""
    agent_name: str
    evaluation_period: Tuple[datetime, datetime]
    status: AgentStatus
    
    # Performance metrics
    total_trades: int
    win_rate: float
    avg_alpha_bps: float
    sharpe_ratio: float
    max_drawdown: float
    
    # KPI assessment
    edge_after_cost_bps: float
    information_coefficient: float
    implementation_shortfall_bps: float
    
    # Risk metrics
    vol_realized: float
    var_95: float
    correlation_to_book: float
    
    # Capacity metrics
    avg_position_size: float
    turnover_daily: float
    
    # Final scores
    profitability_score: float  # 0-100
    risk_score: float          # 0-100
    execution_score: float     # 0-100
    overall_score: float       # 0-100
    
    # Decision
    recommendation: str
    new_allocation: float
    reason: str
    
    timestamp: datetime


@dataclass
class OnboardingConfig:
    """Configuration for agent onboarding process."""
    
    # Evaluation periods
    sandbox_days: int = 14               # 2-week evaluation
    probation_days: int = 10             # 10-day second chance
    review_frequency_days: int = 7       # Weekly reviews
    
    # Initial allocations
    sandbox_allocation: float = 0.01     # 1% for new agents
    probation_allocation: float = 0.005  # 0.5% for probation
    min_active_allocation: float = 0.02  # 2% minimum for active
    max_active_allocation: float = 0.15  # 15% maximum for any agent
    
    # Performance thresholds
    min_edge_after_cost: float = 3.0     # 3 bps minimum edge
    min_information_coefficient: float = 0.05  # 5% IC threshold
    max_implementation_shortfall: float = 8.0  # 8 bps max IS
    min_sharpe_ratio: float = 0.5        # 0.5 Sharpe minimum
    max_drawdown_limit: float = -0.05    # 5% max drawdown
    
    # Risk thresholds
    max_correlation_to_book: float = 0.7  # 70% max correlation
    max_volatility: float = 0.25         # 25% max vol
    min_trade_frequency: int = 5         # 5 trades minimum in evaluation
    
    # Scoring weights
    profitability_weight: float = 0.4    # 40% weight on profits
    risk_weight: float = 0.3            # 30% weight on risk management
    execution_weight: float = 0.3       # 30% weight on execution quality
    
    # Promotion thresholds
    sandbox_pass_score: float = 70.0     # 70% to pass sandbox
    probation_pass_score: float = 75.0   # 75% to pass probation
    veteran_promotion_score: float = 85.0 # 85% for veteran status
    
    # Capacity scaling
    capacity_multiplier_active: float = 1.5   # 50% increase for active
    capacity_multiplier_veteran: float = 2.0  # 100% increase for veteran


class TradeSimulator:
    """Simulate trades for performance evaluation."""
    
    def __init__(self):
        self.logger = logging.getLogger("trade_simulator")
        
    def simulate_agent_trades(self,
                            agent: LegendaryAgent,
                            market_data: pd.DataFrame,
                            start_date: datetime,
                            end_date: datetime,
                            initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Simulate agent trading over evaluation period."""
        
        # Get trading dates
        trading_dates = market_data.loc[start_date:end_date].index
        
        if len(trading_dates) < 5:
            return self._empty_simulation_result()
        
        # Simulation state
        portfolio_value = initial_capital
        positions = {}
        trades = []
        daily_returns = []
        daily_positions = []
        
        try:
            for date in trading_dates:
                # Get market data up to this date
                historical_data = market_data.loc[:date]
                
                if len(historical_data) < 20:  # Need minimum history
                    continue
                
                # Generate signals
                universe = list(market_data.columns)
                signals = agent.generate_signals(historical_data, universe, date)
                
                if not signals.signals:
                    continue
                
                # Execute trades (simplified)
                for signal in signals.signals:
                    ticker = signal.ticker
                    
                    # Current price
                    if ticker not in historical_data.columns:
                        continue
                    current_price = historical_data[ticker].iloc[-1]
                    
                    # Position sizing (simplified)
                    position_value = portfolio_value * 0.05  # 5% per position
                    shares = position_value / current_price
                    
                    if signal.direction == "long":
                        positions[ticker] = shares
                    elif signal.direction == "short":
                        positions[ticker] = -shares
                    
                    # Record trade
                    trades.append({
                        'date': date,
                        'ticker': ticker,
                        'direction': signal.direction,
                        'shares': shares,
                        'price': current_price,
                        'alpha_bps': signal.alpha_bps,
                        'confidence': signal.confidence
                    })
                
                # Calculate daily P&L
                daily_pnl = 0.0
                position_values = {}
                
                for ticker, shares in positions.items():
                    if ticker in historical_data.columns:
                        current_price = historical_data[ticker].iloc[-1]
                        
                        # Calculate P&L from previous day
                        if len(historical_data) > 1:
                            prev_price = historical_data[ticker].iloc[-2]
                            pnl = shares * (current_price - prev_price)
                            daily_pnl += pnl
                        
                        position_values[ticker] = shares * current_price
                
                # Update portfolio value
                portfolio_value += daily_pnl
                daily_return = daily_pnl / portfolio_value if portfolio_value > 0 else 0.0
                
                daily_returns.append(daily_return)
                daily_positions.append(position_values.copy())
                
                # Risk management: close positions if drawdown too large
                if portfolio_value < initial_capital * 0.9:  # 10% drawdown limit
                    positions.clear()
                    self.logger.warning(f"Agent {agent.agent_name}: Hit drawdown limit, closing positions")
            
            # Calculate performance metrics
            return self._calculate_performance_metrics(
                trades, daily_returns, daily_positions, initial_capital, portfolio_value
            )
            
        except Exception as e:
            self.logger.error(f"Simulation error for {agent.agent_name}: {e}")
            return self._empty_simulation_result()
    
    def _calculate_performance_metrics(self,
                                     trades: List[Dict],
                                     daily_returns: List[float],
                                     daily_positions: List[Dict],
                                     initial_capital: float,
                                     final_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        if not trades or not daily_returns:
            return self._empty_simulation_result()
        
        # Basic metrics
        total_return = (final_capital - initial_capital) / initial_capital
        daily_returns_array = np.array(daily_returns)
        
        # Return metrics
        win_rate = len([r for r in daily_returns if r > 0]) / len(daily_returns)
        avg_daily_return = np.mean(daily_returns_array)
        vol_daily = np.std(daily_returns_array)
        sharpe_ratio = avg_daily_return / (vol_daily + 1e-8) * np.sqrt(252)
        
        # Drawdown
        cumulative_returns = np.cumprod(1 + daily_returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Alpha estimation (simplified)
        avg_alpha_bps = np.mean([t['alpha_bps'] for t in trades]) if trades else 0.0
        
        # Position metrics
        avg_num_positions = np.mean([len(pos) for pos in daily_positions]) if daily_positions else 0.0
        
        return {
            'total_trades': len(trades),
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_alpha_bps': avg_alpha_bps,
            'sharpe_ratio': sharpe_ratio,
            'volatility': vol_daily * np.sqrt(252),
            'max_drawdown': max_drawdown,
            'avg_num_positions': avg_num_positions,
            'final_capital': final_capital
        }
    
    def _empty_simulation_result(self) -> Dict[str, Any]:
        """Return empty simulation result."""
        return {
            'total_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'avg_alpha_bps': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'avg_num_positions': 0.0,
            'final_capital': 100000.0
        }


class AgentOnboardingSystem:
    """
    Complete agent onboarding and lifecycle management system.
    
    Manages:
    - 2-week sandbox evaluation
    - Performance monitoring and scoring
    - Automatic promotion/demotion decisions
    - Risk-adjusted capacity allocation
    """
    
    def __init__(self, config: Optional[OnboardingConfig] = None):
        self.config = config or OnboardingConfig()
        self.logger = logging.getLogger("agent_onboarding")
        
        # Agent tracking
        self.agent_registry: Dict[str, Dict] = {}  # Agent metadata
        self.agent_evaluations: Dict[str, List[AgentEvaluation]] = defaultdict(list)
        self.current_allocations: Dict[str, float] = {}
        
        # Performance tracking
        self.trade_simulator = TradeSimulator()
        
        # Initialize with existing agents
        self._initialize_agent_registry()
    
    def _initialize_agent_registry(self):
        """Initialize registry with available agents."""
        for agent_type in LEGENDARY_AGENTS.keys():
            self.agent_registry[agent_type] = {
                'status': AgentStatus.CANDIDATE,
                'created_date': datetime.now(),
                'last_evaluation': None,
                'total_evaluations': 0,
                'allocation': 0.0,
                'is_active': False
            }
    
    def onboard_new_agent(self, agent_type: str) -> bool:
        """Start onboarding process for a new agent."""
        
        if agent_type not in LEGENDARY_AGENTS:
            self.logger.error(f"Unknown agent type: {agent_type}")
            return False
        
        if agent_type in self.agent_registry:
            self.logger.warning(f"Agent {agent_type} already exists")
            return False
        
        # Initialize agent record
        self.agent_registry[agent_type] = {
            'status': AgentStatus.SANDBOX,
            'created_date': datetime.now(),
            'sandbox_start': datetime.now(),
            'last_evaluation': None,
            'total_evaluations': 0,
            'allocation': self.config.sandbox_allocation,
            'is_active': True
        }
        
        self.current_allocations[agent_type] = self.config.sandbox_allocation
        
        self.logger.info(
            f"🚀 Started sandbox evaluation for {agent_type} "
            f"with {self.config.sandbox_allocation:.1%} allocation"
        )
        
        return True
    
    def evaluate_agent(self,
                      agent_type: str,
                      market_data: pd.DataFrame,
                      evaluation_end: datetime) -> Optional[AgentEvaluation]:
        """Evaluate agent performance over specified period."""
        
        if agent_type not in self.agent_registry:
            self.logger.error(f"Agent {agent_type} not found in registry")
            return None
        
        agent_info = self.agent_registry[agent_type]
        
        # Determine evaluation period
        if agent_info['status'] == AgentStatus.SANDBOX:
            evaluation_start = agent_info['sandbox_start']
            required_days = self.config.sandbox_days
        elif agent_info['status'] == AgentStatus.PROBATION:
            evaluation_start = agent_info.get('probation_start', evaluation_end - timedelta(days=self.config.probation_days))
            required_days = self.config.probation_days
        else:
            # Regular review for active agents
            evaluation_start = evaluation_end - timedelta(days=self.config.review_frequency_days)
            required_days = self.config.review_frequency_days
        
        # Check if enough time has passed
        days_elapsed = (evaluation_end - evaluation_start).days
        if days_elapsed < required_days:
            self.logger.info(f"Agent {agent_type}: Not enough time for evaluation ({days_elapsed}/{required_days} days)")
            return None
        
        try:
            # Create agent instance
            agent = create_agent(agent_type)
            if not agent:
                self.logger.error(f"Failed to create agent instance: {agent_type}")
                return None
            
            # Run simulation
            simulation_results = self.trade_simulator.simulate_agent_trades(
                agent, market_data, evaluation_start, evaluation_end
            )
            
            # Calculate scores
            evaluation = self._score_agent_performance(
                agent_type, simulation_results, evaluation_start, evaluation_end
            )
            
            # Make promotion/demotion decision
            self._make_status_decision(evaluation)
            
            # Update agent registry
            self._update_agent_registry(agent_type, evaluation)
            
            # Store evaluation
            self.agent_evaluations[agent_type].append(evaluation)
            
            self.logger.info(
                f"📊 Evaluated {agent_type}: Score={evaluation.overall_score:.1f}, "
                f"Recommendation={evaluation.recommendation}"
            )
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating agent {agent_type}: {e}")
            return None
    
    def _score_agent_performance(self,
                               agent_type: str,
                               simulation_results: Dict[str, Any],
                               evaluation_start: datetime,
                               evaluation_end: datetime) -> AgentEvaluation:
        """Score agent performance across multiple dimensions."""
        
        # Extract simulation metrics
        total_trades = simulation_results['total_trades']
        win_rate = simulation_results['win_rate']
        avg_alpha_bps = simulation_results['avg_alpha_bps']
        sharpe_ratio = simulation_results['sharpe_ratio']
        max_drawdown = simulation_results['max_drawdown']
        volatility = simulation_results['volatility']
        total_return = simulation_results['total_return']
        
        # Calculate profitability score (0-100)
        profitability_score = 0.0
        if total_trades >= self.config.min_trade_frequency:
            # Edge after cost component (40%)
            edge_score = min(max(avg_alpha_bps / 10.0, 0), 40)  # 10 bps = 40 points
            
            # Win rate component (30%)
            win_score = win_rate * 30
            
            # Total return component (30%)
            return_score = min(max(total_return * 300, -30), 30)  # 10% return = 30 points
            
            profitability_score = edge_score + win_score + return_score
        
        # Calculate risk score (0-100)
        risk_score = 100.0  # Start with perfect score
        
        # Sharpe ratio component
        if sharpe_ratio < self.config.min_sharpe_ratio:
            risk_score -= 30
        else:
            risk_score += min((sharpe_ratio - self.config.min_sharpe_ratio) * 20, 20)
        
        # Drawdown penalty
        if max_drawdown < self.config.max_drawdown_limit:
            risk_score -= abs(max_drawdown) * 500  # 10% drawdown = -50 points
        
        # Volatility penalty
        if volatility > self.config.max_volatility:
            risk_score -= (volatility - self.config.max_volatility) * 200
        
        risk_score = max(risk_score, 0)
        
        # Calculate execution score (0-100)
        execution_score = 70.0  # Base score
        
        # Trade frequency
        if total_trades >= self.config.min_trade_frequency:
            execution_score += 20
        else:
            execution_score -= 30
        
        # Alpha prediction consistency (simplified)
        if avg_alpha_bps > 0 and total_return > 0:
            execution_score += 10  # Bonus for consistent alpha
        
        execution_score = max(min(execution_score, 100), 0)
        
        # Calculate overall score
        overall_score = (
            profitability_score * self.config.profitability_weight +
            risk_score * self.config.risk_weight +
            execution_score * self.config.execution_weight
        )
        
        # Estimated metrics (would come from actual KPI calculation)
        edge_after_cost_bps = max(avg_alpha_bps - 5.0, 0)  # Subtract estimated costs
        information_coefficient = min(max(win_rate - 0.5, 0) * 2, 0.2)  # Rough IC estimate
        implementation_shortfall_bps = 5.0 + (1.0 - win_rate) * 5.0  # IS estimate
        
        return AgentEvaluation(
            agent_name=agent_type,
            evaluation_period=(evaluation_start, evaluation_end),
            status=self.agent_registry[agent_type]['status'],
            total_trades=total_trades,
            win_rate=win_rate,
            avg_alpha_bps=avg_alpha_bps,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            edge_after_cost_bps=edge_after_cost_bps,
            information_coefficient=information_coefficient,
            implementation_shortfall_bps=implementation_shortfall_bps,
            vol_realized=volatility,
            var_95=total_return - 1.65 * volatility / np.sqrt(252),  # Rough VaR
            correlation_to_book=0.3,  # Placeholder
            avg_position_size=simulation_results['avg_num_positions'],
            turnover_daily=0.1,  # Placeholder
            profitability_score=profitability_score,
            risk_score=risk_score,
            execution_score=execution_score,
            overall_score=overall_score,
            recommendation="",  # Will be filled by decision logic
            new_allocation=0.0,  # Will be filled by decision logic
            reason="",  # Will be filled by decision logic
            timestamp=datetime.now()
        )
    
    def _make_status_decision(self, evaluation: AgentEvaluation):
        """Make promotion/demotion decision based on evaluation."""
        
        current_status = evaluation.status
        overall_score = evaluation.overall_score
        
        # Decision logic based on current status and score
        if current_status == AgentStatus.SANDBOX:
            if overall_score >= self.config.sandbox_pass_score:
                # Promote to active
                evaluation.recommendation = "PROMOTE_TO_ACTIVE"
                evaluation.new_allocation = self.config.min_active_allocation
                evaluation.reason = f"Passed sandbox with score {overall_score:.1f}"
            else:
                # Move to probation
                evaluation.recommendation = "MOVE_TO_PROBATION"
                evaluation.new_allocation = self.config.probation_allocation
                evaluation.reason = f"Failed sandbox ({overall_score:.1f}), giving second chance"
        
        elif current_status == AgentStatus.PROBATION:
            if overall_score >= self.config.probation_pass_score:
                # Promote to active
                evaluation.recommendation = "PROMOTE_TO_ACTIVE"
                evaluation.new_allocation = self.config.min_active_allocation
                evaluation.reason = f"Passed probation with score {overall_score:.1f}"
            else:
                # Retire agent
                evaluation.recommendation = "RETIRE"
                evaluation.new_allocation = 0.0
                evaluation.reason = f"Failed probation ({overall_score:.1f}), retiring agent"
        
        elif current_status == AgentStatus.ACTIVE:
            if overall_score >= self.config.veteran_promotion_score:
                # Promote to veteran
                evaluation.recommendation = "PROMOTE_TO_VETERAN"
                evaluation.new_allocation = min(
                    self.current_allocations.get(evaluation.agent_name, 0.02) * self.config.capacity_multiplier_veteran,
                    self.config.max_active_allocation
                )
                evaluation.reason = f"Excellent performance ({overall_score:.1f}), promoting to veteran"
            elif overall_score >= self.config.sandbox_pass_score:
                # Maintain active status, possibly adjust allocation
                current_alloc = self.current_allocations.get(evaluation.agent_name, 0.02)
                if overall_score >= 80:
                    new_alloc = min(current_alloc * 1.2, self.config.max_active_allocation)
                    evaluation.recommendation = "INCREASE_ALLOCATION"
                else:
                    new_alloc = current_alloc
                    evaluation.recommendation = "MAINTAIN"
                evaluation.new_allocation = new_alloc
                evaluation.reason = f"Maintaining active status (score {overall_score:.1f})"
            else:
                # Demote to probation
                evaluation.recommendation = "DEMOTE_TO_PROBATION"
                evaluation.new_allocation = self.config.probation_allocation
                evaluation.reason = f"Poor performance ({overall_score:.1f}), moving to probation"
        
        elif current_status == AgentStatus.VETERAN:
            if overall_score >= self.config.veteran_promotion_score:
                # Maintain veteran status
                evaluation.recommendation = "MAINTAIN_VETERAN"
                evaluation.new_allocation = min(
                    self.current_allocations.get(evaluation.agent_name, 0.05) * 1.1,  # Gradual increase
                    self.config.max_active_allocation
                )
                evaluation.reason = f"Maintaining veteran status (score {overall_score:.1f})"
            else:
                # Demote to active
                evaluation.recommendation = "DEMOTE_TO_ACTIVE"
                evaluation.new_allocation = self.config.min_active_allocation
                evaluation.reason = f"Veteran underperforming ({overall_score:.1f}), demoting to active"
        
        else:
            # Default: maintain current status
            evaluation.recommendation = "MAINTAIN"
            evaluation.new_allocation = self.current_allocations.get(evaluation.agent_name, 0.0)
            evaluation.reason = "No change"
    
    def _update_agent_registry(self, agent_type: str, evaluation: AgentEvaluation):
        """Update agent registry based on evaluation."""
        
        agent_info = self.agent_registry[agent_type]
        
        # Update status based on recommendation
        if evaluation.recommendation == "PROMOTE_TO_ACTIVE":
            agent_info['status'] = AgentStatus.ACTIVE
            agent_info['is_active'] = True
        elif evaluation.recommendation == "PROMOTE_TO_VETERAN":
            agent_info['status'] = AgentStatus.VETERAN
            agent_info['is_active'] = True
        elif evaluation.recommendation == "MOVE_TO_PROBATION":
            agent_info['status'] = AgentStatus.PROBATION
            agent_info['probation_start'] = datetime.now()
            agent_info['is_active'] = True
        elif evaluation.recommendation == "DEMOTE_TO_PROBATION":
            agent_info['status'] = AgentStatus.PROBATION
            agent_info['probation_start'] = datetime.now()
            agent_info['is_active'] = True
        elif evaluation.recommendation == "DEMOTE_TO_ACTIVE":
            agent_info['status'] = AgentStatus.ACTIVE
            agent_info['is_active'] = True
        elif evaluation.recommendation == "RETIRE":
            agent_info['status'] = AgentStatus.RETIRED
            agent_info['is_active'] = False
        
        # Update allocation
        agent_info['allocation'] = evaluation.new_allocation
        self.current_allocations[agent_type] = evaluation.new_allocation
        
        # Update tracking fields
        agent_info['last_evaluation'] = datetime.now()
        agent_info['total_evaluations'] += 1
    
    def run_periodic_evaluations(self, market_data: pd.DataFrame, as_of: datetime):
        """Run periodic evaluations for all agents."""
        
        evaluations_run = []
        
        for agent_type, agent_info in self.agent_registry.items():
            if not agent_info['is_active']:
                continue
            
            # Check if evaluation is due
            last_eval = agent_info.get('last_evaluation')
            if last_eval:
                days_since_eval = (as_of - last_eval).days
                required_days = self.config.review_frequency_days
            else:
                # First evaluation
                if agent_info['status'] == AgentStatus.SANDBOX:
                    days_since_start = (as_of - agent_info['sandbox_start']).days
                    required_days = self.config.sandbox_days
                else:
                    days_since_start = (as_of - agent_info['created_date']).days
                    required_days = self.config.review_frequency_days
                
                days_since_eval = days_since_start
            
            if days_since_eval >= required_days:
                evaluation = self.evaluate_agent(agent_type, market_data, as_of)
                if evaluation:
                    evaluations_run.append(evaluation)
        
        return evaluations_run
    
    def get_agent_roster(self) -> Dict[str, Dict]:
        """Get current agent roster with status and allocations."""
        
        roster = {}
        
        for agent_type, agent_info in self.agent_registry.items():
            recent_evaluations = self.agent_evaluations.get(agent_type, [])
            latest_eval = recent_evaluations[-1] if recent_evaluations else None
            
            roster[agent_type] = {
                'status': agent_info['status'].value,
                'allocation': agent_info['allocation'],
                'is_active': agent_info['is_active'],
                'created_date': agent_info['created_date'].strftime('%Y-%m-%d'),
                'total_evaluations': agent_info['total_evaluations'],
                'latest_score': latest_eval.overall_score if latest_eval else None,
                'latest_evaluation': latest_eval.timestamp.strftime('%Y-%m-%d') if latest_eval else None
            }
        
        return roster


if __name__ == "__main__":
    # Demo the onboarding system
    print("🎓 Agent Onboarding & Promotion System")
    print("=" * 50)
    
    # Create sample market data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    np.random.seed(42)
    market_data = pd.DataFrame(
        100 * np.cumprod(1 + np.random.normal(0.001, 0.02, (len(dates), len(symbols))), axis=0),
        index=dates,
        columns=symbols
    )
    
    # Initialize onboarding system
    onboarding = AgentOnboardingSystem()
    
    # Onboard a new agent
    agent_type = "simons_stat_arb"
    success = onboarding.onboard_new_agent(agent_type)
    print(f"✅ Onboarded {agent_type}: {success}")
    
    # Simulate evaluation after 2 weeks
    evaluation_date = dates[-15]  # 15 days ago
    evaluation = onboarding.evaluate_agent(agent_type, market_data, evaluation_date)
    
    if evaluation:
        print(f"\n📊 Evaluation Results for {agent_type}:")
        print(f"  Overall Score: {evaluation.overall_score:.1f}")
        print(f"  Recommendation: {evaluation.recommendation}")
        print(f"  New Allocation: {evaluation.new_allocation:.2%}")
        print(f"  Reason: {evaluation.reason}")
    
    # Show current roster
    print(f"\n👥 Current Agent Roster:")
    roster = onboarding.get_agent_roster()
    for agent, info in roster.items():
        print(f"  {agent}: {info['status']} | {info['allocation']:.2%} allocation")
