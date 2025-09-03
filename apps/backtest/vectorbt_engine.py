"""
Advanced Backtesting Engine with Multiple Backends
Supports vectorbt for parameter sweeps, realistic cost modeling, and walk-forward validation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import vectorbt as vbt
from sklearn.model_selection import TimeSeriesSplit

from platform.config import BacktestConfig
from platform.events import Event, FillEvent, OrderEvent

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    
    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    
    # Cost analysis
    total_commission: float
    total_slippage: float
    implementation_shortfall: float
    turnover: float
    
    # Time series data
    equity_curve: pd.Series
    drawdown_series: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    
    # Attribution
    factor_attribution: Optional[Dict[str, float]] = None
    sector_attribution: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "performance": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "volatility": self.volatility,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "max_drawdown": self.max_drawdown
            },
            "risk": {
                "var_95": self.var_95,
                "cvar_95": self.cvar_95,
                "beta": self.beta,
                "alpha": self.alpha
            },
            "trading": {
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "avg_trade_return": self.avg_trade_return,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss
            },
            "costs": {
                "total_commission": self.total_commission,
                "total_slippage": self.total_slippage,
                "implementation_shortfall": self.implementation_shortfall,
                "turnover": self.turnover
            }
        }


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(
        self, 
        symbol: str,
        side: str,
        quantity: float,
        market_price: float,
        volume: float,
        volatility: float
    ) -> float:
        """Calculate slippage for a trade."""
        pass


class LinearSlippageModel(SlippageModel):
    """Linear slippage model based on order size."""
    
    def __init__(self, base_slippage: float = 0.0005, volume_impact: float = 0.1):
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact
    
    def calculate_slippage(
        self, 
        symbol: str,
        side: str,
        quantity: float,
        market_price: float,
        volume: float,
        volatility: float
    ) -> float:
        """
        Linear slippage = base_slippage + volume_impact * (quantity / avg_volume)
        """
        if volume <= 0:
            return self.base_slippage * market_price
            
        participation_rate = quantity * market_price / (volume * market_price)
        slippage_rate = self.base_slippage + self.volume_impact * participation_rate
        
        # Apply direction (buying costs more)
        direction_multiplier = 1.0 if side.lower() == "buy" else -1.0
        
        return slippage_rate * market_price * direction_multiplier


class SqrtSlippageModel(SlippageModel):
    """Square root slippage model (more realistic for large orders)."""
    
    def __init__(self, impact_coefficient: float = 0.01):
        self.impact_coefficient = impact_coefficient
    
    def calculate_slippage(
        self, 
        symbol: str,
        side: str,
        quantity: float,
        market_price: float,
        volume: float,
        volatility: float
    ) -> float:
        """
        Square root model: slippage = impact_coeff * volatility * sqrt(quantity / volume)
        """
        if volume <= 0:
            return 0.0
            
        participation_rate = quantity * market_price / (volume * market_price)
        slippage_rate = self.impact_coefficient * volatility * np.sqrt(participation_rate)
        
        direction_multiplier = 1.0 if side.lower() == "buy" else -1.0
        
        return slippage_rate * market_price * direction_multiplier


class VectorBTBacktester:
    """High-performance backtesting using vectorbt."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.slippage_model = self._create_slippage_model()
        
    def _create_slippage_model(self) -> SlippageModel:
        """Create slippage model based on config."""
        model_type = self.config.slippage_model.lower()
        
        if model_type == "linear":
            return LinearSlippageModel()
        elif model_type == "sqrt":
            return SqrtSlippageModel()
        else:
            return LinearSlippageModel()  # Default
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None
    ) -> BacktestResults:
        """
        Run vectorized backtest with realistic cost modeling.
        
        Parameters:
        - prices: DataFrame with OHLCV data
        - signals: DataFrame with trading signals (-1 to 1)
        - volumes: Optional volume data for slippage calculation
        - benchmark: Optional benchmark for alpha/beta calculation
        """
        logger.info("Starting vectorbt backtest")
        
        # Prepare data
        close_prices = prices['Close'] if 'Close' in prices.columns else prices
        signals_clean = signals.fillna(0).clip(-1, 1)
        
        # Calculate positions from signals
        positions = signals_clean.shift(1).fillna(0)  # Avoid look-ahead
        
        # Calculate returns without costs first
        returns = close_prices.pct_change()
        strategy_returns = (positions * returns).sum(axis=1)
        
        # Calculate trading costs
        position_changes = positions.diff().fillna(0)
        turnover_series = position_changes.abs().sum(axis=1)
        
        # Commission costs
        commission_rate = self.config.commission_rate
        commission_costs = turnover_series * commission_rate * close_prices.mean(axis=1)
        
        # Slippage costs (simplified)
        slippage_costs = self._calculate_slippage_costs(
            position_changes, close_prices, volumes
        )
        
        # Net returns after costs
        total_costs = commission_costs + slippage_costs
        net_returns = strategy_returns - total_costs / close_prices.mean(axis=1)
        
        # Calculate equity curve
        equity_curve = (1 + net_returns).cumprod()
        
        # Performance metrics
        total_return = equity_curve.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        volatility = net_returns.std() * np.sqrt(252)
        
        # Risk metrics
        drawdown_series = (equity_curve / equity_curve.cummax() - 1)
        max_drawdown = drawdown_series.min()
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = net_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(net_returns, 5)
        cvar_95 = net_returns[net_returns <= var_95].mean()
        
        # Trading statistics
        trades_df = self._calculate_trade_statistics(position_changes, close_prices)
        
        total_trades = len(trades_df)
        win_rate = (trades_df['return'] > 0).mean() if total_trades > 0 else 0
        avg_trade_return = trades_df['return'].mean() if total_trades > 0 else 0
        
        wins = trades_df[trades_df['return'] > 0]['return']
        losses = trades_df[trades_df['return'] < 0]['return']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
        
        # Alpha and beta vs benchmark
        alpha, beta = None, None
        if benchmark is not None:
            benchmark_returns = benchmark.pct_change().dropna()
            aligned_returns = net_returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_returns.index)
            
            if len(aligned_returns) > 0 and len(aligned_benchmark) > 0:
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = aligned_benchmark.var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                alpha = aligned_returns.mean() - beta * aligned_benchmark.mean()
                alpha *= 252  # Annualize
        
        # Costs summary
        total_commission = commission_costs.sum()
        total_slippage = slippage_costs.sum()
        turnover = turnover_series.mean() * 252  # Annualized turnover
        
        # Implementation shortfall (simplified)
        implementation_shortfall = (total_commission + total_slippage) / equity_curve.iloc[0]
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_commission=total_commission,
            total_slippage=total_slippage,
            implementation_shortfall=implementation_shortfall,
            turnover=turnover,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            positions=positions,
            trades=trades_df
        )
    
    def _calculate_slippage_costs(
        self, 
        position_changes: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame]
    ) -> pd.Series:
        """Calculate slippage costs for position changes."""
        if volumes is None:
            # Use simplified slippage if no volume data
            return position_changes.abs().sum(axis=1) * 0.0005 * prices.mean(axis=1)
        
        slippage_costs = pd.Series(0.0, index=position_changes.index)
        
        for timestamp in position_changes.index:
            for symbol in position_changes.columns:
                change = position_changes.loc[timestamp, symbol]
                if abs(change) > 0.001:  # Minimum trade threshold
                    price = prices.loc[timestamp, symbol] if symbol in prices.columns else prices.loc[timestamp].mean()
                    volume = volumes.loc[timestamp, symbol] if symbol in volumes.columns else 1000000
                    
                    # Estimate volatility (simplified)
                    volatility = 0.02  # 2% daily volatility assumption
                    
                    side = "buy" if change > 0 else "sell"
                    slippage = self.slippage_model.calculate_slippage(
                        symbol, side, abs(change), price, volume, volatility
                    )
                    slippage_costs.loc[timestamp] += abs(slippage)
        
        return slippage_costs
    
    def _calculate_trade_statistics(
        self, 
        position_changes: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate individual trade statistics."""
        trades = []
        
        for symbol in position_changes.columns:
            pos_changes = position_changes[symbol]
            symbol_prices = prices[symbol] if symbol in prices.columns else prices.mean(axis=1)
            
            # Find trade entries and exits
            current_position = 0
            entry_price = 0
            entry_time = None
            
            for timestamp, change in pos_changes.items():
                if abs(change) > 0.001:  # Minimum change threshold
                    new_position = current_position + change
                    current_price = symbol_prices.loc[timestamp]
                    
                    if current_position == 0 and new_position != 0:
                        # Opening position
                        entry_price = current_price
                        entry_time = timestamp
                        current_position = new_position
                    elif current_position != 0 and new_position == 0:
                        # Closing position
                        if entry_time is not None:
                            trade_return = (current_price - entry_price) / entry_price
                            if current_position < 0:  # Short position
                                trade_return = -trade_return
                                
                            trades.append({
                                'symbol': symbol,
                                'entry_time': entry_time,
                                'exit_time': timestamp,
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'position_size': abs(current_position),
                                'return': trade_return,
                                'duration': (timestamp - entry_time).days
                            })
                        
                        current_position = 0
                        entry_time = None
                    else:
                        # Adjusting position
                        current_position = new_position
        
        return pd.DataFrame(trades)


class WalkForwardValidator:
    """Walk-forward validation for time series strategies."""
    
    def __init__(self, backtester: VectorBTBacktester, config: BacktestConfig):
        self.backtester = backtester
        self.config = config
    
    def validate(
        self,
        data: pd.DataFrame,
        strategy_func: callable,
        **strategy_params
    ) -> List[BacktestResults]:
        """
        Perform walk-forward validation.
        
        Parameters:
        - data: Full dataset with OHLCV data
        - strategy_func: Function that generates signals given data and parameters
        - strategy_params: Parameters to pass to strategy function
        """
        train_days = self.config.train_period_days
        test_days = self.config.test_period_days
        min_samples = self.config.min_samples
        
        results = []
        
        # Create time-based splits
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date + timedelta(days=train_days)
        
        while current_date + timedelta(days=test_days) <= end_date:
            # Define train and test periods
            train_start = current_date - timedelta(days=train_days)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=test_days)
            
            # Extract data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]
            
            if len(train_data) < min_samples or len(test_data) < 10:
                current_date += timedelta(days=test_days)
                continue
            
            try:
                # Generate signals on training data
                train_signals = strategy_func(train_data, **strategy_params)
                
                # Apply to test data (avoiding look-ahead)
                test_signals = strategy_func(test_data, **strategy_params)
                
                # Run backtest on test period
                result = self.backtester.run_backtest(
                    prices=test_data,
                    signals=test_signals
                )
                
                # Add period information
                result.train_period = (train_start, train_end)
                result.test_period = (test_start, test_end)
                
                results.append(result)
                
                logger.info(f"Walk-forward period {test_start.date()} to {test_end.date()}: "
                          f"Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.2f}")
                
            except Exception as e:
                logger.error(f"Error in walk-forward period {test_start} to {test_end}: {e}")
            
            # Move to next period
            current_date += timedelta(days=test_days)
        
        return results
    
    def aggregate_results(self, results: List[BacktestResults]) -> BacktestResults:
        """Aggregate results from multiple walk-forward periods."""
        if not results:
            raise ValueError("No results to aggregate")
        
        # Combine equity curves
        combined_equity = pd.Series(1.0)
        combined_drawdowns = pd.Series(0.0)
        all_trades = []
        
        for result in results:
            # Chain equity curves
            period_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0]
            combined_equity = combined_equity.append(
                result.equity_curve * combined_equity.iloc[-1] / result.equity_curve.iloc[0]
            )
            combined_drawdowns = combined_drawdowns.append(result.drawdown_series)
            all_trades.append(result.trades)
        
        # Calculate aggregate metrics
        total_return = combined_equity.iloc[-1] - 1
        returns = combined_equity.pct_change().dropna()
        
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Other aggregate metrics
        all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=np.mean([r.sortino_ratio for r in results]),
            calmar_ratio=np.mean([r.calmar_ratio for r in results]),
            max_drawdown=combined_drawdowns.min(),
            var_95=np.mean([r.var_95 for r in results]),
            cvar_95=np.mean([r.cvar_95 for r in results]),
            total_trades=len(all_trades_df),
            win_rate=np.mean([r.win_rate for r in results]),
            profit_factor=np.mean([r.profit_factor for r in results]),
            avg_trade_return=np.mean([r.avg_trade_return for r in results]),
            avg_win=np.mean([r.avg_win for r in results]),
            avg_loss=np.mean([r.avg_loss for r in results]),
            total_commission=sum(r.total_commission for r in results),
            total_slippage=sum(r.total_slippage for r in results),
            implementation_shortfall=np.mean([r.implementation_shortfall for r in results]),
            turnover=np.mean([r.turnover for r in results]),
            equity_curve=combined_equity,
            drawdown_series=combined_drawdowns,
            positions=pd.DataFrame(),  # Would need to combine
            trades=all_trades_df
        )
