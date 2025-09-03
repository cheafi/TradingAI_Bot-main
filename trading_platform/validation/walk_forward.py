"""
Walk-Forward Validation Harness with Embargo Windows
==================================================

This module provides institutional-grade walk-forward validation that:
- Prevents look-ahead bias through strict temporal controls
- Implements embargo periods around rebalancing
- Provides realistic out-of-sample testing
- Tracks model degradation over time

CRITICAL: This is the foundation for realistic strategy validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import logging
import warnings
from abc import ABC, abstractmethod


@dataclass
class ValidationWindow:
    """A single validation window configuration."""
    train_start: datetime
    train_end: datetime
    embargo_start: datetime
    embargo_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int
    
    def __post_init__(self):
        """Validate window consistency."""
        assert self.train_start < self.train_end, "Invalid training period"
        assert self.train_end <= self.embargo_start, "Embargo must start after training"
        assert self.embargo_start < self.embargo_end, "Invalid embargo period"
        assert self.embargo_end <= self.test_start, "Test must start after embargo"
        assert self.test_start < self.test_end, "Invalid test period"


@dataclass
class ValidationConfig:
    """Configuration for walk-forward validation."""
    min_train_days: int = 252  # Minimum training period (1 year)
    test_days: int = 21  # Test period (1 month)
    embargo_days: int = 2  # Embargo period (2 days)
    step_days: int = 21  # Step between windows (1 month)
    max_windows: Optional[int] = None  # Maximum number of windows
    
    # Advanced settings
    expanding_window: bool = False  # Use expanding vs rolling window
    min_observations: int = 1000  # Minimum observations for training
    purge_overlap: bool = True  # Purge overlapping observations
    
    # Model retraining
    retrain_frequency: int = 1  # Retrain every N windows
    model_decay_threshold: float = 0.1  # Retrain if performance drops >10%


class StrategyProtocol(ABC):
    """Protocol that strategies must implement for walk-forward validation."""
    
    @abstractmethod
    def fit(self, 
            features: pd.DataFrame, 
            targets: pd.DataFrame,
            **kwargs) -> None:
        """Train the strategy on the given data."""
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for the given features."""
        pass
    
    @abstractmethod
    def get_positions(self, 
                     predictions: pd.DataFrame,
                     current_positions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Convert predictions to position weights."""
        pass
    
    def clone(self) -> 'StrategyProtocol':
        """Create a copy of the strategy for independent training."""
        # Default implementation - strategies should override if needed
        return self


class WalkForwardValidator:
    """
    Walk-forward validation engine with strict temporal controls.
    
    This validator ensures:
    1. No look-ahead bias through embargo periods
    2. Realistic out-of-sample testing
    3. Model degradation tracking
    4. Proper data leakage prevention
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validation results
        self.validation_windows: List[ValidationWindow] = []
        self.results: Dict[str, Any] = {}
        self.model_performance: List[Dict] = []
        
        # Strategy tracking
        self.trained_strategies: Dict[int, StrategyProtocol] = {}
        
    def create_windows(self, 
                      start_date: datetime,
                      end_date: datetime) -> List[ValidationWindow]:
        """Create walk-forward validation windows."""
        windows = []
        current_date = start_date
        window_id = 0
        
        while current_date < end_date:
            # Calculate training period
            if self.config.expanding_window and window_id > 0:
                # Expanding window: always start from beginning
                train_start = start_date
            else:
                # Rolling window: fixed training period
                train_start = current_date
            
            train_end = current_date + timedelta(days=self.config.min_train_days)
            
            # Check if we have enough data for training
            if train_end >= end_date:
                break
            
            # Calculate embargo period
            embargo_start = train_end
            embargo_end = embargo_start + timedelta(days=self.config.embargo_days)
            
            # Calculate test period
            test_start = embargo_end
            test_end = test_start + timedelta(days=self.config.test_days)
            
            # Check if test period fits
            if test_end > end_date:
                break
            
            window = ValidationWindow(
                train_start=train_start,
                train_end=train_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end,
                window_id=window_id
            )
            
            windows.append(window)
            
            # Move to next window
            current_date += timedelta(days=self.config.step_days)
            window_id += 1
            
            # Check maximum windows limit
            if (self.config.max_windows and 
                len(windows) >= self.config.max_windows):
                break
        
        self.validation_windows = windows
        self.logger.info(f"Created {len(windows)} validation windows")
        
        return windows
    
    def validate_strategy(self,
                         strategy: StrategyProtocol,
                         features: pd.DataFrame,
                         targets: pd.DataFrame,
                         prices: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run walk-forward validation on a strategy.
        
        Args:
            strategy: Strategy to validate
            features: Feature data with datetime index
            targets: Target data with datetime index
            prices: Price data for performance calculation
            
        Returns:
            Comprehensive validation results
        """
        
        if not self.validation_windows:
            raise ValueError("No validation windows created. Call create_windows() first.")
        
        # Validate data alignment
        self._validate_data_alignment(features, targets, prices)
        
        # Initialize results storage
        oos_predictions = []
        oos_returns = []
        window_performances = []
        
        # Run validation for each window
        for window in self.validation_windows:
            self.logger.info(f"Processing window {window.window_id}")
            
            try:
                # Extract data for this window
                train_data = self._extract_window_data(
                    features, targets, window.train_start, window.train_end
                )
                
                test_data = self._extract_window_data(
                    features, targets, window.test_start, window.test_end
                )
                
                if len(train_data[0]) < self.config.min_observations:
                    self.logger.warning(
                        f"Insufficient training data for window {window.window_id}"
                    )
                    continue
                
                # Train strategy (with purging if enabled)
                strategy_copy = strategy.clone()
                
                if self.config.purge_overlap:
                    # Remove overlapping observations from training data
                    train_features, train_targets = self._purge_overlapping_data(
                        train_data[0], train_data[1], window
                    )
                else:
                    train_features, train_targets = train_data
                
                # Fit the strategy
                strategy_copy.fit(train_features, train_targets)
                self.trained_strategies[window.window_id] = strategy_copy
                
                # Generate out-of-sample predictions
                test_features, test_targets = test_data
                predictions = strategy_copy.predict(test_features)
                
                # Calculate positions
                positions = strategy_copy.get_positions(predictions)
                
                # Calculate performance for this window
                window_perf = self._calculate_window_performance(
                    window, predictions, test_targets, positions, prices
                )
                
                window_performances.append(window_perf)
                oos_predictions.append(predictions)
                oos_returns.append(window_perf.get('returns', pd.Series()))
                
            except Exception as e:
                self.logger.error(f"Error in window {window.window_id}: {e}")
                continue
        
        # Aggregate results
        aggregated_results = self._aggregate_results(
            window_performances, oos_predictions, oos_returns
        )
        
        # Analyze model degradation
        degradation_analysis = self._analyze_model_degradation(window_performances)
        
        # Compile final results
        final_results = {
            'windows': self.validation_windows,
            'window_performances': window_performances,
            'aggregated_performance': aggregated_results,
            'model_degradation': degradation_analysis,
            'oos_predictions': oos_predictions,
            'config': self.config,
            'summary_stats': self._calculate_summary_stats(aggregated_results)
        }
        
        self.results = final_results
        return final_results
    
    def _validate_data_alignment(self, 
                               features: pd.DataFrame,
                               targets: pd.DataFrame,
                               prices: Optional[pd.DataFrame]) -> None:
        """Validate that data is properly aligned."""
        
        # Check datetime indices
        if not isinstance(features.index, pd.DatetimeIndex):
            raise ValueError("Features must have DatetimeIndex")
        
        if not isinstance(targets.index, pd.DatetimeIndex):
            raise ValueError("Targets must have DatetimeIndex")
        
        # Check data overlap
        feature_dates = set(features.index)
        target_dates = set(targets.index)
        
        overlap = feature_dates.intersection(target_dates)
        if len(overlap) == 0:
            raise ValueError("No overlapping dates between features and targets")
        
        # Check for sufficient data
        min_date = max(features.index.min(), targets.index.min())
        max_date = min(features.index.max(), targets.index.max())
        
        total_days = (max_date - min_date).days
        min_required = self.config.min_train_days + self.config.test_days + self.config.embargo_days
        
        if total_days < min_required:
            raise ValueError(f"Insufficient data: {total_days} days, need {min_required}")
    
    def _extract_window_data(self,
                           features: pd.DataFrame,
                           targets: pd.DataFrame,
                           start_date: datetime,
                           end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract data for a specific window."""
        
        # Use pd.Timestamp for indexing
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Extract features
        feature_mask = (features.index >= start_ts) & (features.index < end_ts)
        window_features = features.loc[feature_mask].copy()
        
        # Extract targets
        target_mask = (targets.index >= start_ts) & (targets.index < end_ts)
        window_targets = targets.loc[target_mask].copy()
        
        # Align dates
        common_dates = window_features.index.intersection(window_targets.index)
        
        if len(common_dates) == 0:
            raise ValueError(f"No common dates in window {start_date} to {end_date}")
        
        window_features = window_features.loc[common_dates]
        window_targets = window_targets.loc[common_dates]
        
        return window_features, window_targets
    
    def _purge_overlapping_data(self,
                              features: pd.DataFrame,
                              targets: pd.DataFrame,
                              window: ValidationWindow) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove data that might overlap with test period."""
        
        # Remove data close to test period to prevent leakage
        purge_cutoff = window.embargo_start - timedelta(days=1)
        
        purge_mask = features.index <= purge_cutoff
        
        purged_features = features.loc[purge_mask]
        purged_targets = targets.loc[purge_mask]
        
        return purged_features, purged_targets
    
    def _calculate_window_performance(self,
                                    window: ValidationWindow,
                                    predictions: pd.DataFrame,
                                    targets: pd.DataFrame,
                                    positions: pd.DataFrame,
                                    prices: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate performance metrics for a single window."""
        
        performance = {
            'window_id': window.window_id,
            'start_date': window.test_start,
            'end_date': window.test_end,
            'num_predictions': len(predictions)
        }
        
        # Prediction accuracy metrics
        if len(predictions) > 0 and len(targets) > 0:
            # Align predictions and targets
            common_index = predictions.index.intersection(targets.index)
            
            if len(common_index) > 0:
                aligned_preds = predictions.loc[common_index]
                aligned_targets = targets.loc[common_index]
                
                # Calculate correlation
                if aligned_preds.shape[1] == aligned_targets.shape[1]:
                    correlations = []
                    for col in aligned_preds.columns:
                        if col in aligned_targets.columns:
                            corr = aligned_preds[col].corr(aligned_targets[col])
                            if not pd.isna(corr):
                                correlations.append(corr)
                    
                    if correlations:
                        performance['ic_mean'] = np.mean(correlations)
                        performance['ic_std'] = np.std(correlations)
                        performance['ic_ir'] = np.mean(correlations) / max(np.std(correlations), 1e-8)
        
        # Portfolio performance (if prices provided)
        if prices is not None and len(positions) > 0:
            perf_metrics = self._calculate_portfolio_performance(
                positions, prices, window.test_start, window.test_end
            )
            performance.update(perf_metrics)
        
        return performance
    
    def _calculate_portfolio_performance(self,
                                       positions: pd.DataFrame,
                                       prices: pd.DataFrame,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        
        # Extract price data for test period
        price_mask = (prices.index >= start_date) & (prices.index <= end_date)
        test_prices = prices.loc[price_mask]
        
        if len(test_prices) == 0:
            return {}
        
        # Calculate returns
        returns = test_prices.pct_change().dropna()
        
        # Align positions with returns
        common_dates = positions.index.intersection(returns.index)
        common_assets = list(set(positions.columns).intersection(set(returns.columns)))
        
        if len(common_dates) == 0 or len(common_assets) == 0:
            return {}
        
        aligned_positions = positions.loc[common_dates, common_assets]
        aligned_returns = returns.loc[common_dates, common_assets]
        
        # Calculate portfolio returns
        portfolio_returns = (aligned_positions.shift(1) * aligned_returns).sum(axis=1)
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) == 0:
            return {}
        
        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / max(volatility, 1e-8)
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'returns': portfolio_returns
        }
    
    def _aggregate_results(self,
                         window_performances: List[Dict],
                         oos_predictions: List[pd.DataFrame],
                         oos_returns: List[pd.Series]) -> Dict[str, Any]:
        """Aggregate results across all windows."""
        
        if not window_performances:
            return {}
        
        # Aggregate performance metrics
        metrics = ['ic_mean', 'ic_ir', 'annualized_return', 'sharpe_ratio', 'max_drawdown']
        aggregated = {}
        
        for metric in metrics:
            values = [wp.get(metric) for wp in window_performances if wp.get(metric) is not None]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_median'] = np.median(values)
        
        # Combine all out-of-sample returns
        if oos_returns:
            combined_returns = pd.concat([r for r in oos_returns if len(r) > 0])
            if len(combined_returns) > 0:
                aggregated['total_return'] = (1 + combined_returns).prod() - 1
                aggregated['total_volatility'] = combined_returns.std() * np.sqrt(252)
                aggregated['total_sharpe'] = (
                    aggregated.get('annualized_return_mean', 0) / 
                    max(aggregated.get('total_volatility', 1e-8), 1e-8)
                )
        
        return aggregated
    
    def _analyze_model_degradation(self, window_performances: List[Dict]) -> Dict[str, Any]:
        """Analyze model performance degradation over time."""
        
        if len(window_performances) < 2:
            return {}
        
        # Extract performance metrics over time
        ic_series = []
        return_series = []
        
        for wp in window_performances:
            if 'ic_mean' in wp:
                ic_series.append(wp['ic_mean'])
            if 'annualized_return' in wp:
                return_series.append(wp['annualized_return'])
        
        degradation_analysis = {}
        
        # Analyze IC degradation
        if len(ic_series) >= 3:
            ic_trend = np.polyfit(range(len(ic_series)), ic_series, 1)[0]
            degradation_analysis['ic_trend_slope'] = ic_trend
            degradation_analysis['ic_degradation'] = ic_trend < -0.01  # >1% degradation per window
        
        # Analyze return degradation
        if len(return_series) >= 3:
            return_trend = np.polyfit(range(len(return_series)), return_series, 1)[0]
            degradation_analysis['return_trend_slope'] = return_trend
            degradation_analysis['return_degradation'] = return_trend < -0.02  # >2% degradation per window
        
        # Overall degradation flag
        degradation_analysis['significant_degradation'] = (
            degradation_analysis.get('ic_degradation', False) or
            degradation_analysis.get('return_degradation', False)
        )
        
        return degradation_analysis
    
    def _calculate_summary_stats(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        
        summary = {
            'total_windows': len(self.validation_windows),
            'successful_windows': len([w for w in self.model_performance if 'ic_mean' in w]),
            'avg_ic': aggregated_results.get('ic_mean_mean'),
            'avg_sharpe': aggregated_results.get('sharpe_ratio_mean'),
            'consistency_score': self._calculate_consistency_score(aggregated_results)
        }
        
        return summary
    
    def _calculate_consistency_score(self, aggregated_results: Dict[str, Any]) -> Optional[float]:
        """Calculate strategy consistency score (0-1)."""
        
        ic_mean = aggregated_results.get('ic_mean_mean')
        ic_std = aggregated_results.get('ic_mean_std')
        
        if ic_mean is not None and ic_std is not None and ic_std > 0:
            # Higher score for higher mean IC and lower volatility
            consistency = max(0, min(1, ic_mean / max(ic_std, 1e-8) / 2))
            return consistency
        
        return None


def create_embargo_validator(min_train_months: int = 12,
                           test_months: int = 1,
                           embargo_days: int = 2,
                           step_months: int = 1) -> WalkForwardValidator:
    """Create a walk-forward validator with common institutional settings."""
    
    config = ValidationConfig(
        min_train_days=min_train_months * 30,  # Approximate
        test_days=test_months * 30,
        embargo_days=embargo_days,
        step_days=step_months * 30,
        expanding_window=False,
        purge_overlap=True,
        retrain_frequency=1
    )
    
    return WalkForwardValidator(config)


if __name__ == "__main__":
    # Example usage
    
    # Create validator
    validator = create_embargo_validator(
        min_train_months=24,  # 2 years training
        test_months=1,        # 1 month testing
        embargo_days=3,       # 3 days embargo
        step_months=1         # 1 month steps
    )
    
    # Create windows for 2020-2023
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    windows = validator.create_windows(start_date, end_date)
    
    print(f"Created {len(windows)} validation windows")
    for i, window in enumerate(windows[:3]):  # Show first 3
        print(f"Window {i}:")
        print(f"  Train: {window.train_start.date()} to {window.train_end.date()}")
        print(f"  Embargo: {window.embargo_start.date()} to {window.embargo_end.date()}")
        print(f"  Test: {window.test_start.date()} to {window.test_end.date()}")
        print()
