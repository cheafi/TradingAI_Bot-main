"""
Qlib integration for advanced ML research and factor analysis.
Provides a research workflow for systematic strategy development.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import joblib

logger = logging.getLogger(__name__)


class QlibResearchWorkflow:
    """Qlib-inspired research workflow for systematic trading."""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.factor_library = {}
        self.models = {}
        
    def create_factor_library(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create a comprehensive factor library from OHLCV data."""
        factors = {}
        
        # Price factors
        factors['returns_1d'] = df['close'].pct_change()
        factors['returns_5d'] = df['close'].pct_change(5)
        factors['returns_20d'] = df['close'].pct_change(20)
        
        # Volatility factors
        factors['volatility_10d'] = factors['returns_1d'].rolling(10).std()
        factors['volatility_20d'] = factors['returns_1d'].rolling(20).std()
        
        # Momentum factors
        factors['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        factors['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
        # Technical indicators
        factors['rsi_14'] = self._calculate_rsi(df['close'], 14)
        factors['ma_ratio_5_20'] = (
            df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1
        )
        
        # Volume factors
        if 'volume' in df.columns:
            factors['volume_ratio'] = (
                df['volume'] / df['volume'].rolling(20).mean()
            )
            factors['price_volume'] = factors['returns_1d'] * factors['volume_ratio']
        
        # Mean reversion factors
        factors['price_position'] = (
            (df['close'] - df['close'].rolling(20).min()) / 
            (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        )
        
        # Regime factors
        factors['regime_volatility'] = (
            factors['volatility_10d'] > factors['volatility_10d'].rolling(60).quantile(0.8)
        ).astype(int)
        
        self.factor_library = factors
        return factors
    
    def factor_analysis(self, factors: Dict[str, pd.Series], 
                       target: pd.Series) -> pd.DataFrame:
        """Analyze factor performance and correlations."""
        analysis_results = []
        
        for factor_name, factor_values in factors.items():
            # Align data
            aligned_data = pd.concat([factor_values, target], axis=1).dropna()
            if len(aligned_data) < 50:  # Minimum data points
                continue
                
            factor_col = aligned_data.iloc[:, 0]
            target_col = aligned_data.iloc[:, 1]
            
            # Calculate metrics
            correlation = factor_col.corr(target_col)
            
            # IC (Information Coefficient) analysis
            ic_mean = correlation
            ic_std = np.nan  # Simplified for demo
            ic_ir = ic_mean / (ic_std if ic_std and ic_std != 0 else 1)
            
            # Rank correlation (Spearman)
            rank_corr = factor_col.corr(target_col, method='spearman')
            
            analysis_results.append({
                'factor': factor_name,
                'ic_mean': ic_mean,
                'ic_ir': ic_ir,
                'rank_ic': rank_corr,
                'coverage': len(aligned_data) / len(target),
                'factor_mean': factor_col.mean(),
                'factor_std': factor_col.std()
            })
        
        results_df = pd.DataFrame(analysis_results)
        results_df = results_df.sort_values('ic_ir', key=abs, ascending=False)
        
        # Save results
        results_path = self.results_dir / "factor_analysis.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Factor analysis saved to {results_path}")
        
        return results_df
    
    def feature_selection(self, factors_df: pd.DataFrame, 
                         target: pd.Series, top_k: int = 20) -> List[str]:
        """Select top factors based on IC and other criteria."""
        analysis = self.factor_analysis(factors_df, target)
        
        # Multi-criteria selection
        # 1. High IC
        high_ic = analysis.nlargest(top_k // 2, 'ic_ir')['factor'].tolist()
        
        # 2. High rank IC (for non-linear relationships)
        high_rank_ic = analysis.nlargest(top_k // 2, 'rank_ic')['factor'].tolist()
        
        # Combine and deduplicate
        selected_factors = list(set(high_ic + high_rank_ic))[:top_k]
        
        logger.info(f"Selected {len(selected_factors)} factors: {selected_factors}")
        return selected_factors
    
    def backtest_factor_strategy(self, factors: Dict[str, pd.Series],
                                selected_factors: List[str],
                                target: pd.Series) -> Dict:
        """Backtest a factor-based strategy."""
        # Create feature matrix
        feature_df = pd.DataFrame(factors)[selected_factors]
        
        # Align with target
        aligned_data = pd.concat([feature_df, target], axis=1).dropna()
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        if len(X) < 100:
            logger.warning("Insufficient data for backtesting")
            return {"error": "Insufficient data"}
        
        # Simple factor scoring model
        # Weight factors by their IC
        analysis = self.factor_analysis(factors, target)
        factor_weights = {}
        
        for factor in selected_factors:
            weight = analysis[analysis['factor'] == factor]['ic_ir'].iloc[0]
            factor_weights[factor] = weight if not np.isnan(weight) else 0
        
        # Generate signals
        signals = np.zeros(len(X))
        for i, factor in enumerate(selected_factors):
            if factor in feature_df.columns:
                factor_score = X[factor] * factor_weights.get(factor, 0)
                signals += factor_score.fillna(0).values
        
        # Normalize signals to [-1, 1]
        if np.std(signals) > 0:
            signals = signals / np.std(signals)
            signals = np.clip(signals, -1, 1)
        
        # Calculate strategy returns
        strategy_returns = signals[:-1] * y.values[1:]  # Next period returns
        
        # Performance metrics
        total_return = np.prod(1 + strategy_returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cum_returns = np.cumprod(1 + strategy_returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak
        max_drawdown = np.max(drawdown)
        
        results = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": np.sum(np.abs(np.diff(signals)) > 0.1),
            "selected_factors": selected_factors,
            "factor_weights": factor_weights
        }
        
        # Save backtest results
        results_path = self.results_dir / "factor_backtest.json"
        import json
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    serializable_results[k] = v.tolist()
                elif isinstance(v, (np.float64, np.int64)):
                    serializable_results[k] = float(v)
                else:
                    serializable_results[k] = v
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Backtest results saved to {results_path}")
        logger.info(f"Strategy Sharpe: {sharpe:.2f}, Max DD: {max_drawdown:.2%}")
        
        return results
    
    def run_full_research_pipeline(self, df: pd.DataFrame, 
                                  target_column: str = 'future_return') -> Dict:
        """Run the complete research pipeline."""
        logger.info("Starting Qlib research pipeline...")
        
        # 1. Create factor library
        factors = self.create_factor_library(df)
        logger.info(f"Created {len(factors)} factors")
        
        # 2. Prepare target variable
        if target_column not in df.columns:
            # Create forward return as target
            df['future_return'] = df['close'].pct_change().shift(-1)
            target_column = 'future_return'
        
        target = df[target_column]
        
        # 3. Factor analysis
        analysis = self.factor_analysis(factors, target)
        logger.info("Completed factor analysis")
        
        # 4. Feature selection
        factors_df = pd.DataFrame(factors)
        selected_factors = self.feature_selection(factors_df, target)
        
        # 5. Backtest
        backtest_results = self.backtest_factor_strategy(
            factors, selected_factors, target
        )
        
        # 6. Compile final results
        pipeline_results = {
            "factor_analysis": analysis.to_dict('records'),
            "selected_factors": selected_factors,
            "backtest_results": backtest_results,
            "num_factors_created": len(factors),
            "pipeline_status": "completed"
        }
        
        logger.info("Research pipeline completed successfully")
        return pipeline_results
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def run_qlib_research_example():
    """Example usage of the Qlib research workflow."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    
    price_data = []
    price = 100.0
    
    for _ in range(len(dates)):
        price *= (1 + np.random.normal(0.001, 0.02))
        volume = np.random.randint(1000000, 5000000)
        
        price_data.append({
            'date': dates[len(price_data)],
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(price_data)
    df.set_index('date', inplace=True)
    
    # Run research pipeline
    workflow = QlibResearchWorkflow()
    results = workflow.run_full_research_pipeline(df)
    
    print("Research Pipeline Results:")
    print(f"- Created {results['num_factors_created']} factors")
    print(f"- Selected {len(results['selected_factors'])} top factors")
    
    if 'backtest_results' in results:
        bt = results['backtest_results']
        if 'error' not in bt:
            print(f"- Backtest Sharpe: {bt['sharpe_ratio']:.2f}")
            print(f"- Max Drawdown: {bt['max_drawdown']:.2%}")
            print(f"- Annual Return: {bt['annualized_return']:.2%}")


if __name__ == "__main__":
    run_qlib_research_example()
