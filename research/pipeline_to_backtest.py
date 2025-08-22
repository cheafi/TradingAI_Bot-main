"""Load walk-forward models, generate signals and run backtest.

This script is a bridge: models from `research/ml_pipeline.py` are loaded,
applied to historical data to generate buy/sell signals, then passed to
the Backtester in `src/backtesting/backtester.py` for economic evaluation.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import logging
from research import ml_pipeline as mp
from src.backtesting.backtester import Backtester
from src.strategies.signal_strategy import SignalStrategy
from src.utils.risk import sharpe, max_drawdown

logger = logging.getLogger(__name__)


def load_models(path: str):
    """Load ensemble models from joblib pickle."""
    models = joblib.load(path)
    return models


def ensemble_predict(models, X: pd.DataFrame):
    """Average predictions from multiple models."""
    probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probs


def signals_from_probs(probs, threshold=0.55):
    """Convert probabilities to binary signals."""
    return (probs >= threshold).astype(int)


def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float):
    """Calculate trading performance metrics from trades."""
    if trades_df.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0
        }
    
    returns = trades_df['pnl_pct'] if 'pnl_pct' in trades_df else []
    equity = (1 + pd.Series(returns)).cumprod() * initial_capital
    
    total_return = (equity.iloc[-1] / initial_capital - 1) if len(equity) > 0 else 0.0
    ann_return = (1 + total_return) ** (252 / len(equity)) - 1 if len(equity) > 0 else 0.0
    sharpe_ratio = sharpe(returns) if len(returns) > 0 else 0.0
    max_dd = max_drawdown(equity) if len(equity) > 0 else 0.0
    win_rate = (pd.Series(returns) > 0).mean() if len(returns) > 0 else 0.0
    
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "total_trades": len(returns)
    }


def run_pipeline_backtest(symbol: str, model_path: str, start: str, end: str):
    """Run complete pipeline: data -> features -> models -> signals -> backtest."""
    try:
        # Get data and features
        df = mp.fetch_data(symbol, start, end)
        X, y, full = mp.build_features_labels(df)
        
        # Load models and predict
        models = load_models(model_path)
        probs = ensemble_predict(models, X)
        signals = signals_from_probs(probs)
        
        # Prepare data for backtesting
        bt_df = full.copy()
        bt_df["signal"] = signals
        bt_df = bt_df.dropna()
        
        # Run backtest
        initial_capital = 10000
        bt = Backtester(bt_df, SignalStrategy, initial_capital=initial_capital)
        pnl, portfolio_value = bt.run()
        
        # Calculate metrics (simplified for now)
        total_return = (portfolio_value - initial_capital) / initial_capital
        
        # Mock some basic metrics
        metrics = {
            "symbol": symbol,
            "start_date": start,
            "end_date": end,
            "initial_capital": initial_capital,
            "final_portfolio_value": float(portfolio_value),
            "total_pnl": float(pnl),
            "total_return": float(total_return),
            "total_signals": int(signals.sum()),
            "signal_rate": float(signals.mean())
        }
        
        logger.info(f"Backtest completed: {symbol} PnL={pnl:.2f} Return={total_return:.2%}")
        return metrics
        
    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}


if __name__ == "__main__":
    res = run_pipeline_backtest("AAPL", "models/rf_wf.pkl", "2015-01-01", "2023-12-31")
    print(res)
