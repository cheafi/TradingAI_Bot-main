import itertools
import pandas as pd
from src.config import TradingConfig
from src.main import demo_run
from typing import Dict, List
import logging

def grid_search() -> List[Dict]:
    """Run grid search for parameter optimization"""
    param_grid = {
        'ema_period': [10, 20, 30],
        'keltner_mult': [1.0, 1.5, 2.0],
        'stop_loss_atr': [1.0, 1.5, 2.0],
        'take_profit_atr': [2.0, 2.5, 3.0]
    }
    
    results = []
    for params in itertools.product(*param_grid.values()):
        cfg = TradingConfig()
        cfg.symbol = "BTC/USDT"
        for key, value in zip(param_grid.keys(), params):
            setattr(cfg, key, value)
            
        try:
            result = demo_run(cfg)
            results.append({
                **{k: v for k, v in zip(param_grid.keys(), params)},
                'roi': result['roi'],
                'sharpe': result['sharpe'],
                'max_drawdown': result['max_drawdown']
            })
        except Exception as e:
            logging.error(f"Error with params {params}: {e}")
    
    return results

def main():
    results = grid_search()
    df = pd.DataFrame(results)
    df.to_csv('optimization_results.csv', index=False)
    
    best_roi = df.loc[df['roi'].idxmax()]
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    
    print("\nBest ROI Parameters:")
    print(best_roi.to_string())
    print("\nBest Sharpe Parameters:")
    print(best_sharpe.to_string())

if __name__ == "__main__":
    main()
