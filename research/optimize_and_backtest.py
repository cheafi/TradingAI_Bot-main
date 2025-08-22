"""Run Optuna to optimize strategy params using pipeline->backtest bridge."""
import optuna
import logging
from pathlib import Path
from research.pipeline_to_backtest import run_pipeline_backtest

logger = logging.getLogger(__name__)


def objective(trial, symbol="AAPL", start="2020-01-01", end="2023-12-31"):
    """Objective function for Optuna optimization."""
    # Suggest hyperparameters for strategy
    ema_period = trial.suggest_int("ema_period", 10, 50)
    atr_period = trial.suggest_int("atr_period", 5, 30)
    keltner_mult = trial.suggest_float("keltner_mult", 1.0, 2.5)
    prob_threshold = trial.suggest_float("prob_threshold", 0.50, 0.70)
    
    # Store params for potential use in strategy
    trial.set_user_attr("ema_period", ema_period)
    trial.set_user_attr("atr_period", atr_period)
    trial.set_user_attr("keltner_mult", keltner_mult)
    trial.set_user_attr("prob_threshold", prob_threshold)

    try:
        # Run backtest with current parameters
        result = run_pipeline_backtest(symbol, "models/rf_wf.pkl", start, end)
        
        if "error" in result:
            logger.warning(f"Trial failed: {result['error']}")
            return -1.0
            
        # Extract metrics
        total_return = result.get("total_return", 0.0)
        sharpe_ratio = result.get("sharpe_ratio", 0.0)
        max_drawdown = result.get("max_drawdown", 0.0)
        
        # Objective: maximize risk-adjusted return
        # Penalize high drawdown
        objective_value = sharpe_ratio - 2.0 * max_drawdown
        
        logger.info(
            f"Trial {trial.number}: Return={total_return:.2%}, "
            f"Sharpe={sharpe_ratio:.2f}, DD={max_drawdown:.2%}, "
            f"Objective={objective_value:.3f}"
        )
        
        return objective_value
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return -1.0


def optimize(symbol="AAPL", n_trials=20, start="2020-01-01", end="2023-12-31"):
    """Run optimization study."""
    study = optuna.create_study(
        direction="maximize",
        study_name=f"trading_optimization_{symbol}",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize with error handling
    study.optimize(
        lambda trial: objective(trial, symbol, start, end), 
        n_trials=n_trials,
        timeout=3600  # 1 hour timeout
    )
    
    print(f"\nOptimization completed for {symbol}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best objective value: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save study results
    results_path = f"results/optimization_{symbol}_{start}_{end}.pkl"
    Path("results").mkdir(exist_ok=True)
    optuna.study.save_study(study, results_path)
    print(f"Study saved to: {results_path}")
    
    return study


if __name__ == "__main__":
    optimize(10)
