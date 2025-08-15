"""
Optuna-based optimizer to test parameter combinations and save results.
Usage:
    python optimize.py
This writes optimization_results.csv with columns for parameters + roi/sharpe/max_drawdown.
"""
import optuna
import logging
from pathlib import Path
import csv
from src.config import cfg as base_cfg
from src.utils.main import demo_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUT_CSV = Path("optimization_results.csv")

def build_cfg_from_base(**overrides):
    cfg = base_cfg  # Use the global cfg instance
    for k, v in overrides.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            try:
                cfg[k] = v  # type: ignore
            except Exception:
                pass
    return cfg

def objective(trial):
    """Objective function for Optuna optimization."""
    # Suggest hyperparameters
    ema_period = trial.suggest_int("EMA_PERIOD", 10, 50)
    atr_period = trial.suggest_int("ATR_PERIOD", 5, 30)
    keltner_mult = trial.suggest_float("KELTNER_MULT", 1.0, 2.0)
    k_init = trial.suggest_float("K_INIT", 0.5, 1.5)
    take_profit_atr = trial.suggest_float("TAKE_PROFIT_ATR", 1.5, 3.0)

    # Create a temporary config
    cfg = build_cfg_from_base(
        EMA_PERIOD=ema_period,
        ATR_PERIOD=atr_period,
        KELTNER_MULT=keltner_mult,
        K_INIT=k_init,
        TAKE_PROFIT_ATR=take_profit_atr
    )

    # Run the demo and get results
    try:
        result = demo_run(cfg)
        roi = result.get("roi", -1.0)
        drawdown = result.get("max_drawdown", 1.0)
        sharpe = result.get("sharpe", 0.0)

        # Define the objective: maximize Sharpe ratio, penalize drawdown
        objective_value = sharpe - 0.5 * drawdown  # Adjust weights as needed
        logger.info(f"Trial {trial.number}: ROI={roi:.4f}, Sharpe={sharpe:.4f}, Drawdown={drawdown:.4f}, Objective={objective_value:.4f}")
        return objective_value
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return -1.0  # Penalize failed trials

def optimize_strategy(n_trials=50):
    """Optimize trading strategy using Optuna."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save best parameters to a CSV file
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "value"])
        for key, value in trial.params.items():
            writer.writerow([key, value])

if __name__ == "__main__":
    optimize_strategy()
