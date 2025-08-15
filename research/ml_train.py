# research/ml_train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import numpy as np
import pandas as pd
import optuna
import logging
from src.config import TradingConfig
from src.utils.main import demo_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_rf_walkforward(df: pd.DataFrame, features, label_col='UpLabel', n_splits=5, save_path='models/rf_wf.pkl'):
    df = df.dropna(subset=features + [label_col]).copy()
    X = df[features].values
    y = df[label_col].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        models.append(model)
    joblib.dump(models, save_path)
    return models

def predict_ensemble(models, X):
    # average probabilities
    p = np.mean([m.predict_proba(X)[:,1] for m in models], axis=0)
    return p

def monte_carlo(returns, paths=1000, horizon=None):
    import numpy as np
    r = np.asarray(returns)
    if r.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    n = len(r) if horizon is None else horizon
    sims = rng.choice(r, size=(paths, n), replace=True)
    eq = np.cumprod(1 + sims, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (peak - eq) / peak
    max_dd = dd.max(axis=1)
    return float(np.median(max_dd)), float(np.percentile(max_dd, 95))

def sharpe(returns, rf=0.0):
    import numpy as np
    if len(returns) == 0:
        return 0.0
    mu = np.mean(returns) * 252
    sigma = np.std(returns) * np.sqrt(252)
    return 0.0 if sigma == 0 else (mu - rf) / sigma



def kelly_fraction(p=0.6, b=1.5, cap=0.01):
    f = (p*b - (1-p)) / b
    return max(0.0, min(f, cap))

def objective(trial, symbol="BTC/USDT", initial_capital=10000):
    """Objective function for Optuna optimization."""
    # Suggest hyperparameters
    ema_period = trial.suggest_int("ema_period", 10, 50)
    atr_period = trial.suggest_int("atr_period", 5, 30)
    keltner_mult = trial.suggest_float("keltner_mult", 1.0, 2.0)
    k_init = trial.suggest_float("k_init", 0.5, 1.5)
    take_profit_atr = trial.suggest_float("take_profit_atr", 1.5, 3.0)

    # Create a temporary config
    cfg = TradingConfig()
    cfg.symbol = symbol
    cfg.INITIAL_CAPITAL = initial_capital
    cfg.EMA_PERIOD = ema_period
    cfg.ATR_PERIOD = atr_period
    cfg.KELTNER_MULT = keltner_mult
    cfg.K_INIT = k_init
    cfg.TAKE_PROFIT_ATR = take_profit_atr

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

def optimize_strategy(symbol="BTC/USDT", n_trials=50):
    """Optimize trading strategy using Optuna."""
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, symbol), n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save best parameters to a file (optional)
    with open("best_params.txt", "w") as f:
        f.write(str(trial.params))

if __name__ == "__main__":
    optimize_strategy()