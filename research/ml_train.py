# research/ml_train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import numpy as np
import pandas as pd

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