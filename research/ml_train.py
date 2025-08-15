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