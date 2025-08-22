import os
import pandas as pd
import numpy as np
from pathlib import Path

from research import ml_pipeline as mp


def make_synthetic_ohlcv(n=500):
    rng = np.random.default_rng(0)
    prices = np.cumprod(1 + 0.001 * rng.standard_normal(n)) * 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(index=dates)
    df["Open"] = prices * (1 + 0.0005 * rng.standard_normal(n))
    df["High"] = df["Open"] * (1 + 0.001 * np.abs(rng.standard_normal(n)))
    df["Low"] = df["Open"] * (1 - 0.001 * np.abs(rng.standard_normal(n)))
    df["Close"] = prices
    df["Adj Close"] = prices
    df["Volume"] = (1000 * np.abs(rng.standard_normal(n))).astype(int)
    return df


def test_build_features_labels_and_train(tmp_path):
    df = make_synthetic_ohlcv(600)
    # Use pipeline functions directly
    X, y, full = mp.build_features_labels(df)
    assert not X.empty
    assert len(X) == len(y)

    out = tmp_path / "rf_wf_test.pkl"
    models, reports = mp.train_rf_walkforward(X, y, n_splits=3, save_path=str(out))
    assert len(models) == 3
    assert out.exists()
