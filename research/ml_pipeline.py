"""Light baseline ML pipeline.

This file provides a minimal, runnable training pipeline that follows
walk-forward CV best practices. It's intentionally small so it can be
used as a starting point for integrating more advanced workflows from
Stefan Jansen's book.
"""

import argparse
import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
_HAS_TA = True
try:
    import ta
except Exception:
    _HAS_TA = False

_HAS_YFIN = True
try:
    import yfinance as yf
except Exception:
    _HAS_YFIN = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    if not _HAS_YFIN:
        raise RuntimeError("yfinance is not available in this environment")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for {symbol} {start}:{end}")
    df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Momentum
    if _HAS_TA:
        df["rsi_14"] = ta.momentum.rsi(df["Close"], window=14)
        # Volatility (ATR)
        df["atr_14"] = (
            ta.volatility.average_true_range(
                df["High"], df["Low"], df["Close"], window=14
            )
        )
    else:
        # fallback simple RSI/ATR approximations
        df["rsi_14"] = _simple_rsi(df["Close"], 14)
        df["atr_14"] = _simple_atr(df)
    # Trend
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()
    # Returns
    df["ret_1"] = df["Close"].pct_change()
    df["ret_5"] = df["Close"].pct_change(5)
    return df


def _simple_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / down.replace(0, 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _simple_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_features_labels(
    df: pd.DataFrame, ret_horizon: int = 1, up_threshold: float = 0.0
):
    df = df.copy()
    df = add_technical_indicators(df)
    df = df.dropna()
    df["future_ret"] = (
        df["Close"].pct_change(periods=ret_horizon).shift(-ret_horizon)
    )
    df["up_label"] = (df["future_ret"] > up_threshold).astype(int)
    feature_cols = [
        "rsi_14",
        "atr_14",
        "ma_20",
        "ma_50",
        "ma_200",
        "ret_1",
        "ret_5",
    ]
    X = df[feature_cols]
    y = df["up_label"]
    return X, y, df


def train_rf_walkforward(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, save_path: str = None
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    reports = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        reports.append((i, classification_report(y_test, y_pred, output_dict=True)))
        models.append(model)
        logger.info(
            "Fold %d: trained on %d samples, tested on %d samples",
            i,
            len(train_idx),
            len(test_idx),
        )
    if save_path:
        joblib.dump(models, save_path)
        logger.info("Saved %d models to %s", len(models), save_path)
    return models, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--out", default=str(MODEL_DIR / "rf_wf.pkl"))
    args = parser.parse_args()

    df = fetch_data(args.symbol, args.start, args.end)
    X, y, _ = build_features_labels(df)
    models, reports = train_rf_walkforward(
        X, y, n_splits=args.n_splits, save_path=args.out
    )

    # Print simple summary
    for fold, rep in reports:
        print(
            "Fold %d precision for class 1: %.3f, recall: %.3f"
            % (fold, rep["1"]["precision"], rep["1"]["recall"]) 
        )

    print("Saved models to %s" % args.out)


if __name__ == "__main__":
    main()
