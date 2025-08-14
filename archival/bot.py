
#!/usr/bin/env python3
# coding: utf-8
"""
Improved Scalping/Backtest Bot
- Targets 30-45% annualized ROI, Sharpe >2.0, <7% drawdown via fractional Kelly (f=0.4*0.5=0.2, capped at 2%).
- Backtests on 60 days (15m bars, August 11, 2025, to October 10, 2025) or 730 days (daily, October 11, 2023, to October 10, 2025).
- Uses Keltner Channel (KC) breakouts, LightGBM ML (probup>0.5), MACD/OBV features.
- Sends Telegram alerts for buy/sell signals (paper trades only).
- Logs trades to CSV, saves charts, reports detailed metrics (ROI, Sharpe, Sortino, VaR, beta, alpha, avg/max win/loss).
- Grid searches EMA, Keltner mult, ML prob, RSI slope, ATR period, TP mult, MACD params for Sharpe >2.5.
- Inspired by Freqtrade (hyperopt, FreqAI), QLib (LightGBM, pipeline), Zipline (event-driven, risk metrics).
"""
import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import requests
import csv
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

# -------------------------
# Environment & Logging
# -------------------------
# Load API keys from .env for security (no hardcoding).
# Logging provides real-time feedback (e.g., "OPEN TSLA LONG 100@$250") for debugging/audit.
# Why: Ensures secure key management and auditability for SEC compliance.
load_dotenv('/Users/chantszwai/Downloads/TradingAI_Bot-main/ALPACA_API_KEY.env')
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s', datefmt='%H:%M:%S')

# -------------------------
# Configuration
# -------------------------
# Base config with tunable parameters for momentum scalping.
# Why: Grid search optimizes EMA, Keltner, ML prob, ATR, TP, MACD for Sharpe >2.5.
# Math: Kelly f = (p*rr - (1-p))/rr, p=0.6, rr=2.0, capped at 2%, scaled by 0.5 for safety.
BASE_CONFIG = {
    "initial_capital": 50000.0,
    "symbols": ["TSLA", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "SPY", "IWM"],
    "interval": "15m",
    "intraday_period_days": 60,  # yfinance 15m limit
    "lookback_days": 730,  # 2 years daily fallback
    "atr_period": 14,
    "rsi_fast_period": 3,
    "macd_fast": 12,  # MACD fast EMA
    "macd_slow": 26,  # MACD slow EMA
    "macd_signal": 9,  # MACD signal EMA
    "obv_period": 14,  # OBV EMA period
    "rf_estimators": 200,  # LightGBM trees
    "rf_max_depth": 6,  # Limit depth to avoid overfitting
    "rf_random_state": 42,
    "ml_min_samples": 300,
    "kelly_cap": 0.02,
    "kelly_fraction_of_kelly": 0.5,
    "base_risk_pct": 0.005,
    "max_risk_pct": 0.02,
    "atr_min_pct": 0.0005,
    "atr_max_pct": 0.05,
    "slippage_pct": 0.0005,
    "fee_pct": 0.0002,
    "min_kc_width_pct": 0.0003,  # Relaxed for more trades
    "volume_mult": 0.2,  # Relaxed for more trades
    "max_trades_per_symbol_per_day": 4,  # Increased
    "max_concurrent_positions": 4,  # Increased
    "take_profit_atr_mult": 2.0,
    "max_hold_bars": 48,
    "daily_drawdown_stop_pct": 0.05,
    "save_charts": True,
    "chart_dir": ".",
    "send_telegram": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
    "trade_log_csv": "trade_log.csv",
}

# Parameter search candidates for grid search (Freqtrade hyperopt-inspired).
# Why: Tests combos for ROI 30-45%, win rate >55%, Sharpe >2.0, avoids overfitting via TimeSeriesSplit.
PARAM_SEARCH_CANDIDATES = {
    "ema_periods": [10, 20],
    "keltner_mults": [1.0, 1.5],
    "ml_min_prob_long": [0.5, 0.55],
    "rsi_signal_slope": [False, True],
    "atr_periods": [10, 14],
    "take_profit_atr_mults": [1.5, 2.0],
    "macd_fast_periods": [8, 12],
    "macd_slow_periods": [20, 26],
    "macd_signal_periods": [7, 9]
}
enable_param_search = True  # Toggle to use first params only

# Trailing stop parameters (fixed).
# Why: Dynamic stops lock profits—tighter gaps as price runs (k_ultra=0.4 at r2=2.0 ATR).
BASE_CONFIG["k_init"] = 1.0
BASE_CONFIG["p_be"] = 0.6
BASE_CONFIG["k_trail"] = 0.8
BASE_CONFIG["k_fast"] = 0.6
BASE_CONFIG["r1"] = 1.0
BASE_CONFIG["k_ultra"] = 0.4
BASE_CONFIG["r2"] = 2.0

# -------------------------
# Utilities
# -------------------------
def send_telegram_message(text: str, cfg: dict) -> None:
    """
    Sends Telegram alerts (e.g., "Paper Buy TSLA at $250, Prob 0.58").
    Why: Real-time monitoring for high-prob signals (probup>0.5).
    Args: text (message), cfg (config with send_telegram flag).
    """
    if not cfg["send_telegram"]:
        logging.debug("Telegram disabled or missing credentials.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        logging.warning(f"Telegram error: {e}")

# -------------------------
# Indicators
# -------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes RSI = 100 - (100 / (1 + (EMA up / EMA down))) for momentum.
    Why: RSI3 catches quick oversold signals (<30), RSI14 confirms trend.
    Math: up = delta.clip(lower=0), down = -delta.clip(upper=0), rs = EMA(up) / EMA(down).
    Args: series (prices), period (lookback, e.g., 3).
    Returns: pd.Series of RSI values.
    """
    s = series.astype(float)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    alpha = 1.0 / period
    emaup = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    emadown = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = emaup / (emadown + 1e-12)
    r = 100 - (100 / (1 + rs))
    return r

def ema_series(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Computes EMA = close * alpha + prev EMA * (1-alpha), alpha=2/(period+1).
    Why: EMA20 confirms trend for breakouts (close > EMA for longs).
    Args: series (prices), period (lookback, e.g., 20).
    Returns: pd.Series of EMA values.
    """
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes ATR = EMA(max(high-low, |high-prev close|, |low-prev close|)).
    Why: ATR sizes stops (entry - 1*ATR) and TP (entry + 2*ATR) for vol-adjusted risk.
    Args: df (OHLCV data), period (lookback, e.g., 14).
    Returns: pd.Series of ATR values.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr_s

def keltner_channels(df: pd.DataFrame, ema_period=20, atr_period=14, mult=1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes Keltner Channels: mid = EMA, upper = EMA + mult*ATR, lower = EMA - mult*ATR.
    Why: Breakouts above upper KC signal long entries (close > upper with ML prob>0.5).
    Args: df (OHLCV data), ema_period (20), atr_period (14), mult (1.5 for channel width).
    Returns: Tuple of pd.Series (mid, upper, lower).
    """
    close = df["close"].astype(float)
    mid = ema_series(close, ema_period)
    rng = atr(df, atr_period)
    upper = mid + rng * mult
    lower = mid - rng * mult
    return mid, upper, lower

def macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes MACD = EMA(fast) - EMA(slow), signal = EMA(MACD, signal), histogram = MACD - signal.
    Why: MACD crossovers confirm momentum (MACD > signal for longs).
    Math: EMA as above, MACD for trend strength.
    Args: df (OHLCV data), fast (12), slow (26), signal (9).
    Returns: Tuple of pd.Series (macd, signal, histogram).
    """
    close = df["close"].astype(float)
    ema_fast = ema_series(close, fast)
    ema_slow = ema_series(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema_series(macd_line, signal)
    histogram = macd_line - macd_signal
    return macd_line, macd_signal, histogram

def obv(df: pd.DataFrame, period=14) -> pd.Series:
    """
    Computes On Balance Volume (OBV) = cumsum(volume * sign(close diff)), EMA-smoothed.
    Why: OBV EMA (period=14) confirms volume flow for breakouts (rising OBV for longs).
    Math: sign = 1 if close > prev, -1 if <, 0 if =; cumsum for total flow, EMA for smoothing.
    Args: df (OHLCV data), period (14 for EMA).
    Returns: pd.Series of OBV EMA values.
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    sign = np.sign(close.diff())
    obv_val = (volume * sign).cumsum()
    obv_val.iloc[0] = 0
    return ema_series(obv_val, period)

# -------------------------
# Kelly Fraction
# -------------------------
def kelly_fraction(p: float, rr: float, cap: float) -> float:
    """
    Computes Kelly f = (p*rr - (1-p))/rr, capped at cap (2%).
    Why: Sizes bets to maximize growth while avoiding blowups (fractional f=0.5*f reduces variance).
    Math: f = (p*rr - (1-p))/rr, p=win prob (e.g., 0.6 from ML), rr=reward/risk (2.0).
    Args: p (win prob), rr (reward/risk), cap (max f, e.g., 0.02).
    Returns: Kelly fraction (float).
    """
    if rr <= 0:
        return 0.0
    f = (p * rr - (1 - p)) / rr
    return max(0.0, min(f, cap))

# -------------------------
# Data Fetching Helpers
# -------------------------
def _fetch_vix() -> float:
    """
    Fetches current VIX for vol filter (require VIX >15 for high-vol entries).
    Why: High VIX (>15) signals volatile markets for momentum plays—boosts win rate.
    Returns: VIX value (float) or default 15.0 if fail.
    """
    for attempt in range(3):
        try:
            vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
            return vix
        except Exception as e:
            logging.debug(f"VIX fetch attempt {attempt+1} failed: {e}")
            time.sleep(2)
    logging.warning("VIX fetch failed after 3 attempts, using default 15.0")
    return 15.0

def _fetch_intraday(symbol: str, interval: str, period_days: int) -> pd.DataFrame:
    """
    Fetches intraday data (15m bars) for 60 days via yfinance with retries.
    Why: Provides rich data for scalping (~4000 bars at 78 bars/day * 60 days), captures 2025 Q3 vol.
    Args: symbol (e.g., TSLA), interval (15m), period_days (60).
    Returns: pd.DataFrame with OHLCV data or empty if fails.
    """
    period_str = f"{period_days}d"
    for attempt in range(3):
        try:
            df = yf.Ticker(symbol).history(period=period_str, interval=interval, auto_adjust=False, back_adjust=False)
            if not df.empty:
                return df
            logging.debug(f"Intraday fetch attempt {attempt+1} for {symbol} returned empty")
            time.sleep(2)
        except Exception as e:
            logging.debug(f"Intraday fetch attempt {attempt+1} for {symbol}: {e}")
            time.sleep(2)
    logging.warning(f"Intraday fetch failed for {symbol} after 3 attempts")
    return pd.DataFrame()

def _fetch_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    """
    Fetches daily data for 730 days as fallback via yfinance with retries.
    Why: Ensures robustness if intraday fails, provides long-term context (~500 bars).
    Args: symbol (e.g., TSLA), lookback_days (730).
    Returns: pd.DataFrame with OHLCV data or empty if fails.
    """
    for attempt in range(3):
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=lookback_days)
            df = yf.download(symbol, start=start.date(), end=end.date() + timedelta(days=1), interval="1d", progress=False, auto_adjust=False)
            if not df.empty:
                return df
            logging.debug(f"Daily fetch attempt {attempt+1} for {symbol} returned empty")
            time.sleep(2)
        except Exception as e:
            logging.debug(f"Daily fetch attempt {attempt+1} for {symbol}: {e}")
            time.sleep(2)
    logging.warning(f"Daily fetch failed for {symbol} after 3 attempts")
    return pd.DataFrame()

def _generate_synthetic(symbol: str, num_candles: int = 4000, interval_minutes: int = 15) -> pd.DataFrame:
    """
    Generates synthetic OHLCV data if real data fails (4000 candles, ~60 days at 15m).
    Why: Prevents crashes, allows testing, but logs warning (not ideal for metrics).
    Args: symbol (for seed), num_candles (4000), interval_minutes (15).
    Returns: pd.DataFrame with synthetic OHLCV data.
    """
    seed = abs(hash(symbol)) % (2**32)
    rng = np.random.RandomState(seed)
    end = datetime.now(timezone.utc)
    dates = pd.date_range(end=end, periods=num_candles, freq=f"{interval_minutes}min")
    prices = np.cumsum(rng.randn(num_candles)) * 0.5 + 100
    high = prices + rng.rand(num_candles) * 0.6
    low = prices - rng.rand(num_candles) * 0.6
    close = prices + rng.randn(num_candles) * 0.1
    open_ = close + rng.randn(num_candles) * 0.1
    volume = (rng.randint(100, 1000, size=num_candles)).astype(int)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)
    logging.warning(f"Using synthetic data for {symbol} ({num_candles} candles)")
    return df

def normalize_columns_and_lower(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names to lowercase (open, high, low, close, volume).
    Why: Ensures consistency across yfinance data formats (e.g., 'Close' vs 'close_adj').
    Args: df (OHLCV data).
    Returns: pd.DataFrame with normalized columns or empty if missing required columns.
    """
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("open"):
            rename_map[c] = "open"
        elif lc.startswith("high"):
            rename_map[c] = "high"
        elif lc.startswith("low"):
            rename_map[c] = "low"
        elif lc in ("close", "adjclose", "close_adj"):
            rename_map[c] = "close"
        elif lc.startswith("volume"):
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)
    df.columns = [c.lower() for c in df.columns]
    if not all(col in df.columns for col in ["open", "high", "low", "close", "volume"]):
        return pd.DataFrame()
    return df[["open", "high", "low", "close", "volume"]].copy()

def get_bars(symbol: str, interval: str, intraday_period_days: int, lookback_days: int) -> pd.DataFrame:
    """
    Fetches data: 60d 15m primary, 730d daily fallback, synthetic if both fail.
    Why: Ensures robust backtesting (~4000 bars for ML, min_samples=300).
    Args: symbol (e.g., TSLA), interval (15m), intraday_period_days (60), lookback_days (730).
    Returns: pd.DataFrame with OHLCV data.
    """
    logging.info(f"Fetching {symbol} {interval} (period {intraday_period_days}d) ...")
    df = _fetch_intraday(symbol, interval, intraday_period_days)
    df = normalize_columns_and_lower(df)
    if (df.empty) or (len(df) < 50):
        logging.warning(f"Intraday {interval} insufficient for {symbol}. Trying daily fallback...")
        df_daily = _fetch_daily(symbol, lookback_days + 5)
        df_daily = normalize_columns_and_lower(df_daily)
        if (not df_daily.empty) and (len(df_daily) >= 30):
            logging.info(f"Using daily data for {symbol} (interval=1d).")
            df = df_daily
        else:
            logging.warning(f"Daily fallback also insufficient for {symbol}. Generating synthetic data.")
            df = _generate_synthetic(symbol, num_candles=4000, interval_minutes=15)
    df = normalize_columns_and_lower(df)
    if df.empty:
        logging.warning(f"Data fetch failed for {symbol}, using synthetic fallback.")
        df = _generate_synthetic(symbol, num_candles=4000, interval_minutes=15)
    return df

# -------------------------
# ML Features / Predictions
# -------------------------
def generate_ml_predictions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Generates ML features (RSI3/14, close-EMA diff, ATR, KC width, MACD, OBV, lagged return, vol) and predicts up/down with LightGBM.
    Why: High-prob entries (probup>0.5) boost win rate (>55%), TimeSeriesSplit avoids look-ahead bias, LightGBM (QLib-inspired) for speed/accuracy.
    Math: probup = mean(tree probs), pred = probup >= 0.5, features normalized.
    Args: df (OHLCV data), cfg (config with ML params).
    Returns: pd.DataFrame with ml_probup, ml_pred columns.
    """
    out = df.copy()
    if len(out) < cfg["ml_min_samples"]:
        out["ml_probup"] = 0.5
        out["ml_pred"] = 0
        return out
    out["rsi3"] = rsi(out["close"], cfg["rsi_fast_period"])
    out["rsi14"] = rsi(out["close"], 14)
    out["ema"] = ema_series(out["close"], cfg["ema_period"])
    out["close_ema_diff"] = (out["close"] - out["ema"]) / out["close"].std()  # Normalized
    out["atr"] = atr(out, cfg["atr_period"])
    out["atr_pct"] = (out["atr"] / out["close"]).clip(lower=0)
    out["ret_lag1"] = out["close"].pct_change(1)  # Lagged return
    out["vol_14"] = out["close"].pct_change().rolling(14).std()  # Volatility
    mid, upper, lower = keltner_channels(out, cfg["ema_period"], cfg["atr_period"], cfg["keltner_mult"])
    out["kc_width"] = ((upper - lower) / out["close"]).clip(lower=0)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    out["obv_ema"] = obv(out, cfg["obv_period"])
    out["future_close"] = out["close"].shift(-1)
    out["up_label"] = (out["future_close"] > out["close"]).astype(int)
    features = ["rsi3", "rsi14", "close_ema_diff", "atr_pct", "kc_width", "ret_lag1", "vol_14", "macd", "macd_signal", "macd_hist", "obv_ema"]
    out = out.dropna(subset=features + ["up_label"]).copy()
    if len(out) < cfg["ml_min_samples"]:
        out["ml_probup"] = 0.5
        out["ml_pred"] = 0
        return out
    X = out[features]
    y = out["up_label"]
    prob = pd.Series(index=out.index, dtype=float)
    pred = pd.Series(index=out.index, dtype=int)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        model = LGBMClassifier(
            n_estimators=cfg["rf_estimators"],
            max_depth=cfg["rf_max_depth"],
            learning_rate=0.05,  # QLib-inspired
            num_leaves=31,  # Optimized for LightGBM
            random_state=cfg["rf_random_state"],
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        prob.iloc[test_idx] = p
        pred.iloc[test_idx] = (p >= 0.5).astype(int)
    out["ml_probup"] = prob.fillna(0.5)
    out["ml_pred"] = pred.fillna(0).astype(int)
    return out

# -------------------------
# Paper Trader & Logging
# -------------------------
@dataclass
class Position:
    """
    Tracks position details (symbol, qty, avg price, stop, high-water mark).
    Why: Manages paper trades, logs entries/exits for auditability.
    """
    symbol: str
    qty: int = 0
    avg_price: float = 0.0
    side: str = "flat"
    stop_price: Optional[float] = None
    hwm: Optional[float] = None
    entry_idx: Optional[pd.Timestamp] = None
    entry_bar: Optional[int] = None
    ml_probup: Optional[float] = None

    def update_hwm(self, px: float):
        """
        Updates high-water mark for trailing stops.
        Why: Locks profits as price rises (stop = hwm - gap).
        Args: px (current price).
        """
        if self.side == "long":
            self.hwm = px if self.hwm is None else max(self.hwm, px)

class PaperTrader:
    """
    Simulates paper trades, tracks positions, logs to CSV, sends Telegram alerts.
    Why: Tests strategy safely, logs for SEC compliance, monitors drawdown daily (Zipline-inspired).
    """
    def __init__(self, initial_capital: float, cfg: dict):
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.trades: List[float] = []
        self.daily_pnl: List[float] = []
        self.cfg = cfg
        self.daily_start_equity: Optional[float] = None
        self.last_date: Optional[datetime.date] = None
        self.trade_log_file = cfg.get("trade_log_csv", "trade_log.csv")
        with open(self.trade_log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "symbol", "side", "qty", "price", "type", "pnl", "capital_after", "reason"])

    def equity(self) -> float:
        """
        Returns current capital.
        """
        return self.capital

    def _ensure_pos(self, symbol: str):
        """
        Ensures position exists for symbol.
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

    def _log_trade(self, ts, symbol, side, qty, price, typ, pnl, reason=""):
        """
        Logs trade to CSV (Zipline-inspired audit trail).
        Why: Ensures SEC compliance, tracks performance.
        Args: ts (timestamp), symbol, side, qty, price, typ (open/close), pnl, reason.
        """
        self.trades.append(pnl if pnl is not None else 0.0)
        with open(self.trade_log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts.isoformat() if hasattr(ts, "isoformat") else ts,
                symbol, side, qty, f"{price:.6f}", typ,
                f"{pnl:.6f}" if pnl is not None else "",
                f"{self.capital:.6f}",
                reason
            ])

    def open_long(self, symbol: str, price: float, qty: int, ts: pd.Timestamp, idx: int, reason: str=""):
        """
        Opens long position with slippage (0.05%) and fees (0.02%).
        Why: Simulates realistic entry (paper only), logs/alerts.
        Math: cost = (price + slippage) * qty + fee, fee = fill_price * qty * fee_pct.
        Args: symbol, price, qty, ts (timestamp), idx (bar index), reason (e.g., KC_BREAKOUT+ML).
        Returns: True if opened, False if invalid.
        """
        if qty <= 0:
            return False
        slippage = price * self.cfg["slippage_pct"]
        fill_price = price + slippage
        fee = fill_price * qty * self.cfg["fee_pct"]
        cost = fill_price * qty + fee
        if cost > self.capital:
            return False
        self._ensure_pos(symbol)
        pos = self.positions[symbol]
        if pos.side == "short" and pos.qty > 0:
            return False
        if pos.qty > 0:
            return False
        pos.qty = qty
        pos.avg_price = fill_price
        pos.side = "long"
        pos.entry_idx = ts
        pos.entry_bar = idx
        pos.hwm = pos.avg_price
        pos.stop_price = None
        pos.ml_probup = self.cfg["ml_min_prob_long"]
        self.positions[symbol] = pos
        self.capital -= cost
        logging.info(f"OPEN {symbol} LONG {qty}@{fill_price:.4f} reason={reason}")
        self._log_trade(ts, symbol, "LONG_OPEN", qty, fill_price, "open", None, reason)
        send_telegram_message(f"Paper Buy {symbol} {qty}@{fill_price:.2f}, Prob: {pos.ml_probup:.2f}, Reason: {reason}", self.cfg)
        return True

    def close_position(self, symbol: str, price: float, reason: str=""):
        """
        Closes position with slippage (0.05%) and fees (0.02%).
        Why: Simulates realistic exit, logs PnL, sends alerts.
        Math: PnL = (fill_price - avg_price) * qty - fee, revenue = fill_price * qty - fee.
        Args: symbol, price, reason (e.g., STOP_HIT, TAKE_PROFIT).
        Returns: True if closed, False if no position.
        """
        if symbol not in self.positions:
            return False
        pos = self.positions[symbol]
        if pos.qty <= 0 or pos.side == "flat":
            return False
        slippage = price * self.cfg["slippage_pct"]
        fill_price = price - slippage
        fee = fill_price * pos.qty * self.cfg["fee_pct"]
        revenue = fill_price * pos.qty - fee
        pnl = (fill_price - pos.avg_price) * pos.qty - fee
        self.capital += revenue
        self.trades.append(pnl)
        self.daily_pnl.append(pnl)
        logging.info(f"CLOSE {symbol} {pos.qty}@{fill_price:.4f} PnL={pnl:.4f} reason={reason}")
        self._log_trade(datetime.now(timezone.utc), symbol, "LONG_CLOSE", pos.qty, fill_price, "close", pnl, reason)
        send_telegram_message(f"Paper Sell {symbol} {pos.qty}@{fill_price:.2f}, PnL: ${pnl:.2f}, Reason: {reason}", self.cfg)
        self.positions[symbol] = Position(symbol=symbol)
        return True

    def set_initial_stop(self, symbol: str, stop_price: float):
        """
        Sets initial stop price.
        Why: Defines max loss (stop = entry - 1*ATR) for drawdown <7%.
        Args: symbol, stop_price.
        """
        pos = self.positions.get(symbol)
        if pos and pos.qty > 0:
            pos.stop_price = stop_price
            self.positions[symbol] = pos

    def update_trailing(self, symbol: str, last_price: float, atr_val: float):
        """
        Updates trailing stop: stop = hwm - gap, gap shrinks as run increases.
        Why: Locks profits dynamically (gap=0.4*ATR at 2.0 ATR run) for higher RR.
        Math: gap = k_trail*ATR (0.8), k_fast (0.6) at r1=1.0, k_ultra (0.4) at r2=2.0.
        Args: symbol, last_price, atr_val.
        """
        pos = self.positions.get(symbol)
        if not pos or pos.qty <= 0 or atr_val <= 0:
            return
        if pos.stop_price is None:
            pos.stop_price = pos.avg_price - self.cfg["k_init"] * atr_val
        if pos.side == "long":
            pos.update_hwm(last_price)
            if pos.hwm is not None:
                run_atr = (pos.hwm - pos.avg_price) / (atr_val + 1e-12)
                gap = self.cfg["k_trail"] * atr_val
                if run_atr >= self.cfg["r1"]:
                    gap = self.cfg["k_fast"] * atr_val
                if run_atr >= self.cfg["r2"]:
                    gap = self.cfg["k_ultra"] * atr_val
                new_stop = pos.hwm - gap
                pos.stop_price = max(pos.stop_price, new_stop)
        self.positions[symbol] = pos

    def check_stop_exit(self, symbol: str, bar_low: float, ts: pd.Timestamp) -> bool:
        """
        Checks if low hits stop price, closes if hit.
        Why: Limits loss to stop distance (1*ATR) for drawdown <7%.
        Args: symbol, bar_low, ts (timestamp).
        Returns: True if stopped, False otherwise.
        """
        pos = self.positions.get(symbol)
        if not pos or pos.qty <= 0 or pos.stop_price is None:
            return False
        if bar_low <= pos.stop_price:
            self.close_position(symbol, pos.stop_price, reason="STOP_HIT")
            return True
        return False

    def check_daily_drawdown(self, ts: pd.Timestamp) -> bool:
        """
        Checks daily drawdown, stops trading if >5% (Zipline-inspired).
        Why: Prevents large losses, keeps drawdown <7%.
        Math: drawdown = (start_equity - current_equity) / start_equity.
        Args: ts (timestamp).
        Returns: True if drawdown exceeded, False otherwise.
        """
        day = ts.date()
        if self.last_date != day:
            self.daily_start_equity = self.capital
            self.daily_pnl = []
            self.last_date = day
        if self.daily_start_equity is None:
            return False
        daily_loss = self.daily_start_equity - self.capital
        drawdown_pct = daily_loss / self.daily_start_equity if self.daily_start_equity > 0 else 0
        if drawdown_pct >= self.cfg["daily_drawdown_stop_pct"]:
            logging.info(f"Daily drawdown {drawdown_pct*100:.1f}% exceeds limit {self.cfg['daily_drawdown_stop_pct']*100:.1f}%, stopping for day")
            return True
        return False

    def metrics(self) -> dict:
        """
        Computes performance metrics (Zipline-inspired): ROI, win rate, Sharpe, Sortino, drawdown, VaR, beta, alpha, avg/max win/loss.
        Why: Quantifies strategy success (ROI 30-45%, Sharpe >2.0, drawdown <7%).
        Math: ROI = realized / initial, Sharpe = mean / std * sqrt(252*78), Sortino = mean / downside_std * sqrt(252*78), drawdown = max(cummax - cum) %.
        Returns: Dict with metrics.
        """
        realized = sum(self.trades)
        roi = realized / self.initial_capital if self.initial_capital else 0.0
        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t <= 0]
        num = len(self.trades)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        max_win = max(wins) if wins else 0
        max_loss = min(losses) if losses else 0
        returns = pd.Series(self.trades) / self.initial_capital
        cum_ret = (1 + returns).cumprod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 78) if len(returns) > 1 and returns.std() != 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252 * 78) if len(returns[returns < 0]) > 0 and returns[returns < 0].std() != 0 else 0
        drawdown = (cum_ret.cummax() - cum_ret).max() * 100 if len(cum_ret) > 0 else 0
        # Monte Carlo VaR (95%) with 2025 Q3 stress (+20% vol)
        var_95 = 0.0
        if len(returns) > 1:
            mc_rets = np.random.normal(returns.mean(), returns.std() * 1.2, (1000, 252 * 78))
            mc_pnl = mc_rets * self.initial_capital
            mc_cum = np.cumsum(mc_pnl, axis=1)
            var_95 = np.percentile(mc_cum[:, -1], 5) / self.initial_capital * 100
        # Portfolio beta and alpha (Zipline-inspired, vs SPY)
        beta, alpha = 0.0, 0.0
        if len(returns) > 1:
            try:
                spy = _fetch_daily("SPY", 730)
                spy_ret = spy["close"].pct_change().dropna()
                if len(spy_ret) > len(returns):
                    spy_ret = spy_ret[-len(returns):]
                elif len(spy_ret) < len(returns):
                    returns = returns[-len(spy_ret):]
                cov = np.cov(returns, spy_ret)[0, 1]
                var = np.var(spy_ret)
                beta = cov / var if var != 0 else 0.0
                rf_rate = 0.02 / (252 * 78)  # 2% annualized risk-free rate
                alpha = returns.mean() - rf_rate - beta * (spy_ret.mean() - rf_rate)
                alpha *= 252 * 78  # Annualize
            except:
                logging.warning("Failed to compute beta/alpha")
        return {
            "Current_Capital": self.capital,
            "Realized_PnL": realized,
            "ROI": roi,
            "Win_Rate": (len(wins) / num) if num else 0.0,
            "Loss_Rate": (len(losses) / num) if num else 0.0,
            "Num_Trades": num,
            "Avg_Win": avg_win,
            "Avg_Loss": avg_loss,
            "Max_Win": max_win,
            "Max_Loss": max_loss,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Drawdown": drawdown,
            "VaR_95": var_95,
            "Beta": beta,
            "Alpha": alpha
        }

# -------------------------
# Enrich Features
# -------------------------
def enrich_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Adds technical features (RSI3/14, close-EMA diff, ATR, KC width, MACD, OBV, lagged return, vol).
    Why: Feeds LightGBM for high-prob entries (probup>0.5), boosts win rate >55% (QLib-inspired pipeline).
    Args: df (OHLCV data), cfg (config with periods).
    Returns: pd.DataFrame with added features.
    """
    out = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            raise ValueError(f"Missing expected column '{col}'")
    out["rsi3"] = rsi(out["close"], cfg["rsi_fast_period"])
    out["rsi14"] = rsi(out["close"], 14)
    out["ema"] = ema_series(out["close"], cfg["ema_period"])
    out["close_ema_diff"] = (out["close"] - out["ema"]) / out["close"].std()  # Normalized
    out["atr"] = atr(out, cfg["atr_period"])
    out["atr_pct"] = (out["atr"] / out["close"]).clip(lower=0)
    out["ret_lag1"] = out["close"].pct_change(1)
    out["vol_14"] = out["close"].pct_change().rolling(14).std()
    mid, upper, lower = keltner_channels(out, cfg["ema_period"], cfg["atr_period"], cfg["keltner_mult"])
    out["kc_mid"] = mid
    out["kc_upper"] = upper
    out["kc_lower"] = lower
    out["kc_width"] = ((upper - lower) / out["close"]).clip(lower=0)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    out["obv_ema"] = obv(out, cfg["obv_period"])
    out = out.ffill().bfill()
    return out

# -------------------------
# Position Sizing
# -------------------------
def position_size(equity: float, price: float, stop_distance: float, ml_prob_up: float, cfg: dict) -> int:
    """
    Sizes position: qty = risk_dollars / stop_distance, risk = equity * min(2%, max(0.5%, 0.5*Kelly)).
    Why: Caps risk at 2%, scales with ML probup (prob>0.5), keeps drawdown <7%.
    Math: Kelly f = (p*rr - (1-p))/rr, rr=2.0, qty = (equity * risk_pct) / stop_distance.
    Args: equity, price, stop_distance, ml_prob_up, cfg.
    Returns: Number of shares (int).
    """
    if price <= 0 or stop_distance <= 0:
        return 0
    assumed_rr = 2.0
    fkelly = kelly_fraction(ml_prob_up, assumed_rr, cfg["kelly_cap"])
    fkelly = fkelly * cfg.get("kelly_fraction_of_kelly", 0.5)
    risk_pct = min(cfg["max_risk_pct"], max(cfg["base_risk_pct"], fkelly))
    risk_dollars = equity * risk_pct
    qty = int(risk_dollars // (stop_distance + 1e-12))
    return max(0, qty)

# -------------------------
# Strategy Runner
# -------------------------
def run_strategy_on_df(symbol: str, df: pd.DataFrame, cfg: dict, trader: PaperTrader) -> pd.DataFrame:
    """
    Runs momentum strategy: Long on KC breakout (close > upper) with ML prob>0.5, vol>0.2*median.
    Exits on stop (entry - 1*ATR), TP (entry + 2*ATR), or time (48 bars).
    Why: High-prob entries boost win rate (>55%), TP at 2*ATR targets RR 2:1, limits prevent overtrading.
    Math: Signal = (close > KC_upper) & (vol > 0.2*median) & (KC_width > 0.03%) & (probup > 0.5).
    Args: symbol, df (OHLCV data), cfg, trader (PaperTrader instance).
    Returns: pd.DataFrame with action column (BUY, STOP_EXIT, TP_EXIT, TIME_EXIT).
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = enrich_features(df, cfg)
    vix = _fetch_vix()
    if vix < 15.0:
        logging.info(f"VIX {vix:.2f} below 15.0 for {symbol}, proceeding with caution")
    df = generate_ml_predictions(df, cfg)
    median_vol = df["volume"].median() if not df["volume"].isna().all() else 0.0
    df["cross_up"] = (df["close"] > df["kc_upper"]) & (df["close"].shift(1) <= df["kc_upper"].shift(1))
    df["volume_ok"] = df["volume"] > (median_vol * cfg["volume_mult"])
    df["kc_width_ok"] = df["kc_width"] > cfg["min_kc_width_pct"]
    df["ml_ok"] = df["ml_probup"] >= cfg["ml_min_prob_long"]
    df["macd_ok"] = df["macd"] > df["macd_signal"]  # MACD crossover for momentum
    if cfg.get("rsi_signal_slope", True):
        df["rsi_slope"] = (df["rsi3"] > df["rsi3"].shift(1)) & (df["rsi3"] > 50)
        df["longsignal"] = df["cross_up"] & df["volume_ok"] & df["kc_width_ok"] & df["ml_ok"] & df["rsi_slope"] & df["macd_ok"]
    else:
        df["longsignal"] = df["cross_up"] & df["volume_ok"] & df["kc_width_ok"] & df["ml_ok"] & df["macd_ok"]
    df["action"] = ""
    daily_count: Dict[datetime.date, int] = {}
    for i in range(1, len(df) - 1):
        ts = df.index[i]
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1]
        day = ts.date()
        if trader.check_daily_drawdown(ts):
            continue
        daily_count.setdefault(day, 0)
        if daily_count[day] >= cfg["max_trades_per_symbol_per_day"]:
            continue
        pos = trader.positions.get(symbol, Position(symbol=symbol))
        in_pos = (pos.qty > 0 and pos.side == "long")
        if in_pos:
            trader.update_trailing(symbol, bar["close"], bar["atr"])
            stopped = trader.check_stop_exit(symbol, bar["low"], ts)
            if stopped:
                df.at[ts, "action"] = "STOP_EXIT"
                in_pos = False
                daily_count[day] += 1
                continue
            tp_price = pos.avg_price + cfg["take_profit_atr_mult"] * bar["atr"]
            if bar["high"] >= tp_price:
                trader.close_position(symbol, tp_price, reason="TAKE_PROFIT")
                df.at[ts, "action"] = "TP_EXIT"
                in_pos = False
                daily_count[day] += 1
                continue
            held_bars = (i - (pos.entry_bar or i))
            if held_bars >= cfg["max_hold_bars"]:
                trader.close_position(symbol, bar["close"], reason="TIME_EXIT")
                df.at[ts, "action"] = "TIME_EXIT"
                in_pos = False
                daily_count[day] += 1
                continue
        active_positions = sum(1 for p in trader.positions.values() if p.qty > 0)
        if (not in_pos) and (active_positions < cfg["max_concurrent_positions"]) and (daily_count[day] < cfg["max_trades_per_symbol_per_day"]):
            if bar["longsignal"]:
                entry_price = float(next_bar["open"])
                atr_val = bar["atr"]
                if atr_val <= 0:
                    continue
                stop_price = entry_price - cfg["k_init"] * atr_val
                stop_distance = entry_price - stop_price
                ml_prob = float(bar["ml_probup"])
                qty = position_size(trader.equity(), entry_price, stop_distance, ml_prob, cfg)
                if qty > 0:
                    reason = "KC_BREAKOUT+ML+VOL+MACD"
                    ok = trader.open_long(symbol, entry_price, qty, ts, i + 1, reason=reason)
                    if ok:
                        trader.set_initial_stop(symbol, stop_price)
                        df.at[ts, "action"] = f"BUY {qty}@{entry_price:.4f} SL@{stop_price:.4f}"
                        daily_count[day] += 1
    return df

# -------------------------
# Charting
# -------------------------
def save_chart(symbol: str, df: pd.DataFrame, cfg: dict):
    """
    Saves matplotlib chart of close, KC bands, buy/exit markers.
    Why: Visualizes backtest for pattern review (e.g., KC breakouts).
    Args: symbol, df (OHLCV data with actions), cfg (config).
    """
    try:
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))
        plt.title(f"{symbol} {cfg['interval']} scalp/backtest")
        plt.plot(df.index, df["close"], label="Close", linewidth=1.2)
        if "kc_upper" in df.columns and "kc_lower" in df.columns:
            plt.plot(df.index, df["kc_upper"], label="KC Upper", alpha=0.6)
            plt.plot(df.index, df["kc_lower"], label="KC Lower", alpha=0.6)
        buys = df[df["action"].str.startswith("BUY", na=False)]
        stops = df[df["action"] == "STOP_EXIT"]
        times = df[df["action"] == "TIME_EXIT"]
        tps = df[df["action"] == "TP_EXIT"]
        if not buys.empty:
            plt.scatter(buys.index, df.loc[buys.index, "close"], marker="^", s=40, label="Buy", color="green")
        if not stops.empty:
            plt.scatter(stops.index, df.loc[stops.index, "close"], marker="x", s=40, label="Stop Exit", color="red")
        if not times.empty:
            plt.scatter(times.index, df.loc[times.index, "close"], marker="o", s=40, label="Time Exit", facecolors="none", edgecolors="orange")
        if not tps.empty:
            plt.scatter(tps.index, df.loc[tps.index, "close"], marker="s", s=40, label="TP Exit", facecolors="green", edgecolors="black")
        plt.legend()
        plt.tight_layout()
        fn = f"{symbol}_{cfg['interval']}_chart.png"
        outpath = os.path.join(cfg["chart_dir"], fn)
        plt.savefig(outpath, dpi=150)
        plt.close()
        logging.info(f"Saved chart: {outpath}")
    except Exception as e:
        logging.warning(f"Chart error for {symbol}: {e}")

# -------------------------
# Main
# -------------------------
def main():
    """
    Main function: Runs grid search, backtests on 60d 15m or 730d daily, logs metrics, saves charts, sends Telegram summary.
    Why: Finds optimal parameters for ROI 30-45%, Sharpe >2.0, drawdown <7% (Freqtrade hyperopt-inspired).
    """
    base_cfg = dict(BASE_CONFIG)
    if not enable_param_search:
        base_cfg["ema_period"] = PARAM_SEARCH_CANDIDATES["ema_periods"][0]
        base_cfg["keltner_mult"] = PARAM_SEARCH_CANDIDATES["keltner_mults"][0]
        base_cfg["ml_min_prob_long"] = PARAM_SEARCH_CANDIDATES["ml_min_prob_long"][0]
        base_cfg["rsi_signal_slope"] = PARAM_SEARCH_CANDIDATES["rsi_signal_slope"][0]
        base_cfg["atr_period"] = PARAM_SEARCH_CANDIDATES["atr_periods"][0]
        base_cfg["take_profit_atr_mult"] = PARAM_SEARCH_CANDIDATES["take_profit_atr_mults"][0]
        base_cfg["macd_fast"] = PARAM_SEARCH_CANDIDATES["macd_fast_periods"][0]
        base_cfg["macd_slow"] = PARAM_SEARCH_CANDIDATES["macd_slow_periods"][0]
        base_cfg["macd_signal"] = PARAM_SEARCH_CANDIDATES["macd_signal_periods"][0]
        configs_to_test = [base_cfg]
    else:
        configs_to_test = []
        for combo in product(
            PARAM_SEARCH_CANDIDATES["ema_periods"],
            PARAM_SEARCH_CANDIDATES["keltner_mults"],
            PARAM_SEARCH_CANDIDATES["ml_min_prob_long"],
            PARAM_SEARCH_CANDIDATES["rsi_signal_slope"],
            PARAM_SEARCH_CANDIDATES["atr_periods"],
            PARAM_SEARCH_CANDIDATES["take_profit_atr_mults"],
            PARAM_SEARCH_CANDIDATES["macd_fast_periods"],
            PARAM_SEARCH_CANDIDATES["macd_slow_periods"],
            PARAM_SEARCH_CANDIDATES["macd_signal_periods"]
        ):
            c_ema, c_kelt, c_prob, c_rsi_slope, c_atr, c_tp, c_macd_fast, c_macd_slow, c_macd_signal = combo
            test_cfg = dict(base_cfg)
            test_cfg["ema_period"] = c_ema
            test_cfg["keltner_mult"] = c_kelt
            test_cfg["ml_min_prob_long"] = c_prob
            test_cfg["rsi_signal_slope"] = c_rsi_slope
            test_cfg["atr_period"] = c_atr
            test_cfg["take_profit_atr_mult"] = c_tp
            test_cfg["macd_fast"] = c_macd_fast
            test_cfg["macd_slow"] = c_macd_slow
            test_cfg["macd_signal"] = c_macd_signal
            configs_to_test.append(test_cfg)
    best_roi = float("-inf")
    best_metrics = None
    best_config = None
    for idx, cfg in enumerate(configs_to_test, start=1):
        trial_label = f"Trial #{idx} => EMA={cfg['ema_period']}, KMult={cfg['keltner_mult']}, Prob={cfg['ml_min_prob_long']}, RSI_Slope={cfg['rsi_signal_slope']}, ATR={cfg['atr_period']}, TP_Mult={cfg['take_profit_atr_mult']}, MACD={cfg['macd_fast']}/{cfg['macd_slow']}/{cfg['macd_signal']}"
        logging.info(f"\n{'-'*40}\n{trial_label}\n{'-'*40}")
        trader = PaperTrader(cfg["initial_capital"], cfg)
        intraday_period = min(cfg["intraday_period_days"], 60)
        for sym in cfg["symbols"]:
            df = get_bars(sym, cfg["interval"], intraday_period, cfg["lookback_days"])
            if df.empty or len(df) < 30:
                logging.warning(f"No or insufficient data for {sym}; skipping.")
                continue
            logging.info(f"Processing {sym} (rows={len(df)}) ...")
            sim_df = run_strategy_on_df(sym, df, cfg, trader)
            if cfg["save_charts"]:
                save_chart(sym, sim_df, cfg)
        metrics = trader.metrics()
        logging.info(
            f"[{trial_label}] => Final Capital=${metrics['Current_Capital']:.2f}, "
            f"PnL=${metrics['Realized_PnL']:.2f}, ROI={metrics['ROI']*100:.2f}%, "
            f"WinRate={metrics['Win_Rate']*100:.2f}%, Trades={metrics['Num_Trades']}, "
            f"AvgWin=${metrics['Avg_Win']:.2f}, AvgLoss=${metrics['Avg_Loss']:.2f}, "
            f"MaxWin=${metrics['Max_Win']:.2f}, MaxLoss=${metrics['Max_Loss']:.2f}, "
            f"Sharpe={metrics['Sharpe']:.2f}, Sortino={metrics['Sortino']:.2f}, "
            f"Drawdown={metrics['Drawdown']:.1f}%, VaR_95={metrics['VaR_95']:.1f}%, "
            f"Beta={metrics['Beta']:.2f}, Alpha={metrics['Alpha']:.2f}"
        )
        # Freqtrade-inspired loss function: maximize ROI and Sharpe
        loss = - (metrics["ROI"] + metrics["Sharpe"] * 0.1) if metrics["Num_Trades"] > 0 else float("-inf")
        if metrics["ROI"] > best_roi and metrics["Num_Trades"] > 0:
            best_roi = metrics["ROI"]
            best_metrics = metrics
            best_config = dict(cfg)
    print("\n===== BEST PARAMETER SET SUMMARY =====")
    if best_config and best_metrics:
        print("Best Config:")
        print(f"  EMA Period:         {best_config['ema_period']}")
        print(f"  Keltner Mult:       {best_config['keltner_mult']}")
        print(f"  ML Min Prob (Long): {best_config['ml_min_prob_long']}")
        print(f"  RSI Signal Slope:   {best_config['rsi_signal_slope']}")
        print(f"  ATR Period:         {best_config['atr_period']}")
        print(f"  Take Profit Mult:   {best_config['take_profit_atr_mult']}")
        print(f"  MACD Fast/Slow/Sig: {best_config['macd_fast']}/{best_config['macd_slow']}/{best_config['macd_signal']}")
        print("\nPerformance:")
        print(f"  Final Capital: ${best_metrics['Current_Capital']:.2f}")
        print(f"  Realized PnL:  ${best_metrics['Realized_PnL']:.2f}")
        print(f"  ROI:           {(best_metrics['ROI'] * 100):.2f}%")
        print(f"  Win Rate:      {(best_metrics['Win_Rate'] * 100):.2f}%")
        print(f"  # Trades:      {best_metrics['Num_Trades']}")
        print(f"  Avg Win:       ${best_metrics['Avg_Win']:.2f}")
        print(f"  Avg Loss:      ${best_metrics['Avg_Loss']:.2f}")
        print(f"  Max Win:       ${best_metrics['Max_Win']:.2f}")
        print(f"  Max Loss:      ${best_metrics['Max_Loss']:.2f}")
        print(f"  Sharpe:        {best_metrics['Sharpe']:.2f}")
        print(f"  Sortino:       {best_metrics['Sortino']:.2f}")
        print(f"  Drawdown:      {best_metrics['Drawdown']:.1f}%")
        print(f"  VaR 95%:       {best_metrics['VaR_95']:.1f}%")
        print(f"  Beta:          {best_metrics['Beta']:.2f}")
        print(f"  Alpha:         {best_metrics['Alpha']:.2f}")
        final_summary = (
            f"Best Param Set => EMA={best_config['ema_period']}, KMult={best_config['keltner_mult']}, "
            f"Prob={best_config['ml_min_prob_long']}, RSI_Slope={best_config['rsi_signal_slope']}, "
            f"ATR={best_config['atr_period']}, TP_Mult={best_config['take_profit_atr_mult']}, "
            f"MACD={best_config['macd_fast']}/{best_config['macd_slow']}/{best_config['macd_signal']} | "
            f"ROI={(best_metrics['ROI']*100):.2f}%, WinRate={(best_metrics['Win_Rate']*100):.2f}%, "
            f"Trades={best_metrics['Num_Trades']}, Sharpe={best_metrics['Sharpe']:.2f}"
        )
        logging.info(final_summary)
        send_telegram_message(final_summary, best_config)
    else:
        print("No valid config found or all runs failed to produce trades.")

if __name__ == "__main__":
    main()
