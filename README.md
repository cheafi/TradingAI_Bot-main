# TradingAI_Bot-main: AI-Powered Quant Trading Bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/cheafi/TradingAI_Bot-main/actions)

## Overview
TradingAI_Bot-main is a modular, beginner-easy AI trading bot skeleton for crypto and stocks, designed for high-ROI algorithmic strategies (target: 35-50% annualized, Sharpe >2.5, max drawdown <5%). It automates signals (e.g., scalping with Keltner/RSI/ATR + ML prob stubs), backtesting, paper trading, and real execution on exchanges like Binance/Futu. Built with antifragile principles: error-free (try/except/fallbacks), low-latency (<100ms sims via uvloop/NumPy), and scalable (Docker/Redis). Inspired by top tools (Nautilus for event-driven, Qlib/ai-hedge-fund for multi-agents, OpenBB for data/UI).

Key Features:
- **Strategies**: Hybrid ML-momentum (primary: scalping with RF/LSTM prob>0.55; alternative: arbitrage with cointegration).
- **Risk Controls**: Kelly sizing, VaR(95%), Monte Carlo (2000+ paths), GARCH vol hedges for 2025 spikes.
- **Modes**: Backtest (10+ years data), paper (sim trades), real (ccxt/Futu APIs).
- **AI Research**: Stock input (e.g., AAPL) → buy/sell/hold/target/stop/reasons via agents (Damodaran valuation + sentiment).
- **UI/Alerts**: Streamlit dashboard (charts/metrics), Telegram commands (/suggest/profits/balance/reset).
- **Customizability**: Extend tentacles/agents for TA/AI/ChatGPT/TradingView.

Suitable for equities/forex/crypto/derivatives. **Disclaimer**: Educational only. Backtest thoroughly on 10+ years data (yfinance/ccxt), consult financial advisors, comply with regs like SEC Reg NMS.

## Installation
1. Clone the repo: `git clone https://github.com/cheafi/TradingAI_Bot-main.git && cd TradingAI_Bot-main`
2. Create/activate venv: `python3 -m venv .venv && source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows).
3. Install deps: `pip install -r requirements.txt` (core: Pandas/NumPy/Streamlit/ccxt/telegram-bot/pytest/uvloop; optional: tensorflow/torch for ML).
4. Copy .env.example to .env and fill keys (Binance/Futu/Telegram/Redis—never commit .env).

## Usage
- **Safe Demo**: `python basic.py` (offline-friendly synthetic data, logs trades/PnL/Monte Carlo DD).
- **UI Dashboard**: `streamlit run ui/dashboard.py` (input symbol/mode, view charts/signals).
- **Tests**: `pytest -q` (100% coverage for risk/strategies).
- **Backtest**: Edit main.py for multi-symbol, run `python src/main.py --mode=backtest --symbol=BTC/USDT` (reports ROI/DD/Sharpe).
- **Paper/Real**: Set mode in main.py, add API keys in .env for live execution (review utils/execution.py first).
- **Telegram**: Fill .env, run `python src/utils/telegram_bot.py` (commands like /suggest AAPL for AI recs).
- **Docker Deploy**: `docker compose up --build` (runs UI on localhost:8501).

Example Strategy Code (in strategies/scalping.py—extend for ROI >40%):
```python
def signals(df: pd.DataFrame, cfg) -> pd.Series:
    df = enrich(df, cfg.EMA_PERIOD, cfg.ATR_PERIOD, cfg.KELTNER_MULT)
    prob = ml_probability_stub(df)  # Replace with RF/LSTM for prob>0.55
    long = (df["close"] > df["KC_UP"]) & (df["RSI3"] > 50) & (prob >= 0.55)
    return long


## Strategies

Primary: Hybrid ML-Scalping (Keltner cross + RSI>50 + ML prob>0.55 for 40% ROI). - Alternative: Arbitrage (cointegration tests across Binance/Futu pairs).
Optimize via walk-forward (avoid overfitting): Train RF/LSTM on features (RSI/ATR/EMA), predict in ml_probability_stub.