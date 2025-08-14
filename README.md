# TradingAI_Bot-main — Elite Quant Crypto Bot (Modular, Antifragile, Fast)

**Goal:** 30–45% annualized ROI, **Sharpe > 2.0**, **Max DD < 7%** via hybrid ML + momentum scalping, robust risk control (Kelly, VaR(95), Monte Carlo), and antifragile engineering.

## ✨ Features
- **Hybrid ML–Momentum Strategy**: Walk-forward ML (RF/LSTM) + Keltner/RSI/ATR momentum.
- **Risk Engine**: Kelly sizing (capped), VaR(95), Monte Carlo stress (<7% DD target).
- **Exchanges**: `ccxt` (Binance, Coinbase, Bybit, etc.), **paper/live** modes.
- **UI**: **Streamlit dashboard** (PnL, Sharpe, VaR, live positions), strategy picker, parameter sliders.
- **Telegram**: Alerts + commands (`/balance`, `/profits`, `/reset`, interactive buttons).
- **Modular**: `src/strategies`, `src/utils`, `ui/`, `tests/` — inspired by Jesse/Nautilus.
- **Backtesting**: Vectorized NumPy/Pandas; Sharpe = `(mean_ret - rf)/std_ret * sqrt(252)`.
- **Antifragile**: Try/except everywhere, graceful fallbacks, structured logs.

## 🧭 UI/UX Overview (Streamlit)
- **Top Bar**: Account mode (paper/live), exchange, symbol.
- **Left Sidebar**: Strategy selector (Scalping ML / Market Making), risk sliders.
- **Main Panels**:
  1. Equity & Drawdown
  2. Open Positions & Risk (ATR stops)
  3. Trades + Reasons
  4. Metrics: ROI, Sharpe, VaR(95), Win rate
  5. Strategy params + “Deploy” button

## 🧩 Strategies
- **Primary:** Hybrid ML-Momentum (Keltner breakout + RSI slope + RF/LSTM prob filter; ATR trailing stops, TP at ATR multiples)
- **Alternative:** Market-making / arbitrage (narrow spreads, inventory skew)

## ⚙️ Setup
```bash
# Dev
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Streamlit UI
streamlit run ui/dashboard.py

# Telegram bot (set TELEGRAM_TOKEN/CHAT_ID in .env)
python src/utils/telegram_bot.py

# Docker
docker compose up --build