# TradingAI_Bot-main

**What is this?**  
A modular, beginner-friendly AI trading bot skeleton for crypto & stocks. Runs safe demos (no keys), paper trading, and can connect to real exchanges (Binance / Futu).

**Quick start (Day 1):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python basic.py           # safe offline demo
streamlit run ui/dashboard.py
pytest -q

**Simple explanations**
ROI = how much your piggy bank grows in a year (we aim 35–50%).
Kelly = smart bet sizing so you don’t lose your shirt.
VaR(95) = how badly you might lose on a very bad day (95% worst-case).
Monte Carlo = rolling dice many times to see how bad things could get.
Project layout
See src/ (the bot brain), ui/ (dashboard), tests/ (automated checks), research/agents/ (ML experiments).
Disclaimer: Educational only—backtest 10+ years (yfinance/ccxt), consult advisors, and comply with SEC Reg NMS.