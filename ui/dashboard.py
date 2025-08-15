# ui/dashboard.py
import sys
import os
import pandas as pd
import streamlit as st

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.data import synthetic_ohlcv, fetch_ohlcv_ccxt
from src.strategies.scalping import enrich, signals
from src.config import cfg

st.set_page_config(page_title="TradingAI Bot — Demo Dashboard", layout="wide", page_icon="🤖")

# Add a small CSS to tidy up
st.markdown(
    """
    <style>
    .reportview-container { background-color: #0f1115; color: #e6eef6; }
    .sidebar .sidebar-content { background-color: #2d2f33; color: #fff; }
    .stButton>button { background-color: #ff6b6b; color: #fff; }
    .big-title { font-size:48px; font-weight:700; color:#ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# UI controls
st.sidebar.header("Inputs")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
mode = st.sidebar.selectbox("Mode", ["demo", "ccxt"])
bars = st.sidebar.slider("Bars", min_value=200, max_value=3000, value=1200, step=100)
run_btn = st.sidebar.button("Refresh")

st.title("TradingAI Bot — Demo Dashboard")

# Data fetch with safe fallback
try:
    if mode == "demo":
        df = synthetic_ohlcv(symbol, limit=bars)
    else:
        df = fetch_ohlcv_ccxt(symbol, limit=bars)
    # ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

# Enrich and compute signals (explicitly pass cfg)
try:
    enriched = enrich(df, cfg=cfg)
    sig = signals(enriched, cfg=cfg)
except Exception as e:
    st.error(f"Failed to compute indicators/signals: {e}")
    st.stop()

# Top metrics: latest price, ATR%, last signal
col1, col2, col3, col4 = st.columns([2,1,1,1])
col1.metric("Latest Price", f"${enriched['close'].iloc[-1]:.2f}")
col2.metric("ATR (pct)", f"{enriched['ATR_PCT'].iloc[-1]*100:.3f}%")
col3.metric("KC Width", f"{((enriched['KC_UP']-enriched['KC_LOW']).iloc[-1]/enriched['close'].iloc[-1])*100:.3f}%")
col4.metric("Signal (now)", "LONG" if sig.iloc[-1] else "—")

# Price chart with Keltner bands
st.subheader("Market Data")
plot_df = enriched[["close", "KC_UP", "KC_LOW"]].dropna()
st.line_chart(plot_df)

# Signals table: last N rows
st.subheader("Signals (most recent)")
signal_df = sig.astype(int).to_frame("long").assign(time=sig.index)
signal_df = signal_df.set_index("time")
st.dataframe(signal_df.tail(50), height=340)

# Download CSV
st.markdown("**Download enriched data**")
csv = enriched.to_csv().encode("utf-8")
st.download_button("Download CSV", csv, f"{symbol}_enriched.csv", "text/csv")

# Optional: lightweight backtest button (toy)
if st.button("Run quick demo backtest"):
    st.info("Running quick demo (in-memory, toy backtest)...")
    # Simple toy backtest: buy at next open if signal True
    capital = 10000.0
    positions = 0
    log = []
    for i in range(len(enriched)-2):
        if sig.iloc[i] and positions == 0:
            # buy next open
            price = float(enriched.iloc[i+1]["open"])
            qty = int(capital // price)
            if qty > 0:
                positions = qty
                capital -= qty * price
                log.append(f"BUY {qty}@{price:.2f}")
        # simple close at next bar if in position and price > entry (toy)
        if positions > 0:
            cur_price = float(enriched.iloc[i]["close"])
            # implement simple stop at ATR distance
            stop = float(enriched.iloc[i]["close"]) - float(enriched.iloc[i]["ATR"])
            low = float(enriched.iloc[i]["low"])
            if low <= stop:
                # close at stop
                capital += positions * stop
                log.append(f"STOP_EXIT {positions}@{stop:.2f}")
                positions = 0
    st.write("Backtest log (toy):")
    st.write("\n".join(log[:20]) or "No trades")

st.caption("Demo dashboard: indicators are synthetic/informational only.")
