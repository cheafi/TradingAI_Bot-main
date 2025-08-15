# ui/dashboard.py
import streamlit as st
from src.utils.data import synthetic_ohlcv, fetch_ohlcv_ccxt
from src.strategies.scalping import enrich, signals

st.set_page_config(page_title="TradingAI Dashboard", layout="wide")
st.title("TradingAI Bot — Demo Dashboard")

symbol = st.sidebar.text_input("Symbol", value="BTC/USDT")
mode = st.sidebar.selectbox("Mode", ["demo", "ccxt (live if keys)"])
limit = st.sidebar.slider("Bars", 500, 3000, 1200, 100)

st.header("Market Data")
if mode == "demo":
    df = synthetic_ohlcv(symbol, limit)
else:
    df = fetch_ohlcv_ccxt(symbol, limit=limit)

st.line_chart(df["close"].tail(500))

st.header("Signals")
cfg_text = st.sidebar.text("Using built-in config.")
df2 = enrich(df, 20, 14, 1.5)
sig = signals(df, None)  # quick: replace with cfg import in future
st.write("Recent signals (True = Long):")
st.dataframe(sig.tail(20))
