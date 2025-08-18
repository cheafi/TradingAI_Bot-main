# ui/pages/dashboard_main.py
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st
from ui.i18n import ZH_TW as T
from src.utils.data import synthetic_ohlcv, fetch_ohlcv_ccxt
from src.strategies.scalping import enrich, signals
from src.utils.risk import sharpe, max_drawdown, var_gaussian
from src.config import cfg

def _metrics_block(enriched: pd.DataFrame, sig: pd.Series):
    col1, col2, col3, col4 = st.columns([2,1,1,1])
    col1.metric(T.price, f"${enriched['close'].iloc[-1]:.2f}")
    col2.metric(T.atr_pct, f"{enriched['ATR_PCT'].iloc[-1]*100:.3f}%")
    col3.metric(T.keltner_width, f"{((enriched['KC_UP']-enriched['KC_LOW']).iloc[-1]/enriched['close'].iloc[-1])*100:.3f}%")
    col4.metric(T.signal_now, "Strong Buy/Buy (Long)" if bool(sig.iloc[-1]) else "Hold/Sell/Short")

def _risk_block(enriched: pd.DataFrame):
    ret = enriched["close"].pct_change().dropna()
    st.subheader(T.risk_metrics)
    c1, c2, c3 = st.columns(3)
    c1.metric(T.sharpe, f"{sharpe(ret):.2f}")
    c2.metric(T.mdd, f"{max_drawdown((1+ret).cumprod()):.2%}")
    c3.metric(T.var95, f"{var_gaussian(ret, 0.95):.2%}")

def dashboard_page():
    st.markdown("<h1 style='font-weight:800;'>📈 " + T.title + "</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.header(T.inputs)
        symbol = st.text_input(T.symbol, "BTC/USDT")
        mode = st.selectbox(T.mode, ["demo", "ccxt"])
        bars = st.slider(T.bars, 200, 3000, 1200, 100)
        if st.button(T.refresh):
            st.experimental_rerun()

    # Data
    try:
        if mode == "demo":
            df = synthetic_ohlcv(symbol, limit=bars)
        else:
            df = fetch_ohlcv_ccxt(symbol, limit=bars)
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        st.error(f"Data Error: {e}")
        return

    # Indicators + signals
    try:
        enriched = enrich(df, cfg=cfg)
        sig = signals(enriched, cfg=cfg)
    except Exception as e:
        st.error(f"Indicator Error: {e}")
        return

    # Metrics
    st.subheader(T.metrics)
    _metrics_block(enriched, sig)

    # Chart
    st.subheader(T.market_data)
    st.line_chart(enriched[["close","KC_UP","KC_LOW"]].dropna())

    # Signals
    st.subheader(T.signals)
    st.dataframe(sig.astype(int).to_frame("Buy/Long").tail(100), height=320)

    # Risk
    _risk_block(enriched)

    # PnL (toy)
    st.subheader(T.pnl)
    eq = (1 + enriched["close"].pct_change().fillna(0)).cumprod()
    st.line_chart(eq)

    # Download
    st.download_button(T.download_enriched, enriched.to_csv().encode("utf-8"), f"{symbol}_enriched.csv", "text/csv")
