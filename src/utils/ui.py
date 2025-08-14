from __future__ import annotations
import os
import streamlit as st
import pandas as pd

def sidebar_config():
    st.sidebar.title("Config")
    symbol = st.sidebar.text_input("Symbol", value="BTC/USDT")
    timeframe = st.sidebar.selectbox("Timeframe", options=["15m","1h","4h"], index=0)
    return {"symbol": symbol, "timeframe": timeframe}

def metrics_panel(metrics: dict):
    st.subheader("Performance")
    cols = st.columns(5)
    cols[0].metric("Final Capital", f"${metrics['final_capital']:.2f}")
    cols[1].metric("ROI", f"{metrics['roi']*100:.2f}%")
    cols[2].metric("Sharpe", f"{metrics['sharpe']:.2f}")
    cols[3].metric("Max DD", f"{metrics['max_dd']*100:.2f}%")
    cols[4].metric("VaR(95)", f"{metrics['var95']*100:.2f}%")
    st.caption(f"MC DD median: {metrics['mc_dd_med']*100:.2f}% | MC DD 95%: {metrics['mc_dd_95']*100:.2f}%")
