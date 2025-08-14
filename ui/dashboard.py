# ui/dashboard.py
import logging
import streamlit as st
from src.main import RunConfig, run_once
from src.utils.ui import sidebar_config, metrics_panel

st.set_page_config(page_title="TradingAI Bot", layout="wide")
st.title("TradingAI Bot — Hybrid ML Momentum")

cfg_ui = sidebar_config()
if st.button("Run Backtest"):
    try:
        metrics = run_once(RunConfig(symbol=cfg_ui["symbol"], timeframe=cfg_ui["timeframe"]))
        metrics_panel(metrics)
    except Exception as e:
        logging.exception(e)
        st.error(f"Run failed: {e}")
