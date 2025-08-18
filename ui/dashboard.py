# ui/dashboard.py
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from ui.pages.dashboard_main import dashboard_page

st.set_page_config(page_title="TradingAI Bot", layout="wide", page_icon="🤖")

# Route to main page
dashboard_page()
