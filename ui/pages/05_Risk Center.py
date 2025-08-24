import streamlit as st
from src.ui_ext.ui_helpers import render_mode_pill

st.set_page_config(page_title="Risk Center", layout="wide")
st.title("Risk Center")
render_mode_pill()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("VaR (95%)", "-2.1%", delta=None)
with col2:
    st.metric("Max DD 30d", "-7.4%")
with col3:
    st.metric("Max DD 90d", "-12.8%")

st.subheader("What Could Go Wrong")
st.write("- Liquidity dries up on weekend; gaps widen.\n"
         "- BTC loses 50-day EMA; momentum flips.")

st.subheader("Correlation Heatmap")
st.info("No data yet. Upload or run a scan to populate.")
