import streamlit as st
from src.ui_ext.ui_helpers import render_mode_pill

st.set_page_config(page_title="Advisor", layout="wide")
st.title("Advisor")
render_mode_pill()

st.subheader("Guided Questionnaire")
col1, col2, col3 = st.columns(3)
with col1:
    horizon = st.selectbox("Time Horizon", ["Days", "Weeks", "Months"])  # noqa: E501
with col2:
    max_dd = st.slider("Max Drawdown You Can Tolerate", 5, 50, 15, step=5)
with col3:
    focus = st.selectbox("Focus Universe", ["BTC Majors", "Long Tail", "Mixed"])  # noqa: E501

st.subheader("Recommendation")
stance = "Balanced Growth"
st.write(f"Suggested stance: {stance}")

bullets = [
    "Target exposure: 40–60%",  # noqa: E501
    "Max per-trade risk: 1% notional (dry-run)",
    "Top candidates: BTC, ETH, SOL",
    "Stops near ATR×2 below structure; trail on +1R",
]
st.write("\n".join(f"- {item}" for item in bullets))

st.divider()
actions = st.columns([1, 1, 6])
with actions[0]:
    st.button("Add To Watchlist")
with actions[1]:
    st.button("Send Summary To Telegram")

st.caption("Microcopy: Clear, confident, brief. ‘Do X because Y’.")
