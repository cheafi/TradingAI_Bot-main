import streamlit as st
import pandas as pd
from src.ui_ext.ui_helpers import render_mode_pill
from src.ui_ext.data_sources import make_pdf_from_text
from src.ui_ext.settings_store import send_telegram_message

st.set_page_config(page_title="Portfolios", layout="wide")
st.title("Portfolios")
render_mode_pill()

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total P&L", "+$1,240")
with col2:
    st.metric("Exposure", "48%")
with col3:
    st.metric("Max Drawdown", "-6.8%")
with col4:
    st.metric("Win Rate", "57%")

st.subheader("Positions")
df = pd.DataFrame(
    [
        {"Symbol": "BTC", "Qty": 0.25, "Entry": 62000, "Stop": 60100, "Target": 65000, "P&L": 320, "Risk%": 0.7},  # noqa: E501
        {"Symbol": "ETH", "Qty": 1.5, "Entry": 3100, "Stop": 2990, "Target": 3300, "P&L": -45, "Risk%": 0.9},  # noqa: E501
    ]
)
st.dataframe(df, use_container_width=True)

st.divider()
left, mid, right = st.columns([1, 1, 1])
with left:
    st.button("Rebalance")
with mid:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", csv_bytes, "portfolio.csv", "text/csv")
    if st.button("Export PDF"):
        lines = [
            "Portfolio Snapshot",
        ] + [
            (
                f"{r.Symbol} qty {r.Qty} @ {r.Entry} | P&L "
                f"{r._asdict().get('P&L', 0)}"
            )
            for r in df.itertuples()
        ]
        st.session_state["pf_pdf"] = make_pdf_from_text("Portfolio", lines)
    if st.session_state.get("pf_pdf"):
        st.download_button(
            "Download PDF",
            st.session_state["pf_pdf"] or b"",
            "portfolio.pdf",
            "application/pdf",
            disabled=st.session_state["pf_pdf"] is None,
        )
with right:
    if st.button("Notify Telegram"):
        ok = send_telegram_message("Portfolio snapshot available in UI")
        st.success("Sent") if ok else st.warning("Not sent (check Telegram)")

st.warning("ETH is within 0.3% of stop.")
act1, act2 = st.columns([1, 1])
with act1:
    st.button("Raise Stop")
with act2:
    st.button("Close 50%")
