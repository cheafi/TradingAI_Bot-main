import streamlit as st
from src.ui_ext.ui_helpers import render_mode_pill
import pandas as pd

st.set_page_config(page_title="Research Lab", layout="wide")
st.title("Research Lab")
render_mode_pill()
st.caption("Experimental (does not affect alerts)")

st.subheader("Parameter Sweep")
results = pd.DataFrame(
    [
        {"TF": "1h", "EMA": 20, "RSI": 14, "Avg +30d": 0.018},
        {"TF": "4h", "EMA": 50, "RSI": 14, "Avg +30d": 0.022},
    ]
)
st.dataframe(results, use_container_width=True)

best = results.sort_values("Avg +30d", ascending=False).iloc[0]
st.metric("Best Avg +30d", f"{best['Avg +30d']*100:.2f}%", help="From the sweep above")  # noqa: E501
