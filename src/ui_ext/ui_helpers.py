from __future__ import annotations
import streamlit as st
from .settings_store import load_settings


def render_mode_pill() -> None:
    """Render a small DRY-RUN/LIVE pill below page title."""
    s = load_settings()
    dry = bool(s.get("DRY_RUN", True))
    label = "DRY-RUN" if dry else "LIVE"
    if dry:
        bg = "#eef2f7"
        fg = "#334155"
    else:
        bg = "#e6f4ea"
        fg = "#137333"
    html = (
        f"<span style='background:{bg};color:{fg};padding:2px 8px;"
        f"border-radius:12px;font-size:0.85rem;'>Mode: {label}</span>"
    )
    st.markdown(html, unsafe_allow_html=True)
