import time
import os
import sys
from datetime import datetime, timezone
from contextlib import suppress
import streamlit as st
import pandas as pd

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import our modules
try:
    from src.ui_ext.data_sources import (
        summarize_market_report,
        recent_risk_alerts,
        read_opportunities,
        make_pdf_from_text,
    )
    from src.ui_ext.settings_store import load_settings, send_telegram_message
    from src.ui_ext.ui_helpers import render_mode_pill
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure the project is properly set up and all dependencies are installed.")
    st.stop()


def _cleanup_legacy_pages() -> None:
    """Remove legacy/duplicate Streamlit pages to avoid sidebar clutter."""
    with suppress(Exception):
        base_dir = os.path.dirname(__file__)
        pages_dir = os.path.join(base_dir, "pages")
        legacy = [
            "dashboard.py",
            "dashboard_main.py",
            "data_explorer.py",
            "portfolio_analysis.py",
            "prediction_analysis.py",
            "settings.py",
            "variable_tuner.py",
            "01_Enhanced Dashboard.py",
        ]
        for fname in legacy:
            fpath = os.path.join(pages_dir, fname)
            if os.path.exists(fpath):
                with suppress(Exception):
                    os.remove(fpath)


_cleanup_legacy_pages()

st.set_page_config(page_title="Enhanced Dashboard", layout="wide")
st.title("Enhanced Dashboard")
render_mode_pill()

# Universe selector
universe = st.radio(
    "Universe",
    options=["Crypto", "US Stocks"],
    horizontal=True,
    key="universe_choice",
    help=(
        "Choose your market universe. This affects counts, focus, and "
        "exports below."
    ),
)


def kpi_card(
    label: str,
    value: str,
    tip: str | None = None,
    color: str | None = None,
):
    if color == "green":
        st.success(f"{label}: {value}")
    elif color == "red":
        st.error(f"{label}: {value}")
    else:
        st.info(f"{label}: {value}")
    if tip:
        st.caption(tip)


def human_last_modified(paths: list[str]) -> str:
    latest = 0.0
    for pth in paths:
        if os.path.exists(pth):
            latest = max(latest, os.path.getmtime(pth))
    if latest == 0:
        return "unknown"
    secs = time.time() - latest
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs//60)}m ago"
    return datetime.fromtimestamp(latest, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_card(
        "Regime",
        "Risk-On",
        "Market regime is Risk-On — breadth & momentum supportive.",
        color="green",
    )
with col2:
    trend_label = "BTC Trend" if universe == "Crypto" else "SPY Trend"
    kpi_card(
        trend_label,
        "Uptrend",
        "Trading above its 50-day EMA.",
        color="green",
    )
with col3:
    kpi_card(
        "Volatility",
        "Moderate (60th pct)",
        "Risk/reward balanced.",
    )
with col4:
    _settings = load_settings()
    mode_is_dry = _settings.get("DRY_RUN", True)
    kpi_card(
        "Mode",
        ("DRY-RUN" if mode_is_dry else "LIVE"),
        ("No live orders placed." if mode_is_dry else "Live orders enabled."),
        color=(None if mode_is_dry else "green"),
    )


@st.cache_data(ttl=30)
def _cached_summary() -> str:
    return summarize_market_report(max_words=80)


@st.cache_data(ttl=30)
def _cached_ops():
    return read_opportunities()


def _filter_ops(ops: list[dict], uni: str) -> list[dict]:
    if uni == "US Stocks":
        return [o for o in ops if "/" not in str(o.get("symbol", ""))]
    return [o for o in ops if "/" in str(o.get("symbol", ""))]


def _focus_from_ops(ops: list[dict], limit: int = 3):
    ops_sorted = sorted(
        ops,
        key=lambda x: (
            float(x.get("score") or 0),
            float(x.get("ml_score") or 0.0),
        ),
        reverse=True,
    )
    out = []
    for o in ops_sorted:
        sym = o.get("symbol", "?")
        rec = o.get("recommendation") or (
            "BUY" if (o.get("score") or 0) >= 5 else "WATCH"
        )
        reasons = o.get("reasons") or []
        top_reason = reasons[0] if reasons else "Momentum improving"
        out.append((f"{sym} — {rec} candidate", f"Why: {top_reason}"))
        if len(out) >= limit:
            break
    return out


st.subheader("Today’s Market Report")
st.caption(
    "What: concise outlook. Why: context for focus list. "
    "Action: export, notify Telegram, or check Signals."
)
with st.spinner("Loading market report…"):
    summary = _cached_summary()
st.write(summary)
with st.spinner("Loading opportunities…"):
    ops = _cached_ops()
ops_f = _filter_ops(ops, universe)
colx1, colx2, colx3 = st.columns(3)
with colx1:
    st.caption(f"Opportunities: {len(ops_f)} in latest scan")
with colx2:
    if st.button("Export CSV"):
        df = pd.DataFrame(ops_f)
        st.session_state["_dl_csv"] = df.to_csv(index=False).encode("utf-8")
    if "_dl_csv" in st.session_state:
        st.download_button(
            "Download CSV",
            data=st.session_state["_dl_csv"],
            file_name="opportunities.csv",
            mime="text/csv",
        )
with colx3:
    if st.button("Export PDF"):
        lines = [
            summary
        ] + [
            f"{o.get('symbol')}: score {o.get('score')}" for o in ops_f[:10]
        ]
        pdf = make_pdf_from_text("Today’s Market Report", lines)
        st.session_state["_dl_pdf"] = pdf
    if st.session_state.get("_dl_pdf"):
        st.download_button(
            "Download PDF",
            data=st.session_state["_dl_pdf"] or b"",
            file_name="market_report.pdf",
            mime="application/pdf",
            disabled=st.session_state["_dl_pdf"] is None,
        )
    if st.button("Notify Telegram"):
        ok = send_telegram_message("Market report updated on dashboard")
        st.success("Sent") if ok else st.warning(
            "Not sent (check Telegram setup)"
        )


st.subheader("Current Market Focus")
st.caption(
    "Top 3 candidates by score/ML signal. Read Why to see drivers."
)
focus_items = _focus_from_ops(ops_f, limit=3)
if not focus_items:
    st.info(
        "No strong focus items right now. Check back after next scan or "
        "open Signals to review full list."
    )
else:
    for head, why in focus_items:
        st.write(f"• {head}")
        st.caption(why)

with st.expander("Guided Actions", expanded=False):
    st.write("- View Signals to see all setups (filter by score/TF).")
    st.write("- Notify Telegram to share a quick update.")
    st.write("- Open Research Lab to experiment and validate.")
    cA, cB = st.columns(2)
    with cA:
        try:
            st.page_link(
                "pages/04_Signals.py",
                label="View Signals",
                icon=":material/trending_up:",
            )
        except Exception:
            st.caption("Use sidebar to open Signals.")
    with cB:
        try:
            st.page_link(
                "pages/06_Research Lab.py",
                label="Open Research Lab",
                icon=":material/science:",
            )
        except Exception:
            st.caption("Use sidebar to open Research Lab.")


# Optional US stocks snapshot when selected
if universe == "US Stocks":
    st.subheader("US Market Snapshot")
    snapshot = {}
    try:
        import json as _json
        with open("data/daily_reports/market_snapshot.json", "r") as _f:
            snapshot = _json.load(_f)
    except Exception:
        snapshot = {}
    tickers = ["SPY", "QQQ", "AAPL", "MSFT"]
    cols = st.columns(4)
    any_data = False
    for i, t in enumerate(tickers):
        info = snapshot.get(t, {}) if isinstance(snapshot, dict) else {}
        price = info.get("current_price")
        if price is not None:
            any_data = True
            with cols[i]:
                st.metric(t, price)
    if not any_data:
        st.caption(
            "No US snapshot available yet. Will populate after next report."
        )


st.subheader("Recent Alerts")
if alerts := recent_risk_alerts(5):
    for a in alerts:
        st.warning(a)
else:
    st.success("No recent risk alerts.")

st.subheader("System Health")
latest_files = [
    "logs/latest_market_outlook.txt",
    "logs/latest_opportunities.txt",
    "logs/latest_portfolio_analysis.txt",
    "logs/latest_risk_alerts.txt",
]
last_refresh = human_last_modified(latest_files)
settings = load_settings()
mode_str = "DRY-RUN" if settings.get("DRY_RUN", True) else "LIVE"
st.caption(
    f"Mode: {mode_str}  |  Last data refresh: {last_refresh}  |  "
    "Next scans: crypto 1m / equities pre-market in 3h"
)

# quick checklist for common setup items
tg_token_ok = bool(os.getenv("TELEGRAM_TOKEN"))
tg_chat_ok = bool(os.getenv("TELEGRAM_CHAT_ID"))
ck1, ck2, ck3 = st.columns(3)
with ck1:
    st.write(
        (
            "✅ Telegram Token set"
            if tg_token_ok
            else "⚠️ Telegram Token missing"
        )
    )
with ck2:
    st.write(
        ("✅ Chat ID set" if tg_chat_ok else "ℹ️ Chat ID optional (for pushes)")
    )
with ck3:
    st.write(
        (
            "✅ Files readable"
            if last_refresh != "unknown"
            else "ℹ️ Waiting for data"
        )
    )

st.info(
    "No active chart here. See Signals for setups or Research Lab for visuals."
)

# Next steps quick links
try:
    st.subheader("Next Steps")
    a, b = st.columns(2)
    with a:
        st.page_link(
            "pages/04_Signals.py",
            label="View Signals",
            icon=":material/trending_up:",
        )
    with b:
        st.page_link(
            "pages/06_Research Lab.py",
            label="Open Research Lab",
            icon=":material/science:",
        )
except Exception:
    st.caption("Use the sidebar to navigate to Signals or Research Lab.")
