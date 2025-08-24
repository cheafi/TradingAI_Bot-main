import os
import time
from datetime import datetime, timezone
import streamlit as st

st.set_page_config(page_title="Enhanced Dashboard", layout="wide")
st.title("Enhanced Dashboard")


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
    kpi_card("BTC Trend", "Up", "Based on 50-day EMA", color="green")
with col2:
    kpi_card("Crypto 24h Volatility", "6.2%", "Hourly BB width percentile")
with col3:
    kpi_card("Market Breadth", "61% advancers", "Advancers vs decliners")
with col4:
    kpi_card("Mode", "DRY-RUN", "Toggle in Settings")


st.subheader("Today’s Market Report")
report_lines = (
    "Regime: Risk-On. BTC holds above 50-day EMA.",
    "Top Movers (24h): SOL +5.4%, LINK +4.1%, ADA −2.2%.",
    "Volatility: Moderate; BB width in 60th percentile.",
)
st.write("\n".join(f"- {line}" for line in report_lines))


st.subheader("Current Market Focus")
focus_items = (
    "BTC — Possible breakout: ML=0.61, momentum ↑, risk acceptable.",
    "ETH — Range-bound; wait for close above 20-day high.",
    "SOL — Momentum extended; consider partial take-profit.",
)
st.write("\n".join(f"• {item}" for item in focus_items))


st.subheader("System Health")
latest_files = [
    "logs/latest_market_outlook.txt",
    "logs/latest_opportunities.txt",
    "logs/latest_portfolio_analysis.txt",
    "logs/latest_risk_alerts.txt",
]
last_refresh = human_last_modified(latest_files)
st.caption(
    "Mode: DRY-RUN  |  Last data refresh: "
    f"{last_refresh}  |  Next scans: crypto 1m / equities pre-market in 3h"
)

st.info(
    "No chart here to keep focus. Check Signals or Research Lab for visuals."
)
st.caption(
    "Mode: DRY-RUN  |  Last data refresh: "
    f"{last_refresh}  |  Next scans: crypto 1m / equities pre-market in 3h"
)

st.info(
    "No chart shown here. See Signals for setups or Research Lab for visuals."
)
