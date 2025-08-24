import streamlit as st
from datetime import time
from src.ui_ext.settings_store import (
    load_settings,
    save_settings,
    send_telegram_test,
)
from src.ui_ext.ui_helpers import render_mode_pill

st.set_page_config(page_title="Settings", layout="wide")
st.title("Settings")
render_mode_pill()

settings = load_settings()

st.subheader("Mode")
dry_run = st.toggle("DRY-RUN", value=bool(settings.get("DRY_RUN", True)))

st.subheader("Scheduling Windows")
current = settings.get("SCHEDULE_TIME") or ""
hhmm = None
try:
    if current:
        h, m = [int(x) for x in current.split(":")]
        hhmm = time(hour=h, minute=m)
except Exception:
    hhmm = None

scan_time = st.time_input("Daily Scan Time (UTC)", value=hhmm)

st.subheader("Telegram")
chat_id = st.text_input("Chat ID", settings.get("TELEGRAM_CHAT_ID", ""))
colA, colB = st.columns(2)
with colA:
    if st.button("Save Settings"):
        new = {
            "DRY_RUN": bool(dry_run),
            "SCHEDULE_TIME": scan_time.strftime("%H:%M") if scan_time else "",
            "TELEGRAM_CHAT_ID": chat_id.strip(),
            "_LAST_TG_TEST": settings.get("_LAST_TG_TEST", ""),
        }
        save_settings(new)
        st.success("Saved")
with colB:
    if st.button("Send Test to Telegram"):
        import asyncio

        ok = asyncio.run(send_telegram_test(chat_id.strip()))
        st.success("Test sent") if ok else st.warning("Failed to send")

if last := settings.get("_LAST_TG_TEST"):
    st.caption(f"Last Telegram test: {last}")

st.caption(
    "To trade live, turn DRY_RUN off and add exchange keys (read-only here)."
)
