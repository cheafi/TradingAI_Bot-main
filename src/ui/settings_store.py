"""Settings persistence for Streamlit UI.

Stores a small JSON at data/ui_settings.json with keys:
- DRY_RUN: bool
- SCHEDULE_TIME: str (HH:MM) in UTC or empty
- TELEGRAM_CHAT_ID: str

Also provides a helper to send a Telegram test message using
python-telegram-bot if environment variable TELEGRAM_TOKEN is set.
"""
from __future__ import annotations
from typing import Any, Dict
import json
import os
from datetime import datetime
from contextlib import suppress


SETTINGS_PATH = os.path.join("data", "ui_settings.json")


def load_settings() -> Dict[str, Any]:
    with suppress(Exception):
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "DRY_RUN": bool(data.get("DRY_RUN", True)),
                    "SCHEDULE_TIME": str(data.get("SCHEDULE_TIME", "")),
                    "TELEGRAM_CHAT_ID": str(data.get("TELEGRAM_CHAT_ID", "")),
                    "_LAST_TG_TEST": str(data.get("_LAST_TG_TEST", "")),
                }
    return {
        "DRY_RUN": True,
        "SCHEDULE_TIME": "",
        "TELEGRAM_CHAT_ID": "",
        "_LAST_TG_TEST": "",
    }


def save_settings(d: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with suppress(Exception):
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)


async def send_telegram_test(
    chat_id: str, text: str = "Test from Streamlit Settings"
) -> bool:
    token = os.getenv("TELEGRAM_TOKEN", "")
    if not token or not chat_id:
        return False
    try:
        from telegram import Bot
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text)
        # record last success time
        s = load_settings()
        s["_LAST_TG_TEST"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        save_settings(s)
        return True
    except Exception:
        return False

