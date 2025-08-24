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
        s = load_settings()
        s["_LAST_TG_TEST"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        save_settings(s)
        return True
    except Exception:
        return False


def send_telegram_message(text: str) -> bool:
    """Send a Telegram message using TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.

    Uses python-telegram-bot if available; falls back to HTTP API.
    Returns True on success.
    """
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = load_settings().get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    # Try async library quickly
    with suppress(Exception):
        from telegram import Bot  # type: ignore
        import asyncio

        async def _go():
            bot = Bot(token=token)
            await bot.send_message(chat_id=chat_id, text=text)

        asyncio.run(_go())
        return True
    # Fallback to HTTP (urllib)
    with suppress(Exception):
        import json as _json
        from urllib import request as _req

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = _json.dumps({
            "chat_id": chat_id,
            "text": text,
        }).encode("utf-8")
        req = _req.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with _req.urlopen(req, timeout=10) as resp:  # nosec - fixed timeout
            body = resp.read()
            data = _json.loads(body.decode("utf-8"))
            return bool(data.get("ok"))
    return False
