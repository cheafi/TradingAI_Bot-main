# src/utils/telegram_bot.py
"""Telegram interface (optional). Use TELEGRAM_TOKEN + TELEGRAM_CHAT_ID in .env."""
from __future__ import annotations
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
except Exception:
    ApplicationBuilder = None

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


async def suggest_handler(update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
    """/suggest SYMBOL - returns an example suggestion."""
    args = context.args or []
    symbol = args[0].upper() if args else "AAPL"
    text = f"📌 Suggestion for {symbol}\nAction: BUY\nTarget: $250\nStop: $230\nNote: Demo suggestion."
    await update.message.reply_text(text)

def run_telegram_bot():
    if not ApplicationBuilder:
        logger.warning("python-telegram-bot not installed; skip telegram.")
        return
    if not TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_TOKEN missing.")
        return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("suggest", suggest_handler))
    app.run_polling()
