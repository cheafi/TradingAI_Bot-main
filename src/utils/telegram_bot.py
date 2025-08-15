# src/utils/telegram_bot.py
import os
import logging
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False
    logger.debug("python-telegram-bot not installed.")

async def suggest_handler(update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
    args = context.args or []
    symbol = args[0].upper() if args else "AAPL"
    text = f"📌 Suggestion for {symbol}\nAction: BUY (demo)\nTarget: $250\nStop: $230\nReason: demo ensemble"
    await update.message.reply_text(text)

def run_telegram_bot():
    if not TELEGRAM_AVAILABLE:
        logger.warning("telegram not available; skip launching bot.")
        return
    if not TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_TOKEN missing; skip telegram.")
        return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("suggest", suggest_handler))
    app.run_polling()