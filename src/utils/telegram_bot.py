# src/utils/telegram_bot.py
from __future__ import annotations
import os, logging
from typing import Dict
from datetime import datetime, timezone

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
except ModuleNotFoundError:
    raise SystemExit("Install python-telegram-bot: pip install python-telegram-bot==21.*")

TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

STATE: Dict[str, float] = {"capital": 50_000.0, "pnl": 0.0}

def fmt_summary() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cap = STATE["capital"]; pnl = STATE["pnl"]; pnlp = (pnl / max(cap - pnl, 1e-9))*100
    return (f"📈 *Bot Summary* ({ts})\n"
            f"Capital: `${cap:,.2f}`\nRealized PnL: `${pnl:,.2f}` ({pnlp:.2f}%)\n"
            f"Open Positions: 0 (demo)\n")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton("Refresh Summary", callback_data="refresh"),
           InlineKeyboardButton("Reset PnL", callback_data="reset")]]
    await update.message.reply_markdown(fmt_summary(), reply_markup=InlineKeyboardMarkup(kb))

async def profits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_markdown(fmt_summary())

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Balance: ${STATE['capital']:,.2f}")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["pnl"] = 0.0
    await update.message.reply_text("PnL reset.")

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "refresh":
        await q.edit_message_text(fmt_summary(), parse_mode="Markdown")
    elif q.data == "reset":
        STATE["pnl"] = 0.0
        await q.edit_message_text("PnL reset.")

def run():
    if not TOKEN:
        logging.error("TELEGRAM_TOKEN missing.")
        return
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("profits", profits))
    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CallbackQueryHandler(button))
    app.run_polling()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
