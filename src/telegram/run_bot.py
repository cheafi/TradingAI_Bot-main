# Telegram bot entrypoint
from __future__ import annotations
import os
import logging

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass  # dotenv optional

from telegram import __version__ as TG_VER  # type: ignore
from src.telegram.enhanced_bot import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger("run_bot")


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment/.env")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if chat_id:
        logger.info("Startup chat_id detected: %s (will send startup ping)", chat_id)
    else:
        logger.info("No TELEGRAM_CHAT_ID provided; skipping startup notification")

    bot = TradingBot(token)
    logger.info("Starting polling (python-telegram-bot %s)", TG_VER)

    async def _notify_startup() -> None:
        if not chat_id:
            return
        try:
            await bot.application.bot.send_message(chat_id=chat_id, text="Bot started ✅")
            logger.info("Startup notification sent")
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to send startup notification: %s", e)

    # Use create_task after run_polling starts by passing post_init callback
    async def _post_init(application):  # type: ignore
        application.create_task(_notify_startup())

    bot.application.post_init = _post_init  # type: ignore

    bot.application.run_polling(
        drop_pending_updates=True,
        allowed_updates=None,
    )


if __name__ == "__main__":
    main()
