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

    # Log token length only (avoid leaking full token again)
    logger.info("Token length=%d", len(token))

    bot = TradingBot(token)
    logger.info("Starting polling (python-telegram-bot %s)", TG_VER)

    # Fallback startup notification using job queue (more reliable than post_init)
    if chat_id:
        jq = getattr(bot.application, 'job_queue', None)
        if jq is not None:
            async def _startup_job(context):  # type: ignore
                try:
                    await context.bot.send_message(chat_id=chat_id, text="Bot started ✅ (job queue)")
                    logger.info("Startup notification sent via job queue")
                except Exception as e:  # pragma: no cover
                    logger.warning("Failed to send startup notification (job queue): %s", e)
            jq.run_once(_startup_job, when=1)
        else:
            logger.warning("Job queue unavailable; cannot schedule startup notification")

    bot.application.run_polling(
        drop_pending_updates=True,
        allowed_updates=None,
    )


if __name__ == "__main__":
    main()
