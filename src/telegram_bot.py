import os
import json
import logging
from typing import Optional, Set, Tuple, List
from datetime import datetime, timezone
from pathlib import Path
from contextlib import suppress
from zoneinfo import ZoneInfo

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

try:  # optional
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

log = logging.getLogger("tg")

ROOT = Path(__file__).resolve().parents[1]
SUBSCRIBERS_FILE = ROOT / "subscribers.txt"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "data" / "daily_reports"
STATE_DIR = ROOT / ".run"
STATE_DIR.mkdir(exist_ok=True)
SENT_STATE_FILE = STATE_DIR / "sent_reports.json"


def _load_telegram_config() -> Tuple[Optional[str], Optional[str]]:
    """Load TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from env or secrets.toml."""
    # load .env if available
    if load_dotenv:
        with suppress(Exception):
            load_dotenv(dotenv_path=ROOT / ".env", override=False)
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token:
        return token, chat_id
    # try secrets.toml in repo root
    secrets_path = ROOT / "secrets.toml"
    if secrets_path.exists() and tomllib:
        try:
            data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            token = data.get("TELEGRAM_TOKEN") or token
            chat_id = data.get("TELEGRAM_CHAT_ID") or chat_id
            return token, chat_id
        except Exception as e:  # pragma: no cover
            log.warning("Failed reading secrets.toml: %s", e)
    return token, chat_id


TOKEN, DEFAULT_CHAT_ID = _load_telegram_config()
SUBSCRIBERS: Set[int] = set()
TIMEZONE = os.getenv("TIMEZONE", "UTC")
ENABLE_AI_SUMMARY = (
    os.getenv("ENABLE_AI_SUMMARY", "false").lower() in {"1", "true", "yes"}
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


def _load_json(path: Path) -> Optional[dict]:
    with suppress(Exception):
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _load_text(path: Path) -> Optional[str]:
    with suppress(Exception):
        return path.read_text(encoding="utf-8")
    return None


def _latest_in(dir_path: Path, names: List[str]) -> Optional[Path]:
    cands = [
        p for p in (dir_path.glob("**/*")) if p.is_file() and p.name in names
    ]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _compose_market_outlook() -> str:
    # Priority: logs/latest_market_outlook.txt -> reports/market_snapshot.json
    p = (
        _latest_in(LOGS_DIR, ["latest_market_outlook.txt"]) or
        LOGS_DIR / "latest_market_outlook.txt"
    )
    txt = _load_text(p) if p.exists() else None
    if txt:
        return txt.strip()
    if js := _load_json(REPORTS_DIR / "market_snapshot.json"):
        keys = [
            k for k in ("index", "trend", "volatility", "breadth") if k in js
        ]
        lines = ["📈 市場快照:"] + [f"• {k}: {js[k]}" for k in keys]
        return "\n".join(lines)
    return "📈 市場展望暫無資料（等待產出）"


def _compose_opportunities() -> str:
    p = (
        _latest_in(LOGS_DIR, ["latest_opportunities.txt"]) or
        LOGS_DIR / "latest_opportunities.txt"
    )
    txt = _load_text(p) if p.exists() else None
    if txt:
        return txt.strip()
    js = _load_json(REPORTS_DIR / "opportunities.json")
    if js and isinstance(js, dict) and js.get("items"):
        lines = ["💡 機會清單:"]
        for it in js["items"][:5]:
            sym = it.get("symbol", "?")
            view = it.get("view", "n/a")
            score = it.get("score", "-")
            lines.append(f"• {sym}: {view} (score {score})")
        return "\n".join(lines)
    return "💡 尚無新機會（等待產出）"


def _compose_risk_alerts() -> str:
    p = (
        _latest_in(LOGS_DIR, ["latest_risk_alerts.txt"]) or
        LOGS_DIR / "latest_risk_alerts.txt"
    )
    txt = _load_text(p) if p.exists() else None
    if txt:
        return txt.strip()
    if js := _load_json(REPORTS_DIR / "risk.json"):
        items = [f"• {k}: {v}" for k, v in js.items()]
        lines = ["⚠️ 風險提示:"] + items
        return "\n".join(lines)
    return "⚠️ 目前無特別風險提示"


def _compose_closing_summary() -> str:
    p = (
        _latest_in(LOGS_DIR, ["market_close_summary.txt"]) or
        LOGS_DIR / "market_close_summary.txt"
    )
    txt = _load_text(p) if p.exists() else None
    if txt:
        return txt.strip()
    js = _load_json(REPORTS_DIR / "closing_summary.json")
    if js:
        return "📘 收盤總結:\n" + json.dumps(js, ensure_ascii=False, indent=2)
    return "📘 收盤總結暫無資料"


def _load_subscribers_from_disk() -> None:
    with suppress(Exception):
        if SUBSCRIBERS_FILE.exists():
            content = SUBSCRIBERS_FILE.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.isdigit():
                    SUBSCRIBERS.add(int(line))


def _maybe_ai_summarize_sync(text: str) -> str:
    if not ENABLE_AI_SUMMARY:
        return text
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_MODELS_API_KEY")
    if not api_key:
        return text
    try:
        import importlib

        m = importlib.import_module("openai")
        OpenAI = getattr(m, "OpenAI", None)
        if OpenAI is None:
            return text
        client = (
            OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
            if OPENAI_BASE_URL
            else OpenAI(api_key=api_key)
        )
        prompt = (
            "請將以下市場報告濃縮成 8-12 行要點，保留關鍵數字與風險提示，"
            "避免冗詞與口號，輸出為純文字：\n\n" + text
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        content = resp.choices[0].message.content.strip()
        return content or text
    except Exception:
        return text


async def _summary() -> str:
    # TODO: connect to your state/equity/pnl store
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"📊 Summary @ {now}\n"
        "Balance: $50,000\n"
        "PnL (today): +$123.45\n"
        "Open: 0"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton("Summary", callback_data="summary")]]
    await update.message.reply_text(
        (
            "🤖 TradingAI Bot ready. Try /summary /suggest /help or "
            "/subscribe"
        ),
        reply_markup=InlineKeyboardMarkup(kb),
    )


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    base = await _summary()
    outlook = _compose_market_outlook()
    opp = _compose_opportunities()
    risk = _compose_risk_alerts()
    text = f"{base}\n\n{outlook}\n\n{opp}\n\n{risk}"
    await update.message.reply_text(text[:4096])


async def suggest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = (context.args[0] if context.args else "BTC/USDT").upper()
    # TODO: call your ML suggestion; placeholder:
    await update.message.reply_text(
        (
            f"📈 Suggestion for {symbol}: Strong Buy\n"
            "Target: $73,000\n"
            "Stop: ATR-based"
        )
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = [
        "Commands:",
        "/start - greet and menu",
        "/summary - show quick system summary",
        "/suggest <SYMBOL> - demo suggestion (e.g., /suggest AAPL)",
        "/subscribe - subscribe this chat to broadcasts",
        "/status - show bot status",
        "/ping - check bot is alive",
    ]
    await update.message.reply_text("\n".join(parts))


def _persist_subscribers():
    try:
        SUBSCRIBERS_FILE.write_text(
            "\n".join(str(x) for x in sorted(SUBSCRIBERS)),
            encoding="utf-8",
        )
    except Exception as e:  # pragma: no cover
        log.warning("Failed to persist subscribers: %s", e)


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in SUBSCRIBERS:
        SUBSCRIBERS.add(chat_id)
        _persist_subscribers()
    await update.message.reply_text(
        "Subscribed ✅. You'll receive future broadcasts here."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        (
            f"Bot OK. Subscribers: {len(SUBSCRIBERS)}. Time: "
            f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
        )
    )


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong 🟢")


def run_telegram_bot():
    if not TOKEN:
        log.warning("TELEGRAM_TOKEN not set; skipping telegram bot.")
        return
    # seed default chat id if provided
    if DEFAULT_CHAT_ID:
        with suppress(Exception):
            SUBSCRIBERS.add(int(DEFAULT_CHAT_ID))

    # load subscribers from file if exists
    with suppress(Exception):
        if SUBSCRIBERS_FILE.exists():
            content = SUBSCRIBERS_FILE.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.isdigit():
                    SUBSCRIBERS.add(int(line))

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("suggest", suggest))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("ping", ping))
    # scheduling: use JobQueue when available; fallback to asyncio loop
    tz = ZoneInfo(TIMEZONE)

    async def _broadcast(text: str, bot: Bot):
        if not SUBSCRIBERS:
            return
        for cid in list(SUBSCRIBERS):
            with suppress(Exception):
                await bot.send_message(chat_id=cid, text=text[:4096])

    async def _send_morning(_: ContextTypes.DEFAULT_TYPE):
        await _broadcast("🌅 早間市場概要\n\n" + _compose_market_outlook(), app.bot)

    async def _send_midday(_: ContextTypes.DEFAULT_TYPE):
        await _broadcast("🕛 午間快訊\n\n" + _compose_opportunities(), app.bot)

    async def _send_close(_: ContextTypes.DEFAULT_TYPE):
        await _broadcast("🔔 收盤總結\n\n" + _compose_closing_summary(), app.bot)

    async def _send_evening(_: ContextTypes.DEFAULT_TYPE):
        await _broadcast("🌙 晚間前瞻\n\n" + _compose_market_outlook(), app.bot)

    jq = app.job_queue
    if jq:
        from datetime import time as dtime

        jq.run_daily(
            _send_morning, time=dtime(9, 0, tzinfo=tz), name="morning"
        )
        jq.run_daily(_send_midday, time=dtime(12, 0, tzinfo=tz), name="midday")
        jq.run_daily(_send_close, time=dtime(16, 0, tzinfo=tz), name="close")
        jq.run_daily(
            _send_evening, time=dtime(20, 0, tzinfo=tz), name="evening"
        )
        log.info("JobQueue scheduled daily reports in %s", TIMEZONE)
    else:
        import asyncio

        async def _loop():
            sent: dict = {}
            with suppress(Exception):
                if SENT_STATE_FILE.exists():
                    sent = json.loads(
                        SENT_STATE_FILE.read_text(encoding="utf-8")
                    )
            while True:
                now = datetime.now(tz).replace(second=0, microsecond=0)
                h, m = now.hour, now.minute
                key = None
                if (h, m) == (9, 0):
                    await _send_morning(None)
                    key = "morning"
                elif (h, m) == (12, 0):
                    await _send_midday(None)
                    key = "midday"
                elif (h, m) == (16, 0):
                    await _send_close(None)
                    key = "close"
                elif (h, m) == (20, 0):
                    await _send_evening(None)
                    key = "evening"
                if key:
                    sent[key] = now.date().isoformat()
                    with suppress(Exception):
                        SENT_STATE_FILE.write_text(
                            json.dumps(sent), encoding="utf-8"
                        )
                await asyncio.sleep(60)

        app.create_task(_loop())
        log.info("Asyncio fallback scheduler started in %s", TIMEZONE)
    log.info("Starting Telegram bot…")
    app.run_polling(drop_pending_updates=True, allowed_updates=None)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser()
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Send a self-check message and exit",
    )
    p.add_argument(
        "--message",
        default=None,
        help="Custom message for self-test",
    )
    args = p.parse_args()

    if args.self_test:
        if not TOKEN:
            print("Missing TELEGRAM_TOKEN; configure .env or secrets.toml")
            raise SystemExit(1)
        if not DEFAULT_CHAT_ID:
            print(
                "Missing TELEGRAM_CHAT_ID; set env or secrets.toml to run "
                "self-test"
            )
            raise SystemExit(2)
        text = args.message or (
            "TradingAI Bot self-check at "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        import asyncio

        async def _send():
            bot = Bot(token=TOKEN)
            await bot.send_message(chat_id=int(DEFAULT_CHAT_ID), text=text)

        asyncio.run(_send())
        print("Self-test message sent ✅")
    else:
        run_telegram_bot()
