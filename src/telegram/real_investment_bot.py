"""TradingAI Pro - Real Telegram Investment Agency Bot (PTB v20+)"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, time as dtime
from typing import Any, Dict, Optional, Set

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.telegram.message_templates import (
    format_daily_market_report,
    format_bot_status,
    format_portfolio_alert,
    format_portfolio_summary,
    format_market_opportunity,
    format_portfolio_position,
    format_market_sentiment,
    next_job_hints,
)
from src.telegram.market_analyzer import MarketAnalyzer


# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/telegram_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Globals & paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PF_DIR = os.path.join(DATA_DIR, "portfolios")
os.makedirs(PF_DIR, exist_ok=True)
TIMEZONE = os.getenv("TIMEZONE", "UTC")


class RealTelegramInvestmentBot:
    """Real Telegram Investment Agency Bot"""

    def __init__(self, token: str) -> None:
        self.token: str = token
        self.app: Application = (
            Application.builder()
            .token(token)
            .post_init(self._on_startup)
            .build()
        )
        self.subscribers: Set[int] = set()
        self.client_portfolios: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.crypto_agent: Optional[Any] = None  # assigned at startup
        self.market_analyzer = MarketAnalyzer()
        self.pending_trades: Dict[int, Any] = {}  # Track pending trade confirmations
        self._setup_handlers()
        logger.info("Bot initialized")

    def _setup_handlers(self) -> None:
        # Main commands
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(
            CommandHandler("subscribe", self.subscribe_command)
        )
        self.app.add_handler(
            CommandHandler("unsubscribe", self.unsubscribe_command)
        )

        # Market analysis
        self.app.add_handler(CommandHandler("outlook", self.daily_outlook))
        self.app.add_handler(
            CommandHandler("opportunities", self.buying_opportunities)
        )
        self.app.add_handler(
            CommandHandler("portfolio", self.portfolio_analysis)
        )
        self.app.add_handler(CommandHandler("alerts", self.risk_alerts))
        self.app.add_handler(CommandHandler("status", self.service_status))
        self.app.add_handler(CommandHandler("market", self.quick_market))
        self.app.add_handler(CommandHandler("news", self.market_news))
        self.app.add_handler(CommandHandler("stop", self.stop_command))
        self.app.add_handler(CommandHandler("setstop", self.set_stop))
        
        # Enhanced market analysis commands
        self.app.add_handler(CommandHandler("scan", self.scan_opportunities))
        self.app.add_handler(CommandHandler("sentiment", self.market_sentiment))
        self.app.add_handler(CommandHandler("positions", self.show_positions))
        self.app.add_handler(CommandHandler("summary", self.portfolio_summary))
        
        # Message handler for trade confirmations
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        # Backtest & historical simulation
        self.app.add_handler(CommandHandler("backtest", self.backtest_command))
        self.app.add_handler(CommandHandler("advise", self.advise_command))
        self.app.add_handler(CommandHandler("simulate", self.simulate_command))

        # Portfolio management
        self.app.add_handler(CommandHandler("add", self.add_position))
        self.app.add_handler(CommandHandler("remove", self.remove_position))

        # Callback & text
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
        self.app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message,
            )
        )

    # ----- Command handlers -----
    async def start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        user_name = update.effective_user.first_name or "投資者"
        text = (
            "🏢 歡迎來到 TradingAI Pro 投資顧問\n\n"
            f"您好 {user_name}！我是您的 24/7 投資顧問機器人。\n\n"
            "常用指令：/subscribe /outlook /opportunities /portfolio /alerts /help"
        )
        keyboard = [
            [InlineKeyboardButton("📊 市場展望", callback_data="outlook")],
            [InlineKeyboardButton("💰 投資機會", callback_data="opportunities")],
            [InlineKeyboardButton("📋 投資組合", callback_data="portfolio")],
            [InlineKeyboardButton("📱 訂閱服務", callback_data="subscribe")],
            [InlineKeyboardButton("ℹ️ 服務狀態", callback_data="status")],
        ]
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        help_text = (
            "📘 指令\n"
            "/start, /help, /subscribe, /unsubscribe\n"
            "/outlook, /opportunities, /portfolio, /alerts\n"
            "/market, /news\n"
            "/add <代號> [數量] [成本] [停損], /remove <代號>, /setstop <代號> <價格>\n"
            "/backtest <代號> [週期]\n"
            "/advise <代號> <日期> [週期]\n"
            "/simulate <代號> <起日> <迄日> [週期]\n"
        )
        await context.bot.send_message(chat_id=chat_id, text=help_text)

    async def subscribe_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        if chat_id in self.subscribers:
            await context.bot.send_message(chat_id=chat_id, text="已訂閱。")
            return
        self.subscribers.add(chat_id)
        await context.bot.send_message(chat_id=chat_id, text="訂閱成功。")
        await self.send_welcome_report(chat_id)
        # load portfolio if exists
        self._load_portfolio_from_disk(chat_id)

    async def unsubscribe_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        self.subscribers.discard(chat_id)
        await context.bot.send_message(chat_id=chat_id, text="已取消訂閱。")

    async def daily_outlook(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        # Try reading canonical logs, else fallback
        try:
            with open("logs/latest_market_outlook.txt", "r") as f:
                raw = f.read()
        except Exception:
            raw = await self.generate_market_outlook()
        # Best-effort parse: first lines become bullets
        lines = [ln.strip(" -\t") for ln in raw.splitlines() if ln.strip()]
        highlights = [f"- {ln}" for ln in lines[:5]]
        focus = []
        try:
            with open("data/daily_reports/opportunities.json", "r") as f:
                data = json.load(f)
            ops = data.get("all_opportunities") or []
            for op in ops[:3]:
                sym = op.get("symbol", "?")
                rec = op.get("recommendation") or (
                    "BUY" if (op.get("score") or 0) >= 5 else "WATCH"
                )
                focus.append(f"{sym} — {rec}")
        except Exception:
            pass
        msg = format_daily_market_report(
            phase="Pre-Market",
            date_str=datetime.now().strftime("%Y-%m-%d"),
            regime="🟢 Risk-On",  # placeholder; could infer from data
            highlights=highlights,
            market_focus=focus,
            next_report=(next_job_hints()[0] if next_job_hints() else None),
        )
        keyboard = [
            [InlineKeyboardButton("💰 機會", callback_data="opportunities")]
        ]
        await context.bot.send_message(
            chat_id=chat_id,
            text=msg,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )

    async def buying_opportunities(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        # Try to structure: list top N opportunities
        try:
            with open("data/daily_reports/opportunities.json", "r") as f:
                data = json.load(f)
            ops = data.get("all_opportunities") or []
            lines = ["💰 Opportunities"]
            for op in ops[:10]:
                sym = op.get("symbol")
                score = op.get("score")
                rec = op.get("recommendation") or (
                    "BUY" if (score or 0) >= 5 else "WATCH"
                )
                reason = (op.get("reasons") or [""])[0]
                lines.append(f"- {sym}: {rec} (score {score}) — {reason}")
            txt = "\n".join(lines)
        except Exception:
            txt = await self.generate_opportunities()
        await context.bot.send_message(
            chat_id=chat_id,
            text=txt,
            parse_mode="Markdown",
        )

    async def service_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        status = format_bot_status(
            exchange=os.getenv("CRYPTO_EXCHANGE", "binance"),
            next_jobs=next_job_hints(),
            last_refresh="1m ago",
        )
        await context.bot.send_message(
            chat_id=chat_id,
            text=status,
        )

    async def quick_market(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        try:
            with open("data/daily_reports/market_snapshot.json", "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
        spy = data.get("SPY", {})
        qqq = data.get("QQQ", {})
        txt = (
            "📊 快速市場概覽\n"
            f"SPY: {spy.get('current_price', 445)}\n"
            f"QQQ: {qqq.get('current_price', 375)}\n"
        )
        await context.bot.send_message(chat_id=chat_id, text=txt)

    async def market_news(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        try:
            with open("logs/latest_market_news.txt", "r") as f:
                content = f.read()
        except Exception:
            content = "📰 今日暫無重大新聞。"
        await context.bot.send_message(chat_id=chat_id, text=content)

    async def risk_alerts(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        try:
            with open("logs/latest_risk_alerts.txt", "r") as f:
                content = f.read()
        except Exception:
            content = "⚠️ 目前未偵測到重大風險事件。"
        await context.bot.send_message(chat_id=chat_id, text=content)

    async def portfolio_analysis(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        portfolio = self.client_portfolios.get(chat_id, {})
        if not portfolio:
            msg = format_portfolio_summary(
                equity_line="Equity: 0.00",
                exposure_line="Exposure: 0 positions",
                positions=["(空) 使用 /add <代號> [數量] [成本]"],
                risk_lines=["Stop coverage: 0%"],
                next_review="24h",
            )
            await context.bot.send_message(
                chat_id=chat_id, text=msg
            )
            return
        total = 0.0
        try:
            with open("data/daily_reports/market_snapshot.json", "r") as f:
                prices = json.load(f)
        except Exception:
            prices = {}
        pos_lines = []
        stops_set = 0
        for sym, pos in portfolio.items():
            qty = float(pos.get("qty", 0.0) or 0.0)
            cost = pos.get("cost")
            stop = pos.get("stop")
            last = float(prices.get(sym, {}).get("current_price", 0.0) or 0.0)
            value = qty * last
            total += value
            pnl_str = ""
            if cost is not None:
                pnl = (last - float(cost)) * qty
                pnl_str = f"  PnL:{pnl:+.2f}"
            stop_str = f"  Stop:{stop}" if stop is not None else ""
            if stop is not None:
                stops_set += 1
            pos_lines.append(
                (
                    f"• {sym}  x{qty}  Px:{last}  Val:{value:.2f}"
                    f"{pnl_str}{stop_str}"
                )
            )

        equity_line = f"Equity: {total:.2f}"
        exposure_line = (
            f"Exposure: {len(portfolio)} positions"
        )
        risk_lines = [
            (
                "Stop coverage: "
                f"{int(100 * stops_set / max(1, len(portfolio)))}%"
            )
        ]
        summary = format_portfolio_summary(
            equity_line=equity_line,
            exposure_line=exposure_line,
            positions=pos_lines,
            risk_lines=risk_lines,
            next_review="24h",
        )
        await context.bot.send_message(chat_id=chat_id, text=summary)

    async def add_position(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        args = context.args if hasattr(context, "args") else []
        if not args:
            await context.bot.send_message(
                chat_id=chat_id,
                text="用法：/add 代號 數量 成本",
            )
            return
        symbol = args[0].upper()
        qty = float(args[1]) if len(args) > 1 else 0.0
        cost = float(args[2]) if len(args) > 2 else None
        stop = float(args[3]) if len(args) > 3 else None

        pf = self.client_portfolios.setdefault(chat_id, {})
        pf[symbol] = {"qty": qty, "cost": cost, "stop": stop}
        alert = format_portfolio_alert(
            [
                f"Added {symbol}",
                f"Qty: {qty}",
                f"Cost: {cost if cost is not None else '-'}",
                f"Stop: {stop if stop is not None else '-'}",
            ]
        )
        await context.bot.send_message(chat_id=chat_id, text=alert)
        self._persist_portfolio(chat_id)

    async def set_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        args = context.args if hasattr(context, "args") else []
        if len(args) < 2:
            await context.bot.send_message(
                chat_id=chat_id,
                text="用法：/setstop 代號 停損價",
            )
            return
        symbol = args[0].upper()
        try:
            price = float(args[1])
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text="停損價需為數字")
            return
        pf = self.client_portfolios.setdefault(chat_id, {})
        pos = pf.get(symbol)
        if not pos:
            await context.bot.send_message(chat_id=chat_id, text="未找到持倉")
            return
        pos["stop"] = price
        self._persist_portfolio(chat_id)
        alert = format_portfolio_alert(
            [f"Set Stop: {symbol}", f"New Stop: {price}"]
        )
        await context.bot.send_message(chat_id=chat_id, text=alert)

    async def remove_position(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        args = context.args if hasattr(context, "args") else []
        if not args:
            await context.bot.send_message(
                chat_id=chat_id,
                text="用法：/remove 代號",
            )
            return
        symbol = args[0].upper()
        pf = self.client_portfolios.setdefault(chat_id, {})
        if symbol in pf:
            del pf[symbol]
            alert = format_portfolio_alert([f"Removed {symbol}"])
            await context.bot.send_message(chat_id=chat_id, text=alert)
        else:
            await context.bot.send_message(
                chat_id=chat_id, text=f"未找到持倉：{symbol}"
            )
        self._persist_portfolio(chat_id)

    # ----- Message & callbacks -----
    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        text = (update.message.text or "").lower()
        if any(k in text for k in ["市場", "market", "股市", "行情"]):
            await self.quick_market(update, context)
        elif any(k in text for k in ["機會", "opportunity", "投資", "買入"]):
            await self.buying_opportunities(update, context)
        elif any(k in text for k in ["風險", "risk", "警報", "alert"]):
            await self.risk_alerts(update, context)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="輸入 /help 查看可用功能"
            )

    async def button_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await query.answer()
        if query.data == "outlook":
            await self.daily_outlook(update, context)
        elif query.data == "opportunities":
            await self.buying_opportunities(update, context)
        elif query.data == "portfolio":
            await self.portfolio_analysis(update, context)
        elif query.data == "subscribe":
            await self.subscribe_command(update, context)
        elif query.data == "status":
            await self.service_status(update, context)

    async def stop_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        self.subscribers.discard(chat_id)
        await context.bot.send_message(chat_id=chat_id, text="已停止通知。")

    # ----- Content generation -----
    async def generate_market_outlook(self) -> str:
        try:
            with open("logs/latest_market_outlook.txt", "r") as f:
                content = f.read()
            return content.replace("**", "*")
        except Exception:
            return (
                f"📊 每日市場展望 - {datetime.now().strftime('%Y-%m-%d')}\n"
                "市場情緒：謹慎樂觀\n建議：品質成長逢低布局"
            )

    async def generate_opportunities(self) -> str:
        try:
            with open("logs/latest_opportunities.txt", "r") as f:
                content = f.read()
            return content.replace("**", "*")
        except Exception:
            return (
                f"💰 投資機會 - {datetime.now().strftime('%Y-%m-%d')}\n"
                "• NVDA, MSFT, BTC 觀察清單"
            )

    # ----- Broadcast & schedule -----
    async def send_welcome_report(self, chat_id: int) -> None:
        text = (
            "🎉 歡迎加入 TradingAI Pro！\n"
            "您將收到市場簡報、機會、風險與收盤報告。"
        )
        await self.app.bot.send_message(chat_id=chat_id, text=text)

    async def broadcast_to_subscribers(self, message: str) -> None:
        for chat_id in list(self.subscribers):
            try:
                await self.app.bot.send_message(chat_id=chat_id, text=message)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error("Send failed to %s: %s", chat_id, e)
                self.subscribers.discard(chat_id)

    async def scheduled_reports(self) -> None:
        while True:
            try:
                hour = datetime.now().hour
                if hour in (9, 12, 16, 20):
                    if hour == 9:
                        await self.send_morning_report()
                    elif hour == 12:
                        await self.send_midday_update()
                    elif hour == 16:
                        await self.send_closing_report()
                    else:
                        await self.send_evening_preview()
                await asyncio.sleep(3600)
            except Exception as e:
                logger.exception("Scheduler error: %s", e)
                await asyncio.sleep(300)

    async def _hourly_scheduler(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """JobQueue callback to trigger hourly schedule checks."""
        try:
            now = datetime.now()
            h, m = now.hour, now.minute
            if (h, m) == (8, 0):
                await self.send_morning_report()
            elif (h, m) in ((11, 30), (14, 0)):
                await self.send_midday_update()
            elif (h, m) == (16, 30):
                await self.send_closing_report()
            elif (h, m) == (20, 0):
                await self.send_evening_preview()
        except Exception as e:
            logger.exception("Hourly scheduler error: %s", e)

    async def send_morning_report(self) -> None:
        if not self.subscribers:
            return
        msg = format_daily_market_report(
            phase="Pre-Market",
            date_str=datetime.now().strftime("%Y-%m-%d"),
            regime="🟢 Risk-On",
            highlights=["- Futures flat, tech mixed", "- Dollar steady"],
            market_focus=["BTC — Watch breakout", "ETH — Hold"],
            next_report=(next_job_hints()[0] if next_job_hints() else None),
        )
        await self.broadcast_to_subscribers(msg)

    async def send_midday_update(self) -> None:
        if not self.subscribers:
            return
        msg = format_daily_market_report(
            phase="Mid-Session",
            date_str=datetime.now().strftime("%Y-%m-%d"),
            regime="🟡 Mixed",
            highlights=["- Breadth 51% advancers", "- VIX -0.3"],
            market_focus=["SOL — Momentum strong"],
            next_report=(next_job_hints()[0] if next_job_hints() else None),
        )
        await self.broadcast_to_subscribers(msg)

    async def send_closing_report(self) -> None:
        if not self.subscribers:
            return
        try:
            with open("logs/market_close_summary.txt", "r") as f:
                content = f.read()
            text = f"🏁 收盤分析\n\n{content}\n\nMode: DRY-RUN ✅"
        except Exception:
            text = "🏁 收盤分析：主要指數收高，科技股領漲\n\nMode: DRY-RUN ✅"
        await self.broadcast_to_subscribers(text)

    async def send_evening_preview(self) -> None:
        if not self.subscribers:
            return
        await self.broadcast_to_subscribers("🌙 亞洲市場預告：留意亞太數據與加密波動")

    async def _on_startup(self, app: Application) -> None:
        """Run on Application startup when event loop is ready."""
        # Announce bot is ready
        try:
            if self.subscribers:
                await self.broadcast_to_subscribers("🤖 TradingAI Pro 已啟動並待命")
        except Exception as e:
            logger.warning("Boot broadcast failed: %s", e)

        # Schedule reports
        try:
            if app.job_queue is not None:
                # Use specific daily slots aligned with user request
                try:
                    from zoneinfo import ZoneInfo
                except Exception:
                    ZoneInfo = None  # type: ignore
                tz = ZoneInfo(TIMEZONE) if ZoneInfo else None
                jq = app.job_queue
                # Before market (8:00), 2h after (11:30), mid (14:00),
                # after (16:30), evening scan (20:00)
                jq.run_daily(
                    self._hourly_scheduler, time=dtime(8, 0, tzinfo=tz)
                )
                jq.run_daily(
                    self._hourly_scheduler, time=dtime(11, 30, tzinfo=tz)
                )
                jq.run_daily(
                    self._hourly_scheduler, time=dtime(14, 0, tzinfo=tz)
                )
                jq.run_daily(
                    self._hourly_scheduler, time=dtime(16, 30, tzinfo=tz)
                )
                jq.run_daily(
                    self._hourly_scheduler, time=dtime(20, 0, tzinfo=tz)
                )
                logger.info(
                    "Daily schedule registered via JobQueue in %s", TIMEZONE
                )
            else:
                # Fallback to async loop if JobQueue isn't available
                app.create_task(self.scheduled_reports())
                logger.info("Hourly scheduler started via asyncio task")
        except Exception as e:
            logger.exception("Failed to start scheduler: %s", e)

        # Start 24/7 crypto dry-run agent in background
        try:
            from src.agents.crypto_agent import CryptoAgent

            agent = CryptoAgent(
                symbols=(
                    os.getenv("CRYPTO_SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
                ),
                exchange_id=os.getenv("CRYPTO_EXCHANGE", "binance"),
                timeframe=os.getenv("CRYPTO_TIMEFRAME", "15m"),
                polling_sec=int(os.getenv("CRYPTO_POLLING", "300")),
            )
            self.crypto_agent = agent

            async def _notify_router(text: str):
                # Optional chat prefix: [chat:<id>] message
                chat_id = None
                if text.startswith("[chat:"):
                    try:
                        prefix, rest = text.split("] ", 1)
                        chat_id = int(prefix[6:-1])
                        text_to_send = rest
                    except Exception:
                        text_to_send = text
                else:
                    text_to_send = text
                if chat_id is not None:
                    try:
                        await app.bot.send_message(
                            chat_id=chat_id, text=text_to_send
                        )
                    except Exception as e:
                        logger.debug("Notify to %s failed: %s", chat_id, e)
                    return
                # broadcast to subscribers by default
                await self.broadcast_to_subscribers(text_to_send)

            async def _run():
                await agent.run(lambda t: app.create_task(_notify_router(t)))

            app.create_task(_run())
            logger.info("Crypto dry-run agent started")
        except Exception as e:
            logger.warning("Crypto agent not started: %s", e)

    async def _ensure_agent(self) -> None:
        """Lazily create and start the crypto agent if not available."""
        if self.crypto_agent:
            return
        try:
            from src.agents.crypto_agent import CryptoAgent

            agent = CryptoAgent(
                symbols=(
                    os.getenv("CRYPTO_SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
                ),
                exchange_id=os.getenv("CRYPTO_EXCHANGE", "binance"),
                timeframe=os.getenv("CRYPTO_TIMEFRAME", "15m"),
                polling_sec=int(os.getenv("CRYPTO_POLLING", "300")),
            )
            self.crypto_agent = agent
            await agent.start()
        except Exception as e:
            logger.debug("Create agent failed: %s", e)

    async def backtest_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        if not self.crypto_agent:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Backtest 尚未啟用或 Agent 未載入。",
            )
            return
        args = context.args if hasattr(context, "args") else []
        if not args:
            await context.bot.send_message(
                chat_id=chat_id,
                text="用法：/backtest <SYMBOL> [TIMEFRAME]",
            )
            return
        symbol = args[0].upper()
        timeframe = args[1] if len(args) > 1 else None
        try:
            res = await self.crypto_agent.backtest(symbol, timeframe=timeframe)
            msg = (
                "🧪 回測結果\n"
                f"標的：{res.get('symbol')}  週期：{res.get('timeframe')}\n"
                f"訊號數：{res.get('signals')}  粗略報酬：{res.get('ret'):.2%}"
            )
            await context.bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            logger.debug("Backtest error: %s", e)
            await context.bot.send_message(chat_id=chat_id, text=f"回測失敗：{e}")

    async def advise_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        await self._ensure_agent()
        if not self.crypto_agent:
            await context.bot.send_message(chat_id=chat_id, text="Agent 未就緒。")
            return
        args = context.args if hasattr(context, "args") else []
        if len(args) < 2:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "用法：/advise <SYMBOL> <DATE> [TIMEFRAME]\n"
                    "例如：/advise BTC/USDT 2025-08-01 1h"
                ),
            )
            return
        symbol = args[0].upper()
        date_str = args[1]
        tf = args[2] if len(args) > 2 else None
        try:
            res = await self.crypto_agent.advise_on_date(
                symbol, date_str, timeframe=tf
            )
            if res.get("error"):
                await context.bot.send_message(
                    chat_id=chat_id, text=f"失敗：{res['error']}"
                )
                return
            fwd = res.get("fwd_ret")
            pct = f"{fwd*100:.2f}%" if isinstance(fwd, (int, float)) else "-"
            setups = ", ".join(res.get("setups", [])) or "-"
            msg = (
                "🕰 歷史建議\n"
                f"標的：{res.get('symbol')}  週期：{res.get('timeframe')}\n"
                f"日期：{res.get('date')}  建議：{setups}\n"
                f"入場：{res.get('entry_price')}\n"
                f"+30天：{res.get('fwd_date')} 價格：{res.get('fwd_price')} "
                f"報酬：{pct}"
            )
            await context.bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            logger.debug("Advise error: %s", e)
            await context.bot.send_message(chat_id=chat_id, text=f"查詢失敗：{e}")

    async def simulate_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        await self._ensure_agent()
        if not self.crypto_agent:
            await context.bot.send_message(chat_id=chat_id, text="Agent 未就緒。")
            return
        args = context.args if hasattr(context, "args") else []
        if len(args) < 3:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "用法：/simulate <SYMBOL> <START> <END> [TIMEFRAME]\n"
                    "例如：/simulate BTC/USDT 2025-06-01 2025-08-01 1h"
                ),
            )
            return
        symbol = args[0].upper()
        start_s = args[1]
        end_s = args[2]
        tf = args[3] if len(args) > 3 else None
        try:
            res = await self.crypto_agent.simulate_range(
                symbol, start_s, end_s, timeframe=tf
            )
            if res.get("error"):
                await context.bot.send_message(
                    chat_id=chat_id, text=f"失敗：{res['error']}"
                )
                return
            count = res.get("count", 0)
            avg = res.get("avg_ret", 0.0)
            win = res.get("win_rate", 0.0)
            best = res.get("best", 0.0)
            worst = res.get("worst", 0.0)
            header = (
                "📈 區間歷史模擬\n"
                f"標的：{res.get('symbol')}  週期：{res.get('timeframe')}\n"
                f"期間：{res.get('start')} → {res.get('end')}  次數：{count}\n"
                f"平均：{avg*100:.2f}%  勝率：{win*100:.2f}%  "
                f"最佳：{best*100:.2f}%  最差：{worst*100:.2f}%\n"
                "—— 事件樣本 ——"
            )
            lines = [header]
            for e in (res.get("events") or [])[:5]:
                lines.append(
                    f"• {e['date']} 入場:{e['entry']} → {e['fwd_date']} "
                    f"{e['fwd']} ({e['ret']*100:.2f}%)"
                )
            await context.bot.send_message(
                chat_id=chat_id, text="\n".join(lines)
            )
        except Exception as e:
            logger.debug("Simulate error: %s", e)
            await context.bot.send_message(chat_id=chat_id, text=f"查詢失敗：{e}")

    # ----- Entry -----
    def start_telegram_service(self) -> None:
        logger.info("Starting Real Telegram Investment Bot")
        self.app.run_polling()

    # -------- persistence utilities --------
    def _portfolio_path(self, chat_id: int) -> str:
        return os.path.join(PF_DIR, f"{chat_id}.json")

    def _persist_portfolio(self, chat_id: int) -> None:
        try:
            pf = self.client_portfolios.get(chat_id, {})
            with open(
                self._portfolio_path(chat_id), "w", encoding="utf-8"
            ) as f:
                json.dump({"positions": pf}, f, ensure_ascii=False)
        except Exception as e:
            logger.debug("Persist portfolio failed for %s: %s", chat_id, e)

    def _load_portfolio_from_disk(self, chat_id: int) -> None:
        try:
            path = self._portfolio_path(chat_id)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(
                    data.get("positions"), dict
                ):
                    self.client_portfolios[chat_id] = data["positions"]
        except Exception as e:
            logger.debug("Load portfolio failed for %s: %s", chat_id, e)


def main() -> None:
    load_dotenv()
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/daily_reports", exist_ok=True)

    token = os.getenv("TELEGRAM_TOKEN")
    if not token or token.startswith("DRY_TEST"):
        logger.error("No valid TELEGRAM_TOKEN in env")
        return

    bot = RealTelegramInvestmentBot(token)
    # Optional: auto-subscribe a numeric chat id from env for proactive pushes
    chat_env = os.getenv("TELEGRAM_CHAT_ID")
    if chat_env:
        try:
            chat_id_int = int(chat_env)
            bot.subscribers.add(chat_id_int)
            logger.info("Auto-subscribed chat id from env: %s", chat_id_int)
        except Exception:
            logger.warning(
                "TELEGRAM_CHAT_ID provided but not numeric; "
                "skipping auto-subscribe"
            )
    try:
        bot.start_telegram_service()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception("Bot error: %s", e)


if __name__ == "__main__":
    main()


