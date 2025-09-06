"""
Enhanced Telegram bot with advanced features:
- Real-time trading alerts with charts
- Voice messages and commands
- Portfolio analysis
- AI-powered suggestions
"""
from __future__ import annotations
import asyncio
import logging
import io
import time
from datetime import datetime
from typing import Dict, List, Optional
from pprint import pformat

import pandas as pd
try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.express as px  # type: ignore
    _PLOTLY_OK = True
except Exception:  # pragma: no cover
    _PLOTLY_OK = False
import numpy as np
import yfinance as yf  # Added: used in _generate_price_chart
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
from src.strategies.scalping import enrich as strat_enrich, signals as strat_signals
from src.utils.data import synthetic_ohlcv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Enhanced Telegram trading bot."""
    
    def __init__(self, token: str):
        self.token = token
        self.application = Application.builder().token(token).build()
        self.portfolio_value = 100000.0
        self.positions = {}
        self.alerts_enabled = True
        self.start_ts = time.time()
        
        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register all command and message handlers."""
        # Debug tap first (logs every update)
        self.application.add_handler(MessageHandler(filters.ALL, self.debug_tap), group=0)
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("portfolio", self.portfolio))
        self.application.add_handler(CommandHandler("balance", self.balance))
        self.application.add_handler(CommandHandler("profits", self.profits))
        self.application.add_handler(CommandHandler("suggest", self.suggest))
        self.application.add_handler(CommandHandler("chart", self.chart))
        self.application.add_handler(CommandHandler("alerts", self.toggle_alerts))
        self.application.add_handler(CommandHandler("risk", self.risk_status))
        self.application.add_handler(CommandHandler("optimize", self.optimize))
        self.application.add_handler(CommandHandler("voice", self.voice_summary))
        self.application.add_handler(CommandHandler("id", self.show_chat_id))
        self.application.add_handler(CommandHandler("ping", self.ping))
        self.application.add_handler(CommandHandler("ping2", self.ping))
        self.application.add_handler(CommandHandler("status", self.status))
        self.application.add_handler(CommandHandler("edge", self.edge))
        self.application.add_handler(CommandHandler("raw", self.raw))
        # Unknown commands last
        self.application.add_handler(MessageHandler(filters.COMMAND, self.unknown_command))
        # Callback handlers
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        # Text messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        # Error handler
        self.application.add_error_handler(self.on_error)
    
    async def _safe_reply(self, update: Update, text: str, **kwargs):
        msg = update.effective_message
        if not msg:
            return
        try:
            await msg.reply_text(text, **kwargs)
        except Exception as e:  # pragma: no cover
            logger.warning("Reply failed: %s", e)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Welcome message with inline keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("📊 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton("💰 Balance", callback_data="balance"),
            ],
            [
                InlineKeyboardButton("📈 Charts", callback_data="chart"),
                InlineKeyboardButton("🤖 AI Suggest", callback_data="suggest"),
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
                InlineKeyboardButton("❓ Help", callback_data="help"),
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_text = """
🚀 *Welcome to TradingAI Pro Bot!*

Your AI-powered trading assistant is ready to help you:
• 📊 Monitor your portfolio in real-time
• 💰 Track profits and losses
• 📈 Generate trading charts
• 🤖 Get AI-powered trade suggestions
• 🔔 Receive smart alerts
• 🎙️ Voice summaries and commands

Choose an option below or type a command:
        """
        
        await self._safe_reply(update, welcome_text, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help information."""
        help_text = """
📋 *Available Commands:*

*Portfolio & Trading:*
• `/portfolio` - View current positions
• `/balance` - Check account balance
• `/profits` - Show P&L summary
• `/suggest <SYMBOL>` - Get AI trading suggestion

*Analysis & Charts:*
• `/chart <SYMBOL>` - Generate price chart
• `/risk` - Risk management status
• `/optimize` - Run strategy optimization

*Settings & Alerts:*
• `/alerts` - Toggle trading alerts
• `/voice` - Get voice summary

*Quick Actions:*
• Type any stock symbol (e.g., AAPL) for instant analysis
• Send voice messages for voice commands

💡 *Pro Tips:*
• Use /suggest AAPL for detailed AI analysis
• Charts include technical indicators
• Voice messages supported for hands-free operation
        """
        
        await self._safe_reply(update, help_text, parse_mode='Markdown')
    
    async def portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show portfolio summary with chart."""
        # Generate mock portfolio data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC']
        quantities = [10, 5, 3, 8, 0.5]
        prices = [175.0, 380.0, 140.0, 250.0, 45000.0]
        
        portfolio_text = "📊 *Portfolio Summary*\n\n"
        total_value = 0
        
        for symbol, qty, price in zip(symbols, quantities, prices):
            value = qty * price
            total_value += value
            portfolio_text += f"• {symbol}: {qty} shares @ ${price:.2f} = ${value:,.2f}\n"
        
        portfolio_text += f"\n💰 *Total Portfolio Value: ${total_value:,.2f}*"
        portfolio_text += f"\n📈 *Daily Change: +${(total_value * 0.025):,.2f} (+2.5%)*"
        
        # Generate portfolio chart
        chart_buffer = self._generate_portfolio_chart(symbols, quantities, prices)
        
        msg = update.effective_message
        if msg:
            try:
                await msg.reply_photo(photo=chart_buffer, caption=portfolio_text, parse_mode='Markdown')
            except Exception as e:  # pragma: no cover
                logger.warning("Photo send failed: %s; fallback to text", e)
                await self._safe_reply(update, portfolio_text, parse_mode='Markdown')
    
    async def balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show account balance and metrics."""
        balance_text = f"""
💰 *Account Balance*

💵 *Available Cash:* ${self.portfolio_value * 0.2:,.2f}
📊 *Invested:* ${self.portfolio_value * 0.8:,.2f}
💼 *Total Portfolio:* ${self.portfolio_value:,.2f}

📈 *Performance Metrics:*
• Total Return: +12.4%
• Sharpe Ratio: 2.1
• Max Drawdown: -3.2%
• Win Rate: 68%

🔄 *Last Updated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        await self._safe_reply(update, balance_text, parse_mode='Markdown')
    
    async def profits(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show profit/loss analysis."""
        # Generate mock P&L data
        today_pnl = 1250.50
        week_pnl = 3420.75
        month_pnl = 8750.25
        
        profits_text = f"""
📊 *Profit & Loss Analysis*

🔥 *Today:* ${today_pnl:+,.2f} (+2.1%)
📅 *This Week:* ${week_pnl:+,.2f} (+3.8%)
📅 *This Month:* ${month_pnl:+,.2f} (+8.2%)

🏆 *Best Performers:*
• AAPL: +$1,234 (+15.2%)
• MSFT: +$987 (+12.8%)
• GOOGL: +$654 (+8.9%)

📉 *Worst Performers:*
• TSLA: -$123 (-2.1%)

💡 *AI Insight:* Your portfolio is outperforming the market by 4.2%
        """
        
        await self._safe_reply(update, profits_text, parse_mode='Markdown')
    
    async def suggest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Provide AI-powered trading suggestions."""
        # Get symbol from command args
        symbol = "AAPL"  # Default
        if context.args:
            symbol = context.args[0].upper()
        
        # Generate AI suggestion (mock)
        suggestion_text = f"""
🤖 *AI Trading Suggestion for {symbol}*

📊 *Current Analysis:*
• Price: $175.50 (+1.2%)
• RSI: 65 (Neutral)
• MACD: Bullish crossover
• Volume: Above average

🎯 *Recommendation: STRONG BUY* ⭐⭐⭐⭐⭐

💡 *Reasoning:*
• Technical indicators show bullish momentum
• Earnings beat expected by 12%
• Sector rotation favoring tech stocks
• ML model confidence: 78%

📈 *Price Targets:*
• Short-term (1-2 weeks): $185
• Medium-term (1-3 months): $195
• Stop loss: $165

⚠️ *Risk Level:* Medium
💰 *Suggested Position Size:* 5% of portfolio

🔔 *Next catalyst:* Product announcement in 2 weeks
        """
        
        keyboard = [
            [
                InlineKeyboardButton("📈 View Chart", callback_data=f"chart_{symbol}"),
                InlineKeyboardButton("💰 Execute Trade", callback_data=f"trade_{symbol}"),
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await self._safe_reply(update, suggestion_text, parse_mode='Markdown', reply_markup=reply_markup)
    
    def _generate_portfolio_chart(self, symbols, quantities, prices):
        """Generate portfolio allocation pie chart."""
        values = [q * p for q, p in zip(quantities, prices)]
        
        fig = px.pie(
            values=values,
            names=symbols,
            title="Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            width=800,
            height=600,
            font=dict(size=14)
        )
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        fig.write_image(img_buffer, format='png')
        img_buffer.seek(0)
        return img_buffer
    
    def _generate_price_chart(self, symbol):
        """Generate price chart with advanced technical indicators.
        Falls back to None if plotly not available."""
        if not _PLOTLY_OK:
            return None
        try:
            df = yf.download(symbol, period="6mo", interval="1d", progress=False)
            prices = df['Close']
            dates = df.index
        except Exception:
            dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
            prices = 100 + np.cumsum(np.random.randn(60) * 0.02)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price', line=dict(color='blue', width=2)))
        ma20 = pd.Series(prices).rolling(20).mean()
        fig.add_trace(go.Scatter(x=dates, y=ma20, mode='lines', name='MA20', line=dict(color='orange', width=1)))
        rsi = pd.Series(prices).rolling(14).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() else 0)
        fig.add_trace(go.Scatter(x=dates, y=rsi, mode='lines', name='RSI', line=dict(color='purple', width=1, dash='dot')))
        macd = pd.Series(prices).ewm(span=12).mean() - pd.Series(prices).ewm(span=26).mean()
        fig.add_trace(go.Scatter(x=dates, y=macd, mode='lines', name='MACD', line=dict(color='red', width=1, dash='dash')))
        bb_mean = pd.Series(prices).rolling(20).mean()
        bb_std = pd.Series(prices).rolling(20).std()
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std
        fig.add_trace(go.Scatter(x=dates, y=bb_upper, mode='lines', name='BB Upper', line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=dates, y=bb_lower, mode='lines', name='BB Lower', line=dict(color='gray', width=1)))
        fig.update_layout(title=f"{symbol} Price Chart (Advanced Indicators)", xaxis_title="Date", yaxis_title="Price ($)", width=900, height=600, showlegend=True)
        img_buffer = io.BytesIO()
        fig.write_image(img_buffer, format='png')
        img_buffer.seek(0)
        return img_buffer
    
    async def chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate and send trading chart (optional dependency)."""
        symbol = "AAPL"
        if context.args:
            symbol = context.args[0].upper()
        chart_buffer = self._generate_price_chart(symbol)
        if chart_buffer is None:
            await self._safe_reply(update, "Charting dependencies not installed. Install plotly + kaleido for charts.")
            return
        msg = update.effective_message
        if msg:
            try:
                await msg.reply_photo(photo=chart_buffer, caption=f"📈 *{symbol} Price Chart with Technical Indicators*", parse_mode='Markdown')
            except Exception:
                await self._safe_reply(update, f"📈 {symbol} chart unavailable.")

    async def voice_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate voice summary using text-to-speech."""
        summary_text = f"""
        Portfolio update: Your total portfolio value is ${self.portfolio_value:,.0f} dollars.
        Today's performance shows a gain of 2.1 percent.
        Your top performer is Apple, up 15.2 percent.
        Risk levels are within normal parameters.
        The AI suggests considering a buy position in Microsoft.
        Overall portfolio health is excellent.
        """
        
        await self._safe_reply(update, f"🎙️ *Voice Summary*\n\n{summary_text}", parse_mode='Markdown')

    async def show_chat_id(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id if update.effective_chat else 'UNKNOWN'
        await self._safe_reply(update, f"Chat ID: {cid}")

    async def ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._safe_reply(update, "pong ✅")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uptime = int(time.time() - self.start_ts)
        hrs = uptime // 3600
        mins = (uptime % 3600) // 60
        txt = (
            "📊 *Bot Status*\n\n"
            f"Uptime: {hrs}h {mins}m\n"
            f"Alerts: {'ON' if self.alerts_enabled else 'OFF'}\n"
            f"PortfolioVal: ${self.portfolio_value:,.0f}\n"
            f"Version: 0.1.0\n"
            "Health: OK"
        )
        await self._safe_reply(update, txt, parse_mode='Markdown')

    async def risk_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Return simple mock risk metrics."""
        txt = (
            "🛡️ *Risk Status*\n\n"
            "• VaR(95%): 2.3%\n"
            "• Max Drawdown (30d): 4.1%\n"
            "• Leverage: 1.2x\n"
            f"• Alerts: {'On' if self.alerts_enabled else 'Off'}"
        )
        await self._safe_reply(update, txt, parse_mode='Markdown')

    async def optimize(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._safe_reply(update, "🔧 Optimizing parameters ...")
        await self._safe_reply(update, "✅ Optimization complete!\n• Sharpe: +0.12\n• Max DD: -0.8%\n• Win Rate: +5%")
    
    async def edge(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show top strategy signals (mock expected alpha ranking)."""
        try:
            df = synthetic_ohlcv("BTC/USDT", limit=300)
            df.columns = [c.lower() for c in df.columns]
            enriched = strat_enrich(df)
            sigs = strat_signals(enriched)
            recent = enriched.tail(30).copy()
            recent["signal"] = sigs.tail(30)
            recent = recent[recent["signal"]]
            if recent.empty:
                await self._safe_reply(update, "No active long signals.")
                return
            # Mock expected alpha (bps): scaled recent momentum
            recent["exp_alpha_bps"] = (recent["close"].pct_change().fillna(0).rolling(3).mean() * 10000).fillna(0)
            recent = recent.sort_values("exp_alpha_bps", ascending=False).head(5)
            lines = ["📌 *Top Signals* (mock)"]
            for ts, row in recent.iterrows():
                lines.append(f"• {ts.strftime('%H:%M')} close={row['close']:.2f} α≈{row['exp_alpha_bps']:.0f}bps")
            await self._safe_reply(update, "\n".join(lines), parse_mode='Markdown')
        except Exception as e:  # pragma: no cover
            await self._safe_reply(update, f"Edge error: {e}")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button presses."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "portfolio":
            await self.portfolio(query, context)
        elif query.data == "balance":
            await self.balance(query, context)
        elif query.data.startswith("chart_"):
            symbol = query.data.split("_")[1]
            context.args = [symbol]
            await self.chart(query, context)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle general text messages."""
        message_text = update.message.text.upper()
        
        # Check if message looks like a stock symbol
        if len(message_text) <= 5 and message_text.isalpha():
            context.args = [message_text]
            await self.suggest(update, context)
        else:
            await self._safe_reply(update, "👋 Hi! Send /help to see available commands or type a stock symbol (e.g., AAPL) for analysis.")
    
    async def debug_tap(self, update: Update, context: ContextTypes.DEFAULT_TYPE):  # pragma: no cover
        """Log raw update payload (truncated)."""
        try:
            data = update.to_dict() if hasattr(update, 'to_dict') else str(update)
            txt = pformat(data)[:800]
            logger.debug("UPDATE DEBUG: %s", txt)
        except Exception as e:
            logger.debug("debug_tap failed: %s", e)

    async def raw(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.effective_message.text if update.effective_message else ''
        await self._safe_reply(update, f"RAW: {msg[:300]}")

    async def unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._safe_reply(update, "Unknown command. Try /help")

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):  # pragma: no cover
        logger.exception("Handler error: %s", context.error)
        try:
            if hasattr(context, 'bot') and isinstance(update, Update):
                msg = update.effective_message
                if msg:
                    await msg.reply_text("⚠️ Internal error. Check logs.")
        except Exception:
            pass
    
    def run(self):
        """Start the bot."""
        logger.info("Starting Enhanced Trading Bot...")
        self.application.run_polling()


# Example usage
if __name__ == "__main__":
    BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Replace with actual token
    
    bot = TradingBot(BOT_TOKEN)
    bot.run()
