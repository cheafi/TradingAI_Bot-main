from __future__ import annotations
from typing import Iterable, Sequence, Dict, List, Any
from datetime import datetime, timedelta
from contextlib import suppress

try:
    # Optional UI settings integration for DRY-RUN/LIVE
    from src.ui_ext.settings_store import load_settings  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    load_settings = None  # type: ignore


def _mode_badge() -> str:
    mode = "DRY-RUN"
    with suppress(Exception):
        if load_settings:
            s = load_settings()
            if not bool(s.get("DRY_RUN", True)):
                mode = "LIVE"
    return ("DRY-RUN ✅" if mode == "DRY-RUN" else "LIVE 🟢")


def _join_lines(lines: Iterable[str]) -> str:
    valid = [
        ln for ln in lines if ln is not None and str(ln).strip() != ""
    ]  # type: ignore
    return "\n".join(valid)


def format_market_opportunity(opportunity: Any) -> str:
    """Format a market opportunity message"""
    action_emoji = "🟢 BUY" if opportunity.action == "BUY" else "🔴 SHORT"
    confidence_emoji = "🔥" if opportunity.confidence_level > 80 else "⚡" if opportunity.confidence_level > 70 else "💡"
    
    lines = [
        f"🎯 **MARKET OPPORTUNITY** {confidence_emoji}",
        f"",
        f"📊 **{opportunity.symbol}** | {action_emoji}",
        f"💰 Current Price: ${opportunity.current_price:.2f}",
        f"🎯 Target Price: ${opportunity.target_price:.2f}",
        f"🛡️ Stop Loss: ${opportunity.stop_loss:.2f}",
        f"📉 Max Drop: {opportunity.max_drop:.1f}%",
        f"🎲 Confidence: {opportunity.confidence_level:.0f}%",
        f"",
        f"📈 **Analysis:**",
        f"🔍 Reason: {opportunity.reason}",
        f"📰 Event: {opportunity.event}",
        f"🏗️ Setup: {opportunity.setup_type}",
        f"📊 Curve: {opportunity.curve_analysis}",
        f"",
        f"⏰ Time: {opportunity.timestamp.strftime('%H:%M:%S')}",
        f"",
        f"💡 **Reply with position size (e.g., 100, 500, 1000) to enter this trade!**",
        f"🔄 Reply '0' to skip this opportunity",
        f"",
        f"⚠️ {_mode_badge()}"
    ]
    
    return _join_lines(lines)


def format_portfolio_position(position: Any) -> str:
    """Format a portfolio position message"""
    action_emoji = "🟢" if position.action == "BUY" else "🔴"
    pnl_emoji = "💚" if position.pnl >= 0 else "❤️"
    status_emoji = {"ACTIVE": "🔄", "STOPPED": "🛑", "TARGET_REACHED": "🎯", "CLOSED": "✅"}.get(position.status, "❓")
    
    lines = [
        f"{status_emoji} **POSITION UPDATE**",
        f"",
        f"📊 **{position.symbol}** | {action_emoji} {position.action}",
        f"💰 Entry: ${position.entry_price:.2f}",
        f"💱 Current: ${position.current_price:.2f}",
        f"📈 Quantity: {position.quantity:.2f}",
        f"🎯 Target: ${position.target_price:.2f}",
        f"🛡️ Stop: ${position.stop_loss:.2f}",
        f"",
        f"{pnl_emoji} **P&L: ${position.pnl:.2f} ({position.pnl_percent:.1f}%)**",
        f"📊 Status: {position.status}",
        f"⏰ Entry Date: {position.entry_date.strftime('%m/%d %H:%M')}",
        f"",
        f"⚠️ {_mode_badge()}"
    ]
    
    return _join_lines(lines)


def format_portfolio_summary(summary: Dict) -> str:
    """Format portfolio summary message"""
    pnl_emoji = "💚" if summary['total_pnl'] >= 0 else "❤️"
    win_rate_emoji = "🔥" if summary['win_rate'] > 70 else "⚡" if summary['win_rate'] > 50 else "💡"
    
    lines = [
        f"📊 **PORTFOLIO SUMMARY**",
        f"",
        f"{pnl_emoji} **Total P&L: ${summary['total_pnl']:.2f}**",
        f"📈 Avg Return: {summary['avg_return']:.1f}%",
        f"🔄 Active Positions: {summary['active_positions']}",
        f"📊 Total Trades: {summary['total_trades']}",
        f"",
        f"{win_rate_emoji} **Win Rate: {summary['win_rate']:.1f}%**",
        f"✅ Wins: {summary['wins']}",
        f"❌ Losses: {summary['losses']}",
        f"",
        f"⚠️ {_mode_badge()}"
    ]
    
    return _join_lines(lines)


def format_market_sentiment(sentiment: Dict) -> str:
    """Format US market sentiment message"""
    sentiment_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(sentiment['sentiment'], "❓")
    fear_greed_emoji = {"FEAR": "😨", "GREED": "🤑", "NEUTRAL": "😐"}.get(sentiment.get('fear_greed', 'NEUTRAL'), "😐")
    
    lines = [
        f"🇺🇸 **US MARKET SENTIMENT** {sentiment_emoji}",
        f"",
        f"📊 **Overall: {sentiment['sentiment']}**",
        f"📈 Market Change: {sentiment.get('overall_change', 0):.2f}%",
        f"😱 VIX Level: {sentiment.get('vix_level', 0):.1f}",
        f"{fear_greed_emoji} Fear/Greed: {sentiment.get('fear_greed', 'NEUTRAL')}",
        f"",
        f"📊 **Major Indices:**"
    ]
    
    if 'indices' in sentiment:
        for name, data in sentiment['indices'].items():
            change_emoji = "🟢" if data['change_pct'] >= 0 else "🔴"
            lines.append(f"{change_emoji} {name}: {data['change_pct']:+.2f}%")
    
    lines.extend([
        f"",
        f"⏰ Updated: {datetime.now().strftime('%H:%M:%S')}",
        f"",
        f"⚠️ {_mode_badge()}"
    ])
    
    return _join_lines(lines)


def format_daily_market_report(
    phase: str,
    date_str: str,
    regime: str | None,
    highlights: Sequence[str],
    market_focus: Sequence[str],
    next_report: str | None,
) -> str:
    hdr = f"📊 Daily Market Report ({phase})\nDate: {date_str}"
    reg = f"Regime: {regime}" if regime else "Regime: —"
    body = []
    if highlights:
        body.append(_join_lines(highlights))
    if market_focus:
        body.extend(
            ["\nMarket Focus:", _join_lines([f"- {x}" for x in market_focus])]
        )
    footer = _join_lines([
        f"\nSystem Health: {_mode_badge()}",
        f"Next report: {next_report}" if next_report else None,
    ])
    return _join_lines([hdr, reg, *body, footer])


def format_setup_alert(
    title: str,
    trend: str | None,
    momentum: str | None,
    risk: str | None,
    ml_prob: str | None,
    action: str | None,
    confidence: str | None,
) -> str:
    lines = [f"⚡ Setup Alert: {title}"]
    if trend:
        lines.append(f"\nTrend: {trend}")
    if momentum:
        lines.append(f"Momentum: {momentum}")
    if risk:
        lines.append(f"Risk: {risk}")
    if ml_prob:
        lines.append(f"ML Prob: {ml_prob}")
    if action:
        lines.append(f"\n✅ Suggested Action: {action}")
    if confidence:
        lines.append(f"Confidence: {confidence}")
    lines.append(f"\nMode: {_mode_badge()}")
    return _join_lines(lines)


def format_portfolio_alert(lines: Sequence[str]) -> str:
    head = "🛡️ Portfolio Alert"
    body = _join_lines(lines)
    footer = f"\nMode: {_mode_badge()}"
    return _join_lines([head, body, footer])


def format_portfolio_summary(
    equity_line: str,
    exposure_line: str,
    positions: Sequence[str],
    risk_lines: Sequence[str] | None = None,
    next_review: str | None = None,
) -> str:
    head = (
        f"📂 Portfolio Summary ("
        f"{'Dry-Run' if 'DRY' in _mode_badge() else 'Live'}"
        ")"
    )
    pos = _join_lines(positions)
    risk = _join_lines(risk_lines or [])
    footer = _join_lines([
        (risk or None),
        f"Next review in {next_review}" if next_review else None,
        f"Mode: {_mode_badge()}",
    ])
    return _join_lines(
        [head, equity_line, exposure_line, "\nPositions:", pos, footer]
    )


def format_bot_status(
    exchange: str | None,
    next_jobs: Sequence[str],
    last_refresh: str | None,
) -> str:
    lines = ["⚙️ Bot Status", f"\nMode: {_mode_badge()}"]
    if exchange:
        lines.append(f"Exchange: {exchange}")
    if next_jobs:
        lines.append("Next Jobs:")
        lines.extend([f"- {j}" for j in next_jobs])
    if last_refresh:
        lines.append(f"\nLast Refresh: {last_refresh}")
    return _join_lines(lines)


def next_job_hints() -> list[str]:
    # Mirror default schedule in bot (_hourly_scheduler)
    slots = [(8, 0), (11, 30), (14, 0), (16, 30), (20, 0)]
    now = datetime.now()
    today = now.date()
    times = [
        datetime.combine(
            today,
            now.time().replace(
                hour=h, minute=m, second=0, microsecond=0
            ),
        )
        for h, m in slots
    ]
    upcoming = [t for t in times if t > now] or [
        datetime.combine(today + timedelta(days=1), times[0].time())
    ]
    return [f"{t.strftime('%H:%M')} — scan/report" for t in upcoming[:3]]
