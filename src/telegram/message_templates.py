from __future__ import annotations
from typing import Iterable, Sequence
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
