from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
from io import BytesIO
from contextlib import suppress


DAILY_OPP_PATH = os.path.join("data", "daily_reports", "opportunities.json")
LOG_OUTLOOK = os.path.join("logs", "latest_market_outlook.txt")
LOG_RISK = os.path.join("logs", "latest_risk_alerts.txt")


def _safe_read_text(path: str, default: str = "") -> str:
    with suppress(Exception):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
    return default


def _safe_read_json(path: str) -> Dict[str, Any]:
    with suppress(Exception):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def read_opportunities() -> List[Dict[str, Any]]:
    data = _safe_read_json(DAILY_OPP_PATH)
    out: List[Dict[str, Any]] = []
    if data.get("all_opportunities") is not None:
        out.extend(
            [
                {
                    "symbol": op.get("symbol"),
                    "price": op.get("current_price"),
                    "score": op.get("score"),
                    "ml_score": None,
                    "reasons": op.get("reasons", []),
                    "recommendation": op.get("recommendation"),
                    "target_price": op.get("target_price"),
                    "stop_loss": op.get("stop_loss"),
                    # include timeframe if producer provides it
                    "timeframe": (
                        op.get("timeframe")
                        or op.get("tf")
                        or op.get("interval")
                    ),
                    "time": data.get("date"),
                }
                for op in data.get("all_opportunities", [])
            ]
        )
        return out

    agent_paths = [
        DAILY_OPP_PATH,
        os.path.join("data", "opportunities.json"),
        os.path.join("reports", "opportunities.json"),
    ]
    for p in agent_paths:
        d2 = _safe_read_json(p)
        items = d2.get("items")
        if isinstance(items, list):
            out.extend(
                [
                    {
                        "symbol": it.get("symbol"),
                        "price": it.get("price"),
                        "score": it.get("score"),
                        "ml_score": it.get("ml_score"),
                        "reasons": [it.get("view")] if it.get("view") else [],
                        # infer recommendation from view if it looks like one
                        "recommendation": (
                            str(it.get("view")).upper()
                            if isinstance(it.get("view"), str)
                            else None
                        ),
                        "target_price": None,
                        "stop_loss": None,
                        "timeframe": (
                            it.get("timeframe")
                            or it.get("tf")
                            or it.get("interval")
                        ),
                        "time": it.get("time"),
                    }
                    for it in items
                ]
            )
            break
    return out


def summarize_market_report(max_words: int = 80) -> str:
    raw = _safe_read_text(LOG_OUTLOOK)
    if not raw:
        return "No latest market outlook found."
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    text = " ".join(lines[:10])
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "…"
    return text


def recent_risk_alerts(n: int = 5) -> List[str]:
    raw = _safe_read_text(LOG_RISK)
    if not raw:
        return []
    lines = [ln.strip(" \t-•") for ln in raw.splitlines() if ln.strip()]
    alerts: List[str] = []
    for ln in lines:
        lower = ln.lower()
        if lower.startswith(("1.", "2.", "3.", "status")):
            continue
        keys = ["warning", "elevated", "alert", "risk", "volatility"]
        if any(k in lower for k in keys):
            alerts.append(ln)
    if not alerts:
        alerts = [ln for ln in lines if len(ln) > 6][-n:]
    return alerts[-n:]


def focus_from_opportunities(limit: int = 3) -> List[Tuple[str, str]]:
    ops = read_opportunities()
    if not ops:
        return []

    def _key(x: Dict[str, Any]):
        return (float(x.get("score") or 0), float(x.get("ml_score") or 0.0))

    ops_sorted = sorted(ops, key=_key, reverse=True)
    out: List[Tuple[str, str]] = []
    for op in ops_sorted[: limit * 2]:
        sym = op.get("symbol", "?")
        reasons = op.get("reasons") or []
        top_reason = reasons[0] if reasons else "Momentum improving"
        rec = op.get("recommendation") or (
            "BUY" if (op.get("score") or 0) >= 5 else "WATCH"
        )
        headline = f"{sym} — {rec} candidate"
        why = f"Why: {top_reason}"
        out.append((headline, why))
        if len(out) >= limit:
            break
    return out


def make_pdf_from_text(title: str, paragraphs: List[str]) -> Optional[bytes]:
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.lib.units import cm  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
    except Exception:
        return None
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, y, title)
    y -= 1 * cm
    c.setFont("Helvetica", 11)
    for para in paragraphs:
        for line in _wrap_text(para, 90):
            c.drawString(2 * cm, y, line)
            y -= 0.6 * cm
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 11)
        y -= 0.4 * cm
    c.showPage()
    c.save()
    return buffer.getvalue()


def _wrap_text(text: str, width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    line: List[str] = []
    for w in words:
        if sum(len(x) for x in line) + len(line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return lines
