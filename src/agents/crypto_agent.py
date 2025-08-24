"""Crypto 24/7 Dry-Run Agent

This agent scans crypto markets continuously using ccxt (public endpoints)
and emits notifications on:
- Potential buy setups (RSI/EMA cross/breakouts)
- Portfolio risk (stop loss hits or bad setups)

It performs DRY RUN only. No orders are placed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None  # type: ignore

try:
    import ta  # noqa: F401  # used for indicators
except Exception:  # pragma: no cover
    ta = None  # type: ignore


log = logging.getLogger("crypto_agent")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PORTFOLIO_DIR = DATA_DIR / "portfolios"
STATE_FILE = DATA_DIR / "agent_state.json"
REPORTS_DIR = DATA_DIR / "daily_reports"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


NotifyFn = Callable[[str], asyncio.Future]


@dataclass
class SymbolConfig:
    symbol: str
    timeframe: str = "15m"  # scanning timeframe
    limit: int = 200


class CryptoAgent:
    """Dry-run crypto scanning agent.

    - Fetches OHLCV for configured symbols
    - Computes simple signals (RSI cross up, EMA50>EMA200 cross, breakout)
    - Checks portfolio stop-loss and emits alerts
    - Stores last-signal state to avoid spamming
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        exchange_id: str = "binance",
        timeframe: str = "15m",
        polling_sec: int = 300,
        config_path: Optional[str] = None,
    ) -> None:
        self.config = self._load_yaml(config_path)
        self.symbols = (
            symbols
            or self.config.get("symbols")
            or ["BTC/USDT", "ETH/USDT"]
        )
        self.exchange_id = self.config.get("exchange", exchange_id)
        self.default_timeframe = self.config.get("timeframe", timeframe)
        self.polling_sec = max(
            60, int(self.config.get("polling_sec", polling_sec))
        )
        ind = self.config.get("indicators", {})
        self.ema_fast = int(ind.get("ema_fast", 50))
        self.ema_slow = int(ind.get("ema_slow", 200))
        self.rsi_len = int(ind.get("rsi_len", 14))
        self.rsi_oversold = float(ind.get("rsi_oversold", 30))
        self.donchian_len = int(ind.get("donchian_len", 55))
        risk = self.config.get("risk", {})
        self.default_stop_pct = float(risk.get("stop_pct", 0.03))
        self.badsetup_timeframe = str(risk.get("badsetup_timeframe", "1h"))
        notify = self.config.get("notify", {})
        self.ttl_dup = int(notify.get("duplicate_ttl_sec", 3600))
        self.ttl_bad = int(notify.get("badsetup_ttl_sec", 36000))
        self._last_state: Dict[str, Any] = self._load_state()
        self._ex = None

    def _load_yaml(self, path: Optional[str]) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        if not path:
            # default location
            path = str((ROOT / "config" / "crypto_strategy.yml"))
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    # ------------- lifecycle -------------
    async def start(self) -> None:
        self._ensure_exchange()

    async def run(self, notify: Callable[[str], asyncio.Future]) -> None:
        """Main loop. Never returns."""
        if ccxt is None:
            log.warning("ccxt is not available; crypto agent disabled")
            return
        await self.start()
        while True:
            try:
                await self._scan_symbols(notify)
                await self._check_portfolios(notify)
            except Exception as e:
                log.exception("Agent iteration failed: %s", e)
            await asyncio.sleep(self.polling_sec)

    # ------------- exchange helpers -------------
    def _ensure_exchange(self):
        if self._ex is None and ccxt is not None:
            cls = getattr(ccxt, self.exchange_id, None)
            if cls is None:  # pragma: no cover
                raise RuntimeError(f"Exchange {self.exchange_id} not found")
            self._ex = cls(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )

    async def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
        since: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        def _sync_fetch():
            # Many exchanges support `since` in milliseconds
            if since is not None:
                return self._ex.fetch_ohlcv(  # type: ignore[attr-defined]
                    symbol, timeframe=timeframe, since=since, limit=limit
                )
            return self._ex.fetch_ohlcv(  # type: ignore[attr-defined]
                symbol, timeframe=timeframe, limit=limit
            )

        try:
            rows = await asyncio.to_thread(_sync_fetch)
            if not rows:
                return None
            df = pd.DataFrame(
                rows,
                columns=[
                    "ts",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df
        except Exception as e:
            log.warning("fetch_ohlcv failed %s %s: %s", symbol, timeframe, e)
            return None

    # ------------- helpers: timeframes & dates -------------
    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        tf = tf.strip().lower()
        if tf.endswith("m"):
            return int(tf[:-1]) * 60
        if tf.endswith("h"):
            return int(tf[:-1]) * 3600
        if tf.endswith("d"):
            return int(tf[:-1]) * 86400
        # default assume minutes if pure int
        try:
            return int(tf) * 60
        except Exception:
            return 900

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        # Accept formats: YYYY-MM-DD, YYYY/MM/DD,
        # 'Aug 1 2025', '2025-08-01T00:00'
        from dateutil import parser  # type: ignore

        dt = parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    def _calc_limit_for_range(
        self,
        timeframe: str,
        start: datetime,
        end: datetime,
        margin_days: int = 35,
    ) -> Tuple[str, int]:
        # compute bars required; if too many, coarsen timeframe
        tf = timeframe
        sec = self._timeframe_to_seconds(tf)
        span = (end - start).total_seconds() + margin_days * 86400
        bars = int(span // sec) + 100

        # if bars > 1800, try coarsen to reduce
        def coarsen(t: str) -> str:
            if t.endswith("m"):
                m = int(t[:-1])
                return "1h" if m < 60 else "4h"
            if t.endswith("h"):
                h = int(t[:-1])
                return "4h" if h < 4 else "1d"
            return "1d"

        while bars > 1800 and tf != "1d":
            tf = coarsen(tf)
            sec = self._timeframe_to_seconds(tf)
            bars = int(span // sec) + 100
        return tf, min(bars, 1800)

    # ------------- signals -------------
    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        close = d["close"].astype(float)
        # EMA
        d["ema_fast"] = close.ewm(span=self.ema_fast, adjust=False).mean()
        d["ema_slow"] = close.ewm(span=self.ema_slow, adjust=False).mean()
        # RSI
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(span=self.rsi_len, adjust=False).mean()
        roll_down = down.ewm(span=self.rsi_len, adjust=False).mean()
        rs = np.where(roll_down == 0, np.nan, roll_up / roll_down)
        d["rsi"] = 100 - (100 / (1 + rs))
        # Donchian breakout
        d["donchian_high"] = d["high"].rolling(self.donchian_len).max()
        d["donchian_low"] = d["low"].rolling(self.donchian_len).min()
        return d

    def _generate_setups(self, d: pd.DataFrame) -> List[str]:
        msgs: List[str] = []
        if len(d) < max(self.ema_slow, self.donchian_len) + 5:
            return msgs
        last = d.iloc[-1]
        prev = d.iloc[-2]
        # EMA golden cross
        if (
            prev["ema_fast"] <= prev["ema_slow"]
            and last["ema_fast"] > last["ema_slow"]
        ):
            msgs.append("EMA50>EMA200 金叉")
        # RSI cross up from oversold
        if prev["rsi"] < self.rsi_oversold <= last["rsi"]:
            msgs.append("RSI14 由超賣上穿 30")
        # Breakout
        if last["close"] > last["donchian_high"]:
            msgs.append("55期唐奇安突破")
        return msgs

    async def _scan_symbols(
        self, notify: Callable[[str], asyncio.Future]
    ) -> None:
        for symbol in self.symbols:
            df = await self._fetch_ohlcv(
                symbol, self.default_timeframe, limit=250
            )
            if df is None or df.empty:
                continue
            d = self._indicators(df)
            setups = self._generate_setups(d)
            if not setups:
                continue
            state_key = f"signal:{symbol}:{self.default_timeframe}"
            last_sig = self._last_state.get(state_key)
            # Avoid duplicate alerts within same candle
            candle_ts = int(d.iloc[-1]["ts"].value // 10**9)
            if last_sig and last_sig.get("ts") == candle_ts:
                continue
            text = (
                f"🚀 加密幣買進觀察: {symbol} @ {self.default_timeframe}\n"
                f"收盤價: {d.iloc[-1]['close']:.2f}\n"
                f"訊號: {', '.join(setups)}\n"
                f"時間: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            )
            await notify(text)
            self._last_state[state_key] = {"ts": candle_ts, "setups": setups}
            self._persist_state()
            ml_score = self._quick_ml_score(d)
            self._write_opportunities(
                symbol, d.iloc[-1]['close'], setups, ml_score=ml_score
            )

    # ------------- portfolio risk -------------
    def _iter_portfolios(self) -> List[Path]:
        return list(PORTFOLIO_DIR.glob("*.json"))

    def _load_portfolio(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    async def _latest_price(self, symbol: str) -> Optional[float]:
        def _fetch_ticker():
            return self._ex.fetch_ticker(symbol)  # type: ignore[attr-defined]

        try:
            t = await asyncio.to_thread(_fetch_ticker)
            return float(t.get("last") or t.get("close") or 0) or None
        except Exception:
            return None

    async def _check_portfolios(
        self, notify: Callable[[str], asyncio.Future]
    ) -> None:
        files = self._iter_portfolios()
        for f in files:
            pf = self._load_portfolio(f)
            if not pf or not isinstance(pf, dict):
                continue
            chat_id = f.stem
            positions = pf.get("positions", {})
            if not positions:
                continue
            for sym, pos in positions.items():
                try:
                    qty = float(pos.get("qty", 0))
                    if qty <= 0:
                        continue
                    entry = float(pos.get("cost", 0))
                    stop = (
                        float(pos.get("stop", 0)) if pos.get("stop") else None
                    )
                    last = await self._latest_price(sym)
                    if last is None:
                        continue
                    # derive default stop if missing
                    if not stop and entry:
                        stop = entry * (1 - self.default_stop_pct)
                    # stop loss hit
                    if stop and last <= stop:
                        key = f"stophit:{chat_id}:{sym}:{int(last)}"
                        if self._suppress_dup(key, ttl_sec=self.ttl_dup):
                            continue
                        await notify(
                            (
                                f"[chat:{chat_id}] "
                                f"⛔️ 風控：{sym} 觸發停損 {stop:.4f}\n"
                                f"現價 {last:.4f}，建議減倉或退出\n"
                                f"持倉數量: {qty} 成本: {entry}"
                            )
                        )
                    # bad setup example: price below EMA200 on 1h
                    df = await self._fetch_ohlcv(
                        sym, self.badsetup_timeframe, limit=250
                    )
                    if df is not None and not df.empty:
                        d = self._indicators(df)
                        if float(d.iloc[-1]["close"]) < float(
                            d.iloc[-1]["ema_slow"]
                        ):
                            key = (
                                f"badsetup:{chat_id}:{sym}:"
                                f"{int(df.iloc[-1]['ts'].value//1e9)}"
                            )
                            if self._suppress_dup(key, ttl_sec=self.ttl_bad):
                                continue
                            await notify(
                                (
                                    f"[chat:{chat_id}] "
                                    f"⚠️ 風險：{sym} 跌破 1h EMA200，趨勢轉弱\n"
                                    f"現價 {last:.4f}，請評估是否減倉/出場"
                                )
                            )
                except Exception as e:
                    log.debug("check portfolio error for %s: %s", sym, e)

    # ------------- state -------------
    def _load_state(self) -> Dict[str, Any]:
        with suppress(Exception):
            if STATE_FILE.exists():
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return {}

    def _persist_state(self) -> None:
        with suppress(Exception):
            STATE_FILE.write_text(
                json.dumps(self._last_state), encoding="utf-8"
            )

    _recent_keys: Dict[str, float] = {}

    def _suppress_dup(self, key: str, ttl_sec: int = 3600) -> bool:
        """Return True if duplicate and should suppress."""
        now = datetime.now(timezone.utc).timestamp()
        last = self._recent_keys.get(key)
        if last and (now - last) < ttl_sec:
            return True
        self._recent_keys[key] = now
        return False

    # ------------- outputs -------------
    def _write_opportunities(
        self,
        symbol: str,
        price: float,
        setups: List[str],
        ml_score: Optional[float] = None,
    ) -> None:
        """Persist a lightweight opportunities.json for UI/TG to consume."""
        path = REPORTS_DIR / "opportunities.json"
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        data: Dict[str, Any] = {"items": []}
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {"items": []}
        # lightweight ML score (optional): momentum slope over last 10 closes
        item = {
            "symbol": symbol,
            "price": price,
            "view": ", ".join(setups),
            "score": len(setups),
            "ml_score": ml_score,
            "time": now,
        }
        data.setdefault("items", [])
        data["items"].insert(0, item)
        # keep last 50
        data["items"] = data["items"][:50]
        with suppress(Exception):
            path.write_text(json.dumps(data), encoding="utf-8")

    def _quick_ml_score(self, d: pd.DataFrame) -> Optional[float]:
        try:
            closes = d["close"].astype(float).values
            n = min(120, len(closes))
            if n < 20:
                return None
            y = closes[-n:]
            x = np.arange(n)
            # slope via polyfit degree=1
            slope, _ = np.polyfit(x, y, 1)
            last = float(y[-1])
            # normalize slope by price and window size
            norm = (slope / max(last, 1e-8)) * 1000.0
            # clip to [-5, 5] then map to 0..100
            norm = float(np.clip(norm, -5.0, 5.0))
            score = (norm + 5.0) * 10.0
            return round(score, 2)
        except Exception:
            return None

    # ------------- backtest -------------
    async def backtest(
        self, symbol: str, timeframe: Optional[str] = None, limit: int = 1500
    ) -> Dict[str, Any]:
        """Simple event backtest counting signals & naive returns."""
        tf = timeframe or self.default_timeframe
        df = await self._fetch_ohlcv(symbol, tf, limit=limit)
        if df is None or df.empty:
            return {"symbol": symbol, "timeframe": tf, "signals": 0, "ret": 0}
        d = self._indicators(df)
        sigs: List[int] = []
        for i in range(2, len(d)):
            block = d.iloc[: i + 1]
            ev = self._generate_setups(block)
            if ev:
                sigs.append(i)
        # naive: +1% per signal next candle if close up else -1%
        ret = 0.0
        for i in sigs:
            if i + 1 < len(d) and d.iloc[i + 1]["close"] > d.iloc[i]["close"]:
                ret += 0.01
            else:
                ret -= 0.01
        return {
            "symbol": symbol,
            "timeframe": tf,
            "signals": len(sigs),
            "ret": ret,
        }

    # ------------- historical advice -------------
    async def advise_on_date(
        self,
        symbol: str,
        date_str: str,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate advice on a given date and evaluate +30d forward return.

        Returns keys: symbol, timeframe, date, setups, entry_price,
        fwd_date, fwd_price, fwd_ret.
        """
        dt = self._parse_date(date_str)
        tf = timeframe or self.default_timeframe
        start = dt - pd.Timedelta(days=60)
        end = dt + pd.Timedelta(days=35)
        tf2, limit = self._calc_limit_for_range(tf, start, end)
        df = await self._fetch_ohlcv(
            symbol, tf2, limit=limit, since=int(start.timestamp() * 1000)
        )
        if df is None or df.empty:
            return {"symbol": symbol, "timeframe": tf2, "error": "no data"}
        d = self._indicators(df)
        # find index at/after dt
        idx = d.index[d["ts"] >= pd.Timestamp(dt)].tolist()
        if not idx:
            return {"symbol": symbol, "timeframe": tf2, "error": "no bar"}
        i = idx[0]
        entry_block = d.iloc[: i + 1]
        setups = self._generate_setups(entry_block)
        entry = float(d.iloc[i]["close"])
        # forward +30 days
        target_ts = pd.Timestamp(dt) + pd.Timedelta(days=30)
        jidx = d.index[d["ts"] >= target_ts].tolist()
        if not jidx:
            return {
                "symbol": symbol,
                "timeframe": tf2,
                "date": dt.isoformat(),
                "setups": setups,
                "entry_price": entry,
                "error": "+30d data not available",
            }
        j = jidx[0]
        fwd = float(d.iloc[j]["close"])
        fwd_ret = (fwd / entry) - 1.0
        return {
            "symbol": symbol,
            "timeframe": tf2,
            "date": dt.strftime("%Y-%m-%d"),
            "setups": setups,
            "entry_price": entry,
            "fwd_date": d.iloc[j]["ts"].strftime("%Y-%m-%d"),
            "fwd_price": fwd,
            "fwd_ret": fwd_ret,
        }

    async def simulate_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Simulate advice over a range and evaluate +30d returns
        for each signal.
        """
        sdt = self._parse_date(start_date)
        edt = self._parse_date(end_date)
        if edt <= sdt:
            return {"error": "end_date must be after start_date"}
        tf = timeframe or self.default_timeframe
        start = sdt - pd.Timedelta(days=60)
        end = edt + pd.Timedelta(days=35)
        tf2, limit = self._calc_limit_for_range(tf, start, end)
        df = await self._fetch_ohlcv(
            symbol, tf2, limit=limit, since=int(start.timestamp() * 1000)
        )
        if df is None or df.empty:
            return {"symbol": symbol, "timeframe": tf2, "error": "no data"}
        d = self._indicators(df)
        mask = (d["ts"] >= pd.Timestamp(sdt)) & (d["ts"] <= pd.Timestamp(edt))
        events: List[Dict[str, Any]] = []
        for i in d.index[mask]:
            block = d.iloc[: i + 1]
            setups = self._generate_setups(block)
            if not setups:
                continue
            entry = float(d.iloc[i]["close"])
            t0 = d.iloc[i]["ts"]
            target_ts = t0 + pd.Timedelta(days=30)
            jidx = d.index[d["ts"] >= target_ts].tolist()
            if not jidx:
                continue
            j = jidx[0]
            fwd = float(d.iloc[j]["close"])
            fwd_ret = (fwd / entry) - 1.0
            events.append(
                {
                    "date": t0.strftime("%Y-%m-%d"),
                    "setups": setups,
                    "entry": entry,
                    "fwd_date": d.iloc[j]["ts"].strftime("%Y-%m-%d"),
                    "fwd": fwd,
                    "ret": fwd_ret,
                }
            )
        if not events:
            return {
                "symbol": symbol,
                "timeframe": tf2,
                "start": sdt.strftime("%Y-%m-%d"),
                "end": edt.strftime("%Y-%m-%d"),
                "events": [],
                "count": 0,
                "avg_ret": 0.0,
                "win_rate": 0.0,
            }
        rets = [e["ret"] for e in events]
        wins = sum(r > 0 for r in rets)
        return {
            "symbol": symbol,
            "timeframe": tf2,
            "start": sdt.strftime("%Y-%m-%d"),
            "end": edt.strftime("%Y-%m-%d"),
            "events": events[:20],  # cap sample size in reply
            "count": len(events),
            "avg_ret": float(np.mean(rets)),
            "win_rate": round(wins / len(events), 4),
            "best": float(np.max(rets)),
            "worst": float(np.min(rets)),
        }


__all__ = ["CryptoAgent", "SymbolConfig"]
