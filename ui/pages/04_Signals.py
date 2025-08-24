import os
import json
import re
import streamlit as st
import pandas as pd
from src.ui_ext.data_sources import read_opportunities, make_pdf_from_text
from src.ui_ext.settings_store import send_telegram_message
from src.ui_ext.ui_helpers import render_mode_pill

st.set_page_config(page_title="Signals", layout="wide")
st.title("Signals")
render_mode_pill()

universe = st.radio(
    "Universe",
    options=["Crypto", "US Stocks"],
    horizontal=True,
    key="signals_universe",
)

st.subheader("Filters")
f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    symbol = st.text_input("Symbol contains", "")
with f2:
    timeframe = st.selectbox("Timeframe", ["Any", "15m", "1h", "4h", "1d"])
with f3:
    min_conf = st.slider("Min Score", 0, 10, 3, 1)
with f4:
    regime = st.selectbox(
        "Regime", ["Any", "Risk-On", "Risk-Off"]
    )  # visual only
with f5:
    show_only_buy = st.checkbox(
        "Only BUY (if recommendation available)", value=False
    )

g1, g2 = st.columns([3, 1])
with g1:
    reason_kw = st.text_input(
        "Reason keywords (comma-separated, OR match)",
        "",
        placeholder="e.g. breakout, RSI>60, momentum",
        help=(
            "Case-insensitive. Separate multiple keywords by commas. "
            "A row matches if ANY keyword appears in the Reason."
        ),
    )
with g2:
    only_new = st.checkbox("Only New since last snapshot", value=False)

st.subheader("Active Setups")
ops = read_opportunities()
if universe == "US Stocks":
    ops = [o for o in ops if "/" not in str(o.get("symbol", ""))]
else:
    ops = [o for o in ops if "/" in str(o.get("symbol", ""))]
if ops:
    # build dataframe
    def conf_label(score: int | float | None) -> str:
        s = float(score or 0)
        if s >= 7:
            return f"High ({s:.0f})"
        elif s >= 4:
            return f"Medium ({s:.0f})"
        else:
            return f"Low ({s:.0f})"

    def _rr(price, target, stop) -> float | None:
        try:
            p = float(price)
            t = float(target)
            s = float(stop)
            if p and s and p > s:
                return (t - p) / (p - s)
        except Exception:
            return None
        return None

    def _dist(a, b) -> float | None:
        try:
            return float(a) - float(b)
        except Exception:
            return None

    # load last snapshot for change flags
    cache_dir = os.path.join("data", "ui_cache")
    os.makedirs(cache_dir, exist_ok=True)
    snap_path = os.path.join(cache_dir, "signals_last.json")
    last: dict[str, float] = {}
    try:
        if os.path.exists(snap_path):
            with open(snap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    last = {str(k): float(v) for k, v in data.items()}
    except Exception:
        last = {}

    rows = []
    for o in ops:
        sym = o.get("symbol")
        tf = o.get("timeframe")
        score = o.get("score")
        key = f"{sym}:{tf or ''}"
        prev = last.get(key)
        chg = ""
        try:
            s_val = float(score or 0)
            if prev is None:
                chg = "New"
            else:
                if s_val > prev:
                    chg = "Up"
                elif s_val < prev:
                    chg = "Down"
        except Exception:
            chg = ""

        price = o.get("price")
        target = o.get("target_price")
        stop = o.get("stop_loss")
        rows.append(
            {
                "Symbol": sym,
                "Score": score,
                "Confidence": conf_label(score),
                "Reason": (o.get("reasons") or [""])[0],
                "Price": price,
                "Target": target,
                "Stop": stop,
                "RR": _rr(price, target, stop),
                "+ToTarget": _dist(target, price),
                "-ToStop": _dist(price, stop),
                "ML": o.get("ml_score"),
                "TF": tf,
                "Rec": o.get("recommendation"),
                "Change": chg,
            }
        )
    df = pd.DataFrame(rows)
    # optional: pick reasons from suggestions (top occurrences)
    try:
        suggestions = (
            df["Reason"].dropna().astype(str).value_counts().head(20).index
        ).tolist()
    except Exception:
        suggestions = []
    picked_reasons = st.multiselect(
        "Pick reason keywords (OR)",
        options=suggestions,
        help=(
            "Choose one or more items. Rows matching ANY selected "
            "keyword will be shown."
        ),
    )
    # filters
    if symbol:
        df = df[df["Symbol"].astype(str).str.contains(symbol, case=False)]
    if timeframe != "Any":
        df = df[
            df["TF"].fillna("").astype(str).str.lower()
            == timeframe.lower()
        ]
    if show_only_buy and "Rec" in df.columns:
        df = df[
            df["Rec"].fillna("").astype(str).str.contains("BUY", case=False)
        ]
    df = df[df["Score"].fillna(0) >= min_conf]
    if picked_reasons:
        pattern = "|".join(re.escape(x) for x in picked_reasons if x)
        df = df[
            df["Reason"].astype(str).str.contains(pattern, case=False)
        ]
    elif reason_kw.strip():
        kws = [x.strip() for x in reason_kw.split(",") if x.strip()]
        if kws:
            pattern = "|".join(re.escape(x) for x in kws)
            df = df[
                df["Reason"].astype(str).str.contains(pattern, case=False)
            ]
    if only_new and "Change" in df.columns:
        df = df[df["Change"] == "New"]
    # sorting and pagination
    sort_col = st.selectbox(
        "Sort by",
        ["Score", "RR", "+ToTarget", "-ToStop", "Symbol", "Change"],
        index=0,
    )
    ascending = st.checkbox("Ascending", value=False)

    if sort_col == "Change":
        order = {"New": 0, "Up": 1, "Down": 2, "": 3}
        df["_ord"] = df["Change"].map(order).fillna(3)
        df = df.sort_values(["_ord", "Score"], ascending=[True, False])
        df = df.drop(columns=["_ord"])  # cleanup helper col
    else:
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")

    page_size = st.number_input("Page size", 5, 100, 20, 5)
    total = len(df)
    pages = max((total + page_size - 1) // page_size, 1)
    page = st.number_input("Page", 1, pages, 1, 1)
    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(df.iloc[start:end], use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Add Selected to Watchlist"):
            st.success("Added (local session only)")
    with c2:
        if st.button("Export CSV"):
            st.session_state["sig_csv"] = df.to_csv(index=False).encode(
                "utf-8"
            )
        if st.session_state.get("sig_csv"):
            st.download_button(
                "Download CSV",
                st.session_state["sig_csv"],
                "signals.csv",
                "text/csv",
            )
    with c3:
        if st.button("Export PDF"):
            lines = ["Signals Snapshot"] + [
                (
                    f"{r.Symbol} (TF {getattr(r, 'TF', '')}) "
                    f"score {r.Score} RR {getattr(r, 'RR', '')} — "
                    f"{r.Reason} [{getattr(r, 'Change', '')}]"
                )
                for r in df.head(15).itertuples()
            ]
            pdf = make_pdf_from_text("Signals", lines)
            st.session_state["sig_pdf"] = pdf
        if st.session_state.get("sig_pdf"):
            st.download_button(
                "Download PDF",
                st.session_state["sig_pdf"],
                "signals.pdf",
                "application/pdf",
                disabled=st.session_state["sig_pdf"] is None,
            )

    if st.button("Notify Telegram"):
        ok = send_telegram_message(
            "Signals update ready (see dashboard)"
        )
        st.success("Sent") if ok else st.warning(
            "Not sent (check token/chat id)"
        )

    # save snapshot for next comparison
    try:
        curr = {}
        for r in df.itertuples():
            k = f"{r.Symbol}:{getattr(r, 'TF', '')}"
            curr[k] = float(getattr(r, "Score", 0) or 0)
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(curr, f, indent=2)
    except Exception:
        pass
else:
    st.info(
        "No setups right now. We'll notify you on Telegram as soon as "
        "opportunities appear."
    )
    if st.button("Notify me on Telegram when ready"):
        ok = send_telegram_message(
            "Signals empty at the moment. You'll be notified when new "
            "setups appear."
        )
        st.success("Queued notification") if ok else st.warning(
            "Not sent (check token/chat id)"
        )

st.caption(
    "Confidence uses textual badges; RR = (Target-Price)/(Price-Stop)."
)
