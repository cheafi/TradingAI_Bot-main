# ui/i18n.py
from dataclasses import dataclass

@dataclass(frozen=True)
class I18N:
    title: str
    inputs: str
    symbol: str
    mode: str
    bars: str
    refresh: str
    market_data: str
    metrics: str
    price: str
    atr_pct: str
    keltner_width: str
    signal_now: str
    signal_long: str
    signals: str
    download_enriched: str
    backtest_quick: str
    backtest_log: str
    risk_metrics: str
    sharpe: str
    mdd: str
    var95: str
    pnl: str

ZH_TW = I18N(
    title="TradingAI Bot — 量化交易儀表板",
    inputs="輸入",
    symbol="商品代號",
    mode="模式",
    bars="歷史 K 線數",
    refresh="重新整理",
    market_data="市場走勢",
    metrics="關鍵指標",
    price="最新價格",
    atr_pct="ATR 百分比",
    keltner_width="Keltner 寬度",
    signal_now="即時訊號",
    signal_long="Strong Buy/Buy (Long)",
    signals="訊號表 (True=Long)",
    download_enriched="下載強化後資料 (CSV)",
    backtest_quick="快速回測 (示範)",
    backtest_log="回測紀錄",
    risk_metrics="風險指標",
    sharpe="Sharpe",
    mdd="最大回撤",
    var95="VaR 95%",
    pnl="收益曲線",
)
