from __future__ import annotations
import pandas as pd
import yfinance as yf
from .cache import GLOBAL_CACHE


class MarketDataService:
    def history(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        key = f"hist:{symbol}:{period}:{interval}"

        def _load():
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
            )
            return df if df.empty is False else None

        df = GLOBAL_CACHE.cached(key, _load)
        if df is None:
            raise ValueError("No data")
        return df


MARKET_DATA = MarketDataService()

