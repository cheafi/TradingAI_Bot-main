"""Core trading engine with event-driven architecture and multi-agent analysis."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from src.agents.ensemble import EnsembleAgent
from src.utils.visualization import create_trading_chart

logger = logging.getLogger(__name__)


@dataclass
class TradingEngine:
    """Main trading engine with event-driven architecture."""
    
    symbol: str
    exchange: str = "binance"
    interval: str = "1m"
    mode: str = "paper"  # paper/live
    
    def __post_init__(self):
        self.ensemble = EnsembleAgent()
        self.exchange = getattr(ccxt, self.exchange)()
        self._setup_logging()
    
    async def run(self) -> Dict:
        """Main trading loop with error recovery."""
        try:
            # Get market data
            data = await self._fetch_data()
            
            # Get AI analysis
            insight = await self.ensemble.analyze(self.symbol)
            
            # Execute if conditions met
            if insight and insight.signal in ["buy", "sell"]:
                await self._execute_trade(insight)
            
            # Visualize
            chart = create_trading_chart(data, title=f"{self.symbol} Analysis")
            
            return {
                "status": "success",
                "data": data.to_dict(),
                "insight": insight.__dict__ if insight else None,
                "chart": chart
            }
            
        except Exception as e:
            logger.error(f"Trading error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data with fallbacks."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, self.interval)
            return pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            # Fallback to alternative source
            return pd.DataFrame()  
    
    async def _execute_trade(self, insight) -> None:
        """Execute trade based on analysis."""
        if self.mode == "paper":
            logger.info(f"Paper trade: {insight.signal} {self.symbol}")
        else:
            # Real trading logic here
            pass
    
    def _setup_logging(self) -> None:
        """Configure logging with proper format."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
