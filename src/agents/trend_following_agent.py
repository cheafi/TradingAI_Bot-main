import logging
import yfinance as yf
import pandas as pd
import backtrader as bt

logger = logging.getLogger(__name__)

class TrendFollowingAgent:
    """
    Agent based on trend-following strategy.
    """

    def __init__(self, symbol: str, period: int = 20):
        self.symbol = symbol
        self.period = period

    def generate_signal(self) -> str:
        """Generate trading signal based on trend-following strategy."""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period="1y")
            sma = data['Close'].rolling(window=self.period).mean()
            current_price = data['Close'][-1]
            if current_price > sma[-1]:
                return "BUY"
            elif current_price < sma[-1]:
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            logger.error(f"Error generating signal for {self.symbol}: {e}")
            return "HOLD"
