import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

class MeanReversionAgent:
    """
    Agent based on mean-reversion strategy.
    """

    def __init__(self, symbol: str, period: int = 20, std_dev: int = 2):
        self.symbol = symbol
        self.period = period
        self.std_dev = std_dev

    def generate_signal(self) -> str:
        """Generate trading signal based on mean-reversion strategy."""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period="1y")
            mean = data['Close'].rolling(window=self.period).mean()
            std = data['Close'].rolling(window=self.period).std()
            upper_band = mean + self.std_dev * std
            lower_band = mean - self.std_dev * std
            current_price = data['Close'][-1]

            if current_price < lower_band[-1]:
                return "BUY"
            elif current_price > upper_band[-1]:
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            logger.error(f"Error generating signal for {self.symbol}: {e}")
            return "HOLD"
