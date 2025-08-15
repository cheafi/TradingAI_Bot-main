import logging
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class GrahamAgent:
    """
    Agent based on Benjamin Graham's investment principles.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

    def fetch_financial_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch financial data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(self.symbol)
            income_statement = ticker.income_stmt
            balance_sheet = ticker.balance_sheet
            return income_statement, balance_sheet
        except Exception as e:
            logger.error(f"Error fetching financial data for {self.symbol}: {e}")
            return None, None

    def calculate_graham_number(self, income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
        """Calculate Graham Number."""
        try:
            # Extract EPS and Book Value per Share
            eps = income_statement.loc["Net Income Applicable To Common Shares"].values[0]
            bvps = balance_sheet.loc["Total Stockholder Equity"].values[0] / yf.Ticker(self.symbol).fast_info.shares

            # Check for non-positive values
            if eps <= 0 or bvps <= 0:
                return 0.0

            # Calculate Graham Number
            graham_number = np.sqrt(22.5 * eps * bvps)  # Graham's formula

            return graham_number
        except Exception as e:
            logger.error(f"Error calculating Graham Number for {self.symbol}: {e}")
            return 0.0

    def generate_signal(self) -> str:
        """Generate trading signal based on Graham Number."""
        income_statement, balance_sheet = self.fetch_financial_data()
        if income_statement is None or balance_sheet is None:
            return "HOLD"

        graham_number = self.calculate_graham_number(income_statement, balance_sheet)
        try:
            ticker = yf.Ticker(self.symbol)
            current_price = ticker.fast_info.last_price
        except Exception as e:
            logger.error(f"Error fetching current price for {self.symbol}: {e}")
            return "HOLD"

        if graham_number > current_price * 1.2:
            return "STRONG BUY"
        elif graham_number > current_price * 1.1:
            return "BUY"
        elif graham_number < current_price * 0.9:
            return "SELL"
        else:
            return "HOLD"
