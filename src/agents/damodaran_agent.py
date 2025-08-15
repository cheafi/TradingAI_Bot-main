import logging
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DamodaranAgent:
    """
    Agent based on Aswath Damodaran's valuation principles.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

    def fetch_financial_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch financial data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(self.symbol)
            income_statement = ticker.income_stmt
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            return income_statement, balance_sheet, cash_flow
        except Exception as e:
            logger.error(f"Error fetching financial data for {self.symbol}: {e}")
            return None, None, None

    def calculate_dcf(self, income_statement: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> float:
        """Calculate Discounted Cash Flow (DCF) valuation."""
        try:
            # Extract Free Cash Flow (FCF)
            fcf = cash_flow.loc["Free Cash Flow"].values[0]

            # Extract discount rate (Weighted Average Cost of Capital - WACC)
            # This is a placeholder and needs to be replaced with actual WACC calculation
            wacc = 0.08  # Assume 8% WACC for now

            # Project FCF for 5 years (placeholder)
            fcf_projection = [fcf * (1 + 0.05)**i for i in range(1, 6)]  # Assume 5% growth

            # Calculate terminal value (placeholder)
            terminal_value = fcf_projection[-1] * (1 + 0.03) / (wacc - 0.03)  # Assume 3% terminal growth

            # Discount FCF and terminal value
            pv_fcf = [fcf_projection[i] / (1 + wacc)**(i+1) for i in range(5)]
            pv_terminal_value = terminal_value / (1 + wacc)**5

            # Calculate DCF value
            dcf_value = sum(pv_fcf) + pv_terminal_value

            return dcf_value
        except Exception as e:
            logger.error(f"Error calculating DCF for {self.symbol}: {e}")
            return 0.0

    def generate_signal(self) -> str:
        """Generate trading signal based on DCF valuation."""
        income_statement, balance_sheet, cash_flow = self.fetch_financial_data()
        if income_statement is None or balance_sheet is None or cash_flow is None:
            return "HOLD"

        dcf_value = self.calculate_dcf(income_statement, balance_sheet, cash_flow)
        try:
            ticker = yf.Ticker(self.symbol)
            current_price = ticker.fast_info.last_price
        except Exception as e:
            logger.error(f"Error fetching current price for {self.symbol}: {e}")
            return "HOLD"

        if dcf_value > current_price * 1.2:
            return "STRONG BUY"
        elif dcf_value > current_price * 1.1:
            return "BUY"
        elif dcf_value < current_price * 0.9:
            return "SELL"
        else:
            return "HOLD"
