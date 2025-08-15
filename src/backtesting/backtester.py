import logging
import backtrader as bt
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: bt.Strategy, initial_capital: float = 10000):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(initial_capital)

    def add_data(self):
        """Add data to Cerebro."""
        try:
            data = bt.feeds.PandasData(dataname=self.data)
            self.cerebro.adddata(data)
        except Exception as e:
            logger.error(f"Error adding data: {e}")

    def add_strategy(self):
        """Add strategy to Cerebro."""
        try:
            self.cerebro.addstrategy(self.strategy)
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")

    def run(self):
        """Run the backtest."""
        try:
            self.add_data()
            self.add_strategy()
            self.cerebro.run()
            portfolio_value = self.cerebro.broker.getvalue()
            pnl = portfolio_value - self.initial_capital
            return pnl, portfolio_value
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return 0, self.initial_capital
