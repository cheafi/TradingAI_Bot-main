import logging
import backtrader as bt

logger = logging.getLogger(__name__)

class SampleStrategy(bt.Strategy):
    params = (('period', 20),)

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.period)

    def next(self):
        if self.data.close[0] > self.sma[0] and not self.position:
            self.buy()
        elif self.data.close[0] < self.sma[0] and self.position:
            self.sell()
