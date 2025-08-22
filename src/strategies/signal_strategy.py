"""
Signal-based strategy for Backtester integration.
Executes trades based on pre-computed signals in the dataframe.
"""
import backtrader as bt
import logging

logger = logging.getLogger(__name__)


class SignalStrategy(bt.Strategy):
    """Strategy that trades based on pre-computed signals."""
    
    params = (
        ('signal_col', 'signal'),  # Column name for signals (0/1)
        ('position_size', 0.95),   # Fraction of available cash to use
    )
    
    def __init__(self):
        self.signal = self.data.signal if hasattr(self.data, 'signal') else None
        self.order = None
        
    def next(self):
        if self.order:
            return  # Skip if order is pending
            
        if self.signal is None:
            return
            
        current_signal = self.signal[0]
        
        # Long signal and not in position
        if current_signal > 0.5 and not self.position:
            size = int((self.broker.getcash() * self.params.position_size) / self.data.close[0])
            if size > 0:
                self.order = self.buy(size=size)
                logger.debug(f"BUY signal: size={size}, price={self.data.close[0]:.2f}")
                
        # Exit signal or no signal and in position
        elif current_signal < 0.5 and self.position:
            self.order = self.sell(size=self.position.size)
            logger.debug(f"SELL signal: size={self.position.size}, price={self.data.close[0]:.2f}")
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"BUY EXECUTED: {order.executed.size} @ {order.executed.price:.2f}")
            else:
                logger.info(f"SELL EXECUTED: {order.executed.size} @ {order.executed.price:.2f}")
        self.order = None
