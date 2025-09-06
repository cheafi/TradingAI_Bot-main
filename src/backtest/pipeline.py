"""Minimal backtest pipeline: data -> agent -> positions -> PnL metrics."""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ..agents.momentum_agent import MomentumAgent
from ..utils.data import synthetic_ohlcv
from ..cost_model import cost_model


def run_backtest(symbol: str = "BTC/USDT", initial_capital: float = 50_000.0) -> Dict[str, Any]:
    data = synthetic_ohlcv(symbol, 2000)
    agent = MomentumAgent()
    signals = agent.generate_signals(data)
    aligned = data.loc[signals.index]

    # Position sizing: simple proportional to confidence (max 1x notional)
    position = signals['signal'] * signals['confidence']
    # Shift to represent entering at next bar open
    position = position.shift(1).fillna(0)

    ret = aligned['close'].pct_change().fillna(0)
    strat_ret = position * ret

    # Apply trading costs when position changes
    trades = position.diff().fillna(0).abs()
    avg_price = aligned['close']
    trade_costs = trades * avg_price.apply(lambda p: cost_model(1, p)) / initial_capital
    net_ret = strat_ret - trade_costs

    equity_curve = (1 + net_ret).cumprod() * initial_capital
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    ann_factor = 365 * 24 * 4  # 15m bars per year approx
    vol = net_ret.std() * np.sqrt(ann_factor)
    sharpe = (net_ret.mean() * ann_factor) / (net_ret.std() + 1e-12)
    max_dd = drawdown.min()
    # Turnover = average absolute position change per bar
    try:
        turnover_raw = float(np.asarray(trades.sum()).item())
    except Exception:
        turnover_raw = 0.0
    turnover = turnover_raw / max(len(trades), 1)

    return {
        'symbol': symbol,
        'total_return': total_return,
        'sharpe': sharpe,
        'volatility': vol,
        'max_drawdown': max_dd,
        'turnover': turnover,
        'last_equity': equity_curve.iloc[-1],
        'bars': len(net_ret),
    }

if __name__ == "__main__":  # pragma: no cover
    import json
    stats = run_backtest()
    print(json.dumps(stats, indent=2))
