# Cost model for simulation

def cost_model(trade_size, price, fee_rate=0.0005, slippage_bps=2):
    """
    Calculate total cost for a simulated trade.
    Args:
        trade_size (float): Notional size of trade
        price (float): Trade price
        fee_rate (float): Exchange fee rate (default 0.05%)
        slippage_bps (float): Slippage in basis points (default 2bps)
    Returns:
        float: Total cost in currency units
    """
    fee = abs(trade_size) * price * fee_rate
    slippage = abs(trade_size) * price * (slippage_bps / 10000)
    return fee + slippage
