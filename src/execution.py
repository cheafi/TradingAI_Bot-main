import os, time, logging
from typing import Dict, Any
import ccxt

log = logging.getLogger("exec")

def _exchange():
    key = os.getenv("BINANCE_KEY")
    secret = os.getenv("BINANCE_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    params = {"enableRateLimit": True}
    if key and secret:
        params["apiKey"] = key
        params["secret"] = secret
    ex = ccxt.binanceusdm(params) if os.getenv("BINANCE_UM", "false").lower()=="true" else ccxt.binance(params)
    if testnet and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)
    return ex

def create_order(symbol: str, side: str, qty: float, price: float=None, type_:str="market") -> Dict[str, Any]:
    try:
        ex = _exchange()
        if type_ == "market":
            order = ex.create_order(symbol=symbol, type="market", side=side, amount=qty)
        else:
            order = ex.create_order(symbol=symbol, type="limit", side=side, amount=qty, price=price)
        log.info("Order placed: %s", order)
        return {"ok": True, "order": order}
    except Exception as e:
        log.exception("Order error: %s", e)
        return {"ok": False, "error": str(e)}
