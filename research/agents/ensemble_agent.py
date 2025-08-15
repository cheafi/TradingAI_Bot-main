# research/agents/ensemble_agent.py
from .valuation_agent import valuation_dcf_stub
from .sentiment_agent import sentiment_stub

def ensemble_suggestion(ticker: str) -> dict:
    val = valuation_dcf_stub(ticker)
    sent = sentiment_stub(ticker)
    score = 0.5 * (val["fair_value"]) + 0.5 * (100 * (sent["sentiment"] + 1))  # toy scoring
    return {"ticker": ticker, "score": score, "val": val, "sent": sent, "recommendation": "BUY" if score > 50 else "HOLD"}