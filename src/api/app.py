from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .service import MARKET_DATA

app = FastAPI(title="TradingAI API", version="0.1.0")


class OHLCRequest(BaseModel):
    symbol: str
    period: str = "6mo"
    interval: str = "1d"


@app.get("/health")
async def health():  # pragma: no cover - trivial
    return {"status": "ok"}


@app.post("/history")
async def history(req: OHLCRequest):
    try:
        df = MARKET_DATA.history(req.symbol, req.period, req.interval)
        return {
            "symbol": req.symbol.upper(),
            "rows": len(df),
            "data": df.reset_index().to_dict(orient="records"),
        }
    except Exception as e:  # pragma: no cover - broad fallback
        raise HTTPException(status_code=400, detail=str(e)) from e
