# Product Charter

**Vision:**
A dependable, backtest-to-paper trading platform for rapid idea testing, fair comparison, and robust, low-cost strategy promotion.

**Scope (30 days):**
- 1 asset class: BTC/ETH
- 1–3 strategies (agent set)
- Full pipeline: data → signals → portfolio/risk → execution simulation → report

**KPIs:**
- Out-of-sample Sharpe ≥ 1.0 (paper)
- Max drawdown ≤ 15% (paper)
- Median implementation shortfall ≤ planned cost model
- Paper/live parity checks pass for 2 consecutive weeks

**Guardrails:**
- Promote to paper only if DSR ≥ 0, CPCV used, econ sanity checks pass
- Runbooks for outages, slippage spikes, daily close
