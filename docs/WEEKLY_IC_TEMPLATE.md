# Weekly Investment Committee Template
Version: 1.0

Week Ending: YYYY-MM-DD
Prepared By: <name>

## 1. Executive Snapshot
| Metric | This Week | Prev Week | 4w Avg | Notes |
|--------|-----------|----------|--------|-------|
| Portfolio Sharpe (paper) | | | | |
| Max Drawdown (rolling 30d) | | | | |
| Net Edge After Cost (bps) | | | | |
| T1 Hit-Rate (C5 / C4 / C3) | | | | |
| Avg Alert Latency p95 (s) | | | | |
| Data Freshness p95 (s) | | | | |
| Turnover (%) | | | | |
| Error Budget Burn (%) | | | | |

## 2. Reliability & Calibration
- Conviction Reliability Table:
```
Tier  30d Hit  60d Hit  90d Hit  Action
C5    xx%      xx%      xx%      (retain / demote)
C4    xx%      xx%      xx%      (promote?
C3    xx%      xx%      xx%      (stable)
```
- Any auto-demotions/promotions executed?

## 3. Strategy Attribution
| Strategy | Alerts | Net Edge After Cost (bps) | Sharpe (paper) | Hit-Rate T1 | Shadow? | Action |
|----------|--------|---------------------------|----------------|-------------|---------|--------|

## 4. Regime & Allocation
- Current Regime: <TR_UP_VOL_LOW | ...>
- Enabled Agents: [...]
- Disabled Agents (why): [...]
- Theme Exposure vs Caps:
```
AI: xx% / 25%
Crypto: xx% / 15%
Single Largest: <symbol> xx% / 6%
```

## 5. Risk & Incidents
- Drawdown events: summary
- Data / latency incidents: count & MTTR
- Kill-switch activations: details

## 6. Cost & Slippage
| Symbol | Model Slippage (bps) | Realized (bps) | Drift % | Notes |
|--------|----------------------|---------------|---------|-------|

## 7. Experiments & A/B Tests
| Experiment | Variant A vs B | KPI | Result | Next Step |
|------------|---------------|-----|--------|-----------|

## 8. Decisions Log
- Parameter changes approved
- Strategies promoted/demoted
- New caps or thresholds

## 9. Next Week Focus
- Reliability tasks
- Calibration updates
- New experiments

## 10. Open Risks / Blockers
- <risk> — mitigation plan

## 11. Appendices
- Reliability chart (conviction tiers)
- Equity curve & turnover plot
- Shadow strategy performance summary

---
Usage: Populate Monday morning; archive to reports/weekly/YYYY-MM-DD.md (ignored from VCS if private) and circulate.
