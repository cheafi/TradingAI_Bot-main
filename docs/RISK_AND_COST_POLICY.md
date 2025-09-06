# Risk & Cost Policy

Default Markets: BTC, ETH, US Large-Cap Tech (AAPL, MSFT, NVDA, AMZN, GOOGL, META)
Default Risk Profile: Balanced
Document Version: 1.0

## 1. Portfolio Risk Parameters
- Target Annualized Volatility: 10–12%
- Max Single Position: 6% equity (600 bp)
- Max Theme Exposure (AI/MegaCap): 25% equity (2500 bp)
- Max Aggregate Crypto Exposure: 15% equity
- Max Daily Turnover (notional / equity): 35%
- Max Weekly Turnover: 140%
- Intraday Kill-Switch Drawdown: -3%
- Daily Hard Loss Limit: -5% (flatten + halt new alerts 24h)

## 2. Position Sizing
Base bp by conviction (pre-adjust): C5=150, C4=100, C3=70
```
adjusted_size_bp = base_bp * (target_portfolio_vol / current_realized_vol)
```
Clamp: single 600 bp, theme 2500 bp, crypto pair 800 bp. Scale down proportionally if breach.

## 3. Cost Model
all_in_cost_bps = fees + slippage + spread + borrow(shorts) + impact_buffer
- fees: venue config
- slippage: k * sqrt(order_notional / ADV_notional)
- spread: median inside spread * participation_factor
- borrow: annual_borrow * (horizon_days) * 10,000
- impact_buffer: +20% (slippage+spread) in high-vol regime
Gate: expected_alpha_bps ≥ 2 × all_in_cost_bps (else shadow)

## 4. Regime Multipliers
| Regime | Size Mult | Turnover Adj | Notes |
|--------|-----------|--------------|-------|
| TR_UP_VOL_LOW | 1.00 | baseline | Full agents |
| TR_UP_VOL_HIGH | 0.85 | -5% abs | Trim mean-revert |
| TR_DOWN_VOL_LOW | 0.90 | baseline | Momentum bias |
| TR_DOWN_VOL_HIGH | 0.75 | -10% abs | Disable discretionary low-conv |

## 5. Drawdown State Machine
NORMAL -> (DD <= -3%) -> DEGRADED -> (recover > -1%) -> NORMAL
DEGRADED -> (DD <= -5%) -> HALTED
Actions:
- DEGRADED: suppress non-C5, -15% size
- HALTED: flatten, archive, require manual reset

## 6. Turnover Control
If 5d avg turnover > 150% target → reduce base sizes 20% until back within band.

## 7. Liquidity / Participation
- Max per-bar participation: 10% est. vs historical median bar volume
- If projected > limit → slice OR shadow

## 8. Compliance Filters
- Blacklist & earnings embargo pre-check
- Min median daily $ volume (equities): $20M
- Region exclusions applied pre-signal

## 9. Data Quality Gates
- Price freshness ≤ 30s
- Dual-source divergence ≤ 0.5% (else hold)

## 10. Monitoring KPIs
- Realized vol vs target
- Max DD (daily / rolling 30d)
- Turnover (daily / weekly)
- Net Edge After Cost (strat)
- T1 Hit-Rate (conviction tier)
- Slippage vs model (bps)

## 11. Change Control
Material parameter edits require weekly IC notation (version bump + rationale).

## 12. Shadow Mode Policy
Bottom decile Net Edge After Cost (rolling 20d) → shadow 10 trading days; auto-review.

## 13. Escalation
Sequence: Auto trigger → Telegram /status → log incident file → optional email/pager integration (future).

---
Future: dynamic borrow sourcing, queue-aware microstructure impact model, cross-asset correlation-adjusted sizing.
