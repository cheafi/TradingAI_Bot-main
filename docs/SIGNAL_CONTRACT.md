# Signal & Alert Policy (Signal Contract)

Default Markets: BTC, ETH, AAPL, MSFT, NVDA, AMZN, GOOGL, META  
Default Risk Profile: Balanced  
Daily Max Alerts (post-throttling): 8

## 1. Signal Object (Authoritative Schema)
Each signal MUST serialize to JSON with these fields:
```
{
  "id": <uuid>,
  "timestamp_utc": <ISO8601>,
  "symbol": "BTC-USD" | "ETH-USD" | "AAPL" | ...,
  "market": "crypto" | "equity",
  "side": "long" | "short",
  "entry_range": {"lower": float, "upper": float},
  "targets": [{"level": 1, "price": float}, {"level": 2, "price": float?}],
  "stop": float,
  "horizon_hours": int,  // intended evaluation window
  "expected_alpha_bps": int,  // gross edge vs entry mid
  "all_in_cost_bps": int,  // estimated total cost (see section 3)
  "net_edge_bps": int,  // expected_alpha_bps - all_in_cost_bps
  "conviction": "C3" | "C4" | "C5",
  "confidence_score": float,  // 0-1 calibrated
  "liquidity_score": float,   // 0-1 relative intraday / ADV
  "alert_score": float,       // ranking metric (section 6)
  "regime_tag": "TR_UP_VOL_LOW" | ...,  // 2x2 regime code
  "theme": "AI" | "MegaCap" | "CryptoMomentum" | ...,
  "position_size_bp_equity": int, // recommended gross exposure in basis points
  "why": {
    "setup": string,      // pattern / factor
    "catalyst": string,   // event / flow / context
    "risk": string        // primary risk / invalidation rationale
  },
  "source_agent": string, // agent name
  "version": "1.0",
  "shadow": bool          // if true, logged only (not alerted)
}
```

## 2. Field Definitions & Rules
- id: Deterministic UUIDv5 (namespace=agent-name, name=symbol+timestamp) for idempotency.
- entry_range: Lower/upper bound; if single price, both equal.
- targets: T1 mandatory; T2 optional (performance attribution tracks per level).
- stop: Hard invalidation; execution simulation assumes worst print if gapped through.
- horizon_hours: Defines SLA for evaluation (hit T1, stop, or expired outcome recorded).
- conviction: Three tiers only. C5 must maintain ≥ rolling 60-day T1-before-stop hit-rate of threshold (default 60%). Auto-demote if breached for 10 consecutive trading days.
- confidence_score: Reliability calibration output (isotonic or Platt scaling) mapping historical bucket performance.
- liquidity_score: Normalized (0-1) vs 30-day median notional volume participation at suggested size.
- alert_score: net_edge_bps * conviction_multiplier * liquidity_score (see section 6).
- position_size_bp_equity: Volatility-adjusted recommendation (Section 5) in basis points (100 bp = 1%).
- shadow: True for quarantined strategies (bottom decile net edge) during observation.

## 3. Cost Model (All-in Cost)
all_in_cost_bps = fees_bps + slippage_bps + spread_bps + borrow_bps (if short)  
Default Balanced profile parameters:
- fees_bps: broker_taker_fee (crypto) or commission_equivalent (equity) (configurable per venue)
- slippage_bps: k * sqrt(order_notional / ADV_notional) (k calibrated weekly)
- spread_bps: rolling median inside spread * participation_factor
- borrow_bps: annualized borrow * (horizon_hours / (24*365)) * 10,000

Pass/Fail Badge:
- PASS if expected_alpha_bps ≥ 2 * all_in_cost_bps
- ELSE: Signal demoted to shadow OR truncated (no alert) based on policy priority.

## 4. Net Edge & Quality Gates
- net_edge_bps must be > 0 to surface.
- Minimum absolute expected_alpha_bps threshold (configurable; default 15 bps) to avoid noise.
- If cumulative daily surfaced signals would exceed max alerts (8), rank by alert_score and drop surplus.

## 5. Sizing Logic (Portfolio-Aware)
Target annualized portfolio volatility: 10–12%.  
Per-signal recommended bp:
```
raw_size_bp = base_bp * (target_portfolio_vol / current_realized_vol)
```
Where base_bp defaults:
- C5: 150 bp
- C4: 100 bp
- C3: 70 bp
Clamp rules:
- Single name cap: 600 bp (6%)
- Theme cap: 2500 bp aggregate (25%)
- Crypto pair cap: 800 bp each
If breach would occur, proportionally downscale the new order.

## 6. Alert Scoring & Throttling
conviction_multiplier: C3=1.0, C4=1.25, C5=1.5  
alert_score = net_edge_bps * conviction_multiplier * liquidity_score
Daily selection:
1. Compute alert_score for all candidate signals.
2. Filter PASS (Edge ≥ 2×Cost) & non-shadow.
3. Sort descending; pick top N (N=8 default).
4. Bundle correlated (|ρ| ≥ 0.75) symbols into basket suggestions (user chooses 1–2) – only highest alert_score acts as representative.

## 7. Conviction Calibration Process
Weekly job:
- Bucket outcomes by conviction tier (C3/C4/C5).
- Compute T1-before-stop rate over trailing 30, 60, 90 days.
- Produce reliability chart (line per tier). Auto-demotion triggers if C5 below threshold for sustained window.
- Adjust confidence_score mapping (recalibrate isotonic model) and store revision id.

## 8. Regime Tagging
Regime matrix (2×2): Trend (Up/Down) × Volatility (Low/High).  
Detection: 
- Trend: 20d vs 100d moving average slope & price position.  
- Vol: Realized 10d σ vs 1y median.  
Disable/Enable agents per regime via config (regime_agents.yml) to reduce false positives.

## 9. Lifecycle & Outcomes
A signal closes when:
- T1 hit (partial success; if T2 exists and is hit later, mark layered success),
- Stop hit (failure),
- Expired (no T1/stop within horizon_hours),
- Cancelled (structural reason, e.g., earnings release emerges).  
Outcomes recorded for calibration & KPI.

## 10. Telegram Rendering (User-Facing)
Example (PASS):
```
LONG NVDA (C4) | Net Edge +24 bps (Alpha 40 / Cost 16) [PASS]
Entry: 116.20 – 116.80  Stop: 113.90  T1: 118.40  T2: 120.10  Horizon: 24h
Size: 1.0–1.2% equity (Suggested 110 bp)  Regime: TR_UP_VOL_LOW  Theme: AI
Why: Setup: Pullback to rising 20d VWAP; Catalyst: NVDA supplier bullish guide; Risk: Broad tech fade.
ID: 7ac4...  Not advice.
```

## 11. Shadow Mode
Signals failing Edge≥2×Cost OR demoted strategies: shadow=true → stored, included in evaluation datasets, excluded from immediate alerts. After 2-week observation, auto-review for reinstatement.

## 12. Compliance & Safeguards
- Blackout list & earnings embargo enforced pre-publication.
- Educational disclaimer appended.
- Kill-switch: If intraday net drawdown > configured threshold (e.g., -3%), new non-priority alerts suppressed until reset window.

## 13. Versioning
This document version: 1.0  
Any schema change increments minor version & updates signal.version field.

---
Future Extensions: optional probability distribution over outcomes, dynamic multi-stage target path, slippage model per venue microstructure.
