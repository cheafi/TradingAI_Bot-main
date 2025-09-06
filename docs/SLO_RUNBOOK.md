# SLOs & Operational Runbook
Version: 1.0

## 1. Service Level Objectives
| Metric | Objective | Window | Notes |
|--------|-----------|--------|-------|
| Alert Latency (p95) | < 60s | 30d | Signal generation → user delivery |
| Data Freshness | < 30s | 30d | Time since last valid price |
| Uptime (Core Bot) | ≥ 99.5% | 90d | Excludes planned maintenance |
| Calibration Job Success | 100% | 30d | Miss → alert |
| Cost Model Drift | < 10% error | 30d | Model vs realized slippage |

Error Budget = (1 - Objective). Breach triggers feature freeze until burn rate < threshold.

## 2. Monitoring & Alerts
- Latency tracker wraps send pipeline
- Freshness watchdog per symbol
- Heartbeat every 60s; 3 consecutive misses → incident
- Slippage vs model tracked per fill (paper approximation)

## 3. Operational States
| State | Description | Entry Trigger | Exit |
|-------|-------------|---------------|------|
| NORMAL | All SLOs healthy | Startup / recovery | n/a |
| DEGRADED | One SLO near breach | Latency p95 > 55s or freshness > 25s | Metrics < warning for 2h |
| HALTED | Hard breach / safety trigger | Drawdown, data stale global, multiple failures | Manual reset |

## 4. Runbooks
### 4.1 Latency Degradation
1. Check queue depth
2. Disable non-critical agents
3. Increase batch size or parallel workers
4. If unresolved 15m → enter DEGRADED broadcast status

### 4.2 Data Staleness
1. Verify primary feed connection
2. Switch to alternate source
3. If both stale > 5m → HALTED
4. Notify via /status update

### 4.3 Drawdown Breach
1. Auto flatten paper book
2. Set state=HALTED
3. Archive open signals
4. Require reset token (/resume <token>)

### 4.4 Cost Model Drift
1. Recompute k parameter for slippage
2. Compare prior rolling window
3. If deviation persists 3 windows → escalate to IC

## 5. /status Output (Example)
```
STATE: NORMAL  Uptime: 99.7% (30d)
Active Agents: momentum, mean_reversion
Drawdown Today: -0.8%  Turnover: 24%
Alerts Sent: 5 / 8 cap
Data Freshness (avg): 12s  Latency p95: 41s
Shadow: 2 strategies (low edge)
```

## 6. Incident Logging
- Append JSON line objects to runtime/incidents/YYYY-MM-DD.log (ignored by VCS)

## 7. Weekly Review Checklist
- Error budget consumption
- Top incident categories
- Mean recovery time
- Calibration shift summary
- Pending risk parameter changes

## 8. Versioning & Ownership
- Owner: Ops / Reliability (placeholder)
- Changes: PR with rationale + version bump

---
Future: automated pager integration, anomaly-based preemptive throttling, chaos drills.
