# Security & Operational Safeguards
Version: 1.0

## 1. Secrets Management
- All secrets via environment variables (.env NOT committed)
- Never store API keys in code or logs
- Rotating Keys: schedule: quarterly or on incident
- Optional: use OS keychain / cloud secret manager abstraction later

## 2. Data Sources & Validation
- Dual price sources (primary + alt); if both stale >30s → suppress signals
- Divergence >0.5% mid-price triggers HOLD state for affected symbol

## 3. Logging & PII
- Logs exclude secrets, position sizes beyond aggregated metrics
- Use structured JSON logs (future) with redaction middleware

## 4. Access Control (Future Roadmap)
- Local dev: implicit trust
- Production: least-privilege service accounts per integration (data, execution, analytics)

## 5. File & Artifact Hygiene
- models/ & model_artifacts/ ignored unless explicitly versioned via registry
- data/ local only (excluded from VCS except .gitkeep)

## 6. Integrity & Tamper Signals
- Hash governance policy docs on load (optional) to detect unauthorized edits
- Maintain SHA256 manifest of core strategy files

## 7. Incident Classes
| Class | Example | Action |
|-------|---------|--------|
| DATA_STALE | No fresh price | Suppress alerts, retry loop |
| DIVERGENCE | Source mismatch | Flag symbol HOLD |
| SECRET_INVALID | Auth failure | Rotate key; pause affected agents |
| ABNORMAL_TURNOVER | > threshold | Reduce sizing 20% |
| DRAWDOWN_BREACH | > limits | Trigger state machine (halt) |

## 8. Kill-Switch
- Manual: /halt command (authorized users list)
- Automatic: drawdown, data integrity, or cost anomaly triggers
- Effect: cancel queued signals, shadow new, flatten open (paper)

## 9. Supply Chain
- Pin critical deps (pandas, numpy, telegram bot client)
- Weekly dependency diff review
- Optional: pip hash checking phase

## 10. Telemetry & Metrics (Planned)
- Expose /health (latency, freshness, uptime % window)
- Prometheus / OpenMetrics adapter (future)

## 11. Backup & Recovery
- Config & policy docs: git authoritative
- Runtime state (calibration stats) serialized daily to versioned snapshot directory (excluded from repo)

## 12. User Privacy
- No user-identifiable trading behavior stored; only aggregate acceptance metrics for optimization

## 13. Change Governance
- PR required for strategy logic modifications
- Automatic test suite must pass (unit + calibration) prior to merge

---
Hardening Roadmap: mTLS for internal services, signed model artifacts, anomaly detection on signal generation rate, automated secret rotation hooks.
