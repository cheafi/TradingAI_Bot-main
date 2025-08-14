from .scalping import ScalpingConfig, enrich, signal
REGISTRY = {
    "scalping_ml": {"config": ScalpingConfig, "enrich": enrich, "signal": signal},
    # "market_making": {...}
}