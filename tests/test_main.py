import pytest
from src.main import RunConfig, demo_run


def test_demo_run_returns_dict():
    cfg = RunConfig(mode="demo", symbol="BTC/USDT", interval="1m", dry_run=True)
    res = demo_run(cfg)
    assert isinstance(res, dict)
    assert res.get("symbol") == "BTC/USDT"
    assert res.get("mode") == "demo"
    assert res.get("status") == "ok"
