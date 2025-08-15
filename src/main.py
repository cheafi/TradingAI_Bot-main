from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import argparse
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class RunConfig:
    mode: str = "demo"
    symbol: str = "BTC/USDT"
    interval: str = "1m"
    dry_run: bool = True


def demo_run(cfg: RunConfig) -> Dict:
    """Lightweight demo entry used by UI/tests/basic scripts."""
    logging.info("Starting demo_run: %s", cfg)
    result = {
        "symbol": cfg.symbol,
        "mode": cfg.mode,
        "status": "ok",
        "notes": "demo placeholder",
    }
    print(result)
    return result


def run_once(cfg: RunConfig) -> Dict:
    """Run a single iteration (used by UI/dashboard)."""
    return demo_run(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.main")
    parser.add_argument(
        "--mode",
        choices=["demo", "backtest", "live"],
        default="demo",
        help="Run mode: demo/backtest/live",
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Symbol to operate on, e.g. BTC/USDT",
    )
    parser.add_argument(
        "--interval",
        default="1m",
        help="Data interval (e.g. 1m, 5m, 1h)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing real orders (safe mode)",
    )
    args = parser.parse_args()
    cfg = RunConfig(
        mode=args.mode,
        symbol=args.symbol,
        interval=args.interval,
        dry_run=args.dry_run,
    )
    demo_run(cfg)


if __name__ == "__main__":
    main()
