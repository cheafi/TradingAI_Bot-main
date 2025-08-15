#!/usr/bin/env python3
# basic.py — safe demo runner with robust imports

import sys
import os
import importlib.util
import traceback
import logging
from src.config import TradingConfig

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try normal package import first
demo_run = None
try:
    from src.main import demo_run  # type: ignore
except Exception:
    # fallback: load src/main.py directly by path
    try:
        main_path = os.path.join(ROOT, "src", "main.py")
        spec = importlib.util.spec_from_file_location("src.main", main_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        demo_run = getattr(module, "demo_run", None)
    except Exception:
        print("Failed to import src.main. Traceback:\n")
        traceback.print_exc()
        raise

if demo_run is None:
    raise RuntimeError("demo_run function not found in src.main")

def main():
    cfg = TradingConfig()
    cfg.symbol = "BTC/USDT"  # Set symbol in config
    logging.info("Running safe demo...")
    demo_run(cfg)  # Pass config object

if __name__ == "__main__":
    main()
