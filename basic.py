#!/usr/bin/env python3
# basic.py — safe demo runner with robust imports

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.main import demo_run
from src.config import Config

def main():
    cfg = Config()
    cfg.mode = "demo"
    cfg.symbol = "BTC/USDT"
    demo_run(cfg)

if __name__ == "__main__":
    main()
