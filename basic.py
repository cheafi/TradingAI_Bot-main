# basic.py
"""Run a one-step demo so newcomers can see the bot in action without keys."""
from src.main import demo_run

if __name__ == "__main__":
    print("Running demo (safe, offline-friendly)...")
    demo_run("BTC/USDT")
    print("Demo finished. See logs for details.")
