#!/usr/bin/env python3
import sys, os, importlib.util, traceback

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _import_demo_run():
    try:
        from src.main import demo_run  # type: ignore
        return demo_run
    except Exception:
        try:
            main_path = os.path.join(ROOT, "src", "main.py")
            spec = importlib.util.spec_from_file_location("src.main", main_path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)  # type: ignore
            return getattr(module, "demo_run", None)
        except Exception:
            traceback.print_exc()
            return None

demo_run = _import_demo_run()
if demo_run is None:
    raise RuntimeError("Could not import demo_run from src.main")

if __name__ == "__main__":
    print("Running demo for BTC/USDT …")
    demo_run("BTC/USDT")
    print("Done.")
