# tests/conftest.py
import sys
import os

# ensure project root is in sys.path for tests
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
