"""Top-level package for TradingAI_Bot.

Provides a minimal package marker and convenience imports so
`from src.main import ...` works when the package is installed or
when PYTHONPATH points to the project root.
"""

__all__ = ["main"]
__version__ = "0.1.0"

# Convenience import so scripts can do `from src.main import ...`
from . import main  # noqa: F401
