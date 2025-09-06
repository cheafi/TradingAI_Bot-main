"""Top-level package for TradingAI_Bot.

Lightweight init so that importing lightweight modules (e.g. telegram bot)
does not force heavy dependencies (pandas/numpy) via src.main.

Use `from src import main` only when needed; lazy attribute provided.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"

# Lazy attribute access for backward compatibility
def __getattr__(name):  # pragma: no cover
    if name == "main":
        from . import main as _main  # local import to avoid eager heavy deps
        return _main
    raise AttributeError(name)
