"""
Local package named `platform` collides with Python's stdlib module `platform`.
This shim loads the real stdlib platform.py from disk and delegates any missing
attributes to it, so libraries expecting stdlib behavior (e.g. uuid, streamlit,
python-telegram-bot) work correctly.
"""

from types import ModuleType
from typing import Any
import importlib.util
import sysconfig
import sys
import os

_stdlib_platform: ModuleType | None = None


def _stdlib_platform_path() -> str:
	stdlib_dir = sysconfig.get_paths().get("stdlib")
	if not stdlib_dir:
		version = f"{sys.version_info.major}.{sys.version_info.minor}"
		stdlib_dir = os.path.join(sys.base_prefix, "lib", f"python{version}")
	return os.path.join(stdlib_dir, "platform.py")


def _load_stdlib_platform() -> ModuleType:
	global _stdlib_platform
	if _stdlib_platform is None:
		path = _stdlib_platform_path()
		spec = importlib.util.spec_from_file_location("_stdlib_platform", path)
		if spec is None or spec.loader is None:
			raise ImportError(f"Cannot load stdlib platform module from {path}")
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)  # type: ignore[attr-defined]
		_stdlib_platform = mod
	return _stdlib_platform


def __getattr__(name: str) -> Any:
	std = _load_stdlib_platform()
	return getattr(std, name)


def __dir__():
	std = _load_stdlib_platform()
	return sorted(set(globals().keys()) | set(dir(std)))
