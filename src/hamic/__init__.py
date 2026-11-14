"""HAMIC: Vietnamese-first language model research toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hamic")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
