try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version as _distribution_version

        __version__ = _distribution_version("adam-core")
    except ImportError:
        __version__ = "0.0.0dev0"

# The migrated package has no rustless runtime mode. Importing `adam_core`
# validates that the compiled extension is present and matches this Python
# surface before any later submodule can fall into legacy behavior.
from . import _rust as _rust

__all__ = ["__version__", "_rust"]
