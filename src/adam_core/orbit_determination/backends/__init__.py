"""
Backend wrappers for :func:`~adam_core.orbit_determination.fit_orbit.fit_orbit`.

Each backend is imported lazily so that missing optional dependencies do not
prevent the rest of adam_core from loading.  After this module has been
imported, the ``*_AVAILABLE`` booleans can be queried to determine which
backends are usable in the current environment.

Availability flags
------------------
FIND_ORB_AVAILABLE : bool
    ``True`` if the ``adam_fo`` package is importable.
ORBFIT_AVAILABLE : bool
    ``True`` if the ``orbfit`` package is importable.
LAYUP_AVAILABLE : bool
    ``True`` if the ``layup`` package is importable.

Examples
--------
>>> from adam_core.orbit_determination.backends import FIND_ORB_AVAILABLE
>>> if FIND_ORB_AVAILABLE:
...     from adam_core.orbit_determination.backends import FindOrbBackend
"""

# flake8: noqa: F401

# --- FindOrb ----------------------------------------------------------------
try:
    import adam_fo  # noqa: F401

    FIND_ORB_AVAILABLE = True
except ImportError:
    FIND_ORB_AVAILABLE = False

# --- OrbFit -----------------------------------------------------------------
try:
    import orbfit  # noqa: F401

    ORBFIT_AVAILABLE = True
except ImportError:
    ORBFIT_AVAILABLE = False

# --- Layup ------------------------------------------------------------------
try:
    import layup  # noqa: F401

    LAYUP_AVAILABLE = True
except ImportError:
    LAYUP_AVAILABLE = False

# Always-available backends
from .adam import AdamBackend
from .base import BackendWrapper

# Conditionally-available backends (always importable; raise at runtime if
# the underlying package is absent)
from .find_orb import FindOrbBackend
from .layup import LayupBackend
from .orbfit import OrbFitBackend

__all__ = [
    "FIND_ORB_AVAILABLE",
    "ORBFIT_AVAILABLE",
    "LAYUP_AVAILABLE",
    "BackendWrapper",
    "AdamBackend",
    "FindOrbBackend",
    "OrbFitBackend",
    "LayupBackend",
]
