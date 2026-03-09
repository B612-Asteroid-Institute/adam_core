"""
Layup backend wrapper for
:func:`~adam_core.orbit_determination.fit_orbit.fit_orbit`.

.. note::

    This backend is a **stub**.  Once the ``layup`` package exposes a stable
    Python OD API this module should:

    1. Convert :class:`~adam_core.orbit_determination.evaluate.OrbitDeterminationObservations`
       to the format expected by ``layup``.
    2. Invoke the Layup orbit-determination routine.
    3. Parse the result into
       :class:`~adam_core.orbit_determination.fitted_orbits.FittedOrbits` and
       :class:`~adam_core.orbit_determination.fitted_orbits.FittedOrbitMembers`.

Installation (once available)
------------------------------
::

    pip install layup
    # or
    pip install adam_core[layup]

TODO(od-module): Implement Layup backend once layup OD API is stable.
"""

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Tuple

import pyarrow as pa

from ..config import BackendConfig
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from .base import BackendWrapper

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------
try:
    import layup as _layup  # noqa: F401

    _LAYUP_AVAILABLE = True
except ImportError:
    _LAYUP_AVAILABLE = False


def _get_layup_version() -> Optional[str]:
    try:
        return version("layup")
    except PackageNotFoundError:
        return None


class LayupBackend(BackendWrapper):
    """Orbit-determination backend wrapping the Layup solver.

    .. warning::
        This backend is not yet implemented.  Calling :meth:`fit` will
        raise :exc:`NotImplementedError`.

    Raises
    ------
    ImportError
        If the ``layup`` package is not installed.
    NotImplementedError
        Always, until this stub is completed.
    """

    AVAILABLE: bool = _LAYUP_AVAILABLE
    BACKEND_NAME: str = "layup"

    def fit(
        self,
        observations: OrbitDeterminationObservations,
        config: BackendConfig,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Not yet implemented.

        Parameters
        ----------
        observations : OrbitDeterminationObservations
            Observations for a single object.
        config : BackendConfig
            Runtime configuration (currently unused).

        Raises
        ------
        ImportError
            If ``layup`` is not installed.
        NotImplementedError
            Always — this backend is a stub.
        """
        self._check_available()
        # TODO(od-module): Implement Layup input/output translation and invocation.
        raise NotImplementedError(
            "The Layup backend is not yet implemented.  "
            "Contributions are welcome — see the adam_core orbit determination "
            "documentation for the expected input/output contract."
        )
