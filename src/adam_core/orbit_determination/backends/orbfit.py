"""
OrbFit backend wrapper for
:func:`~adam_core.orbit_determination.fit_orbit.fit_orbit`.

.. note::

    This backend is a **stub**.  The ``orbfit`` Python package API has not
    yet been finalised.  Once a stable API is available this module should:

    1. Convert :class:`~adam_core.orbit_determination.evaluate.OrbitDeterminationObservations`
       to the input format expected by ``orbfit``.
    2. Invoke the OrbFit solver.
    3. Parse the solver's output into
       :class:`~adam_core.orbit_determination.fitted_orbits.FittedOrbits` and
       :class:`~adam_core.orbit_determination.fitted_orbits.FittedOrbitMembers`.

Installation (once available)
------------------------------
::

    pip install orbfit
    # or
    pip install adam_core[orbfit]

TODO(od-module): Implement OrbFit backend once orbfit Python API is stable.
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
    import orbfit as _orbfit  # noqa: F401

    _ORBFIT_AVAILABLE = True
except ImportError:
    _ORBFIT_AVAILABLE = False


def _get_orbfit_version() -> Optional[str]:
    try:
        return version("orbfit")
    except PackageNotFoundError:
        return None


class OrbFitBackend(BackendWrapper):
    """Orbit-determination backend wrapping the OrbFit solver.

    .. warning::
        This backend is not yet implemented.  Calling :meth:`fit` will
        raise :exc:`NotImplementedError`.

    Parameters
    ----------
    None

    Raises
    ------
    ImportError
        If the ``orbfit`` package is not installed.
    NotImplementedError
        Always, until this stub is completed.
    """

    AVAILABLE: bool = _ORBFIT_AVAILABLE
    BACKEND_NAME: str = "orbfit"

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
            If ``orbfit`` is not installed.
        NotImplementedError
            Always — this backend is a stub.
        """
        self._check_available()
        # TODO(od-module): Implement OrbFit input/output translation and invocation.
        raise NotImplementedError(
            "The OrbFit backend is not yet implemented.  "
            "Contributions are welcome — see the adam_core orbit determination "
            "documentation for the expected input/output contract."
        )
