"""
Abstract base class for orbit-determination backend wrappers.

Every backend must:

* Set the class-level ``AVAILABLE`` flag at import time by attempting to
  import its optional dependency.
* Implement :meth:`fit` to accept
  :class:`~adam_core.orbit_determination.evaluate.OrbitDeterminationObservations`
  and :class:`~adam_core.orbit_determination.config.BackendConfig` and return
  ``(FittedOrbits, FittedOrbitMembers)``.
* Implement :meth:`__getstate__` / :meth:`__setstate__` from the parent
  :class:`~adam_core.orbit_determination.orbit_fitter.OrbitFitter` ABC so the
  backend can be serialised for Ray-based parallelism.

Notes
-----
``BackendWrapper`` intentionally does *not* subclass ``OrbitFitter`` because
``OrbitFitter.initial_fit`` expects a pre-linked ``object_id`` and is oriented
toward THOR's cluster-and-link pipeline.  ``BackendWrapper.fit`` is a
higher-level entrypoint that accepts a full observation set for a *single*
tracklet and produces a complete fitted orbit including residuals.
"""

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple

from ..config import BackendConfig
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits

logger = logging.getLogger(__name__)


class BackendWrapper(ABC):
    """Abstract wrapper around an external (or internal) orbit-determination backend.

    Attributes
    ----------
    AVAILABLE : bool
        Set to ``True`` at class definition time if the backend's required
        package / binary is detectable.  Checked by
        :func:`~adam_core.orbit_determination.fit_orbit.fit_orbit` before
        dispatching.
    BACKEND_NAME : str
        Short human-readable label used in provenance and error messages.
    """

    AVAILABLE: ClassVar[bool] = False
    BACKEND_NAME: ClassVar[str] = "unknown"

    # ------------------------------------------------------------------
    # Pickling support required by Ray / multiprocessing
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        observations: OrbitDeterminationObservations,
        config: BackendConfig,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Fit an orbit to *observations* using this backend.

        Parameters
        ----------
        observations : OrbitDeterminationObservations
            All observations for a single object.  The caller is responsible
            for ensuring all observations belong to one object.
        config : BackendConfig
            Runtime configuration including weighting policy and any
            backend-specific kwargs.

        Returns
        -------
        fitted_orbits : FittedOrbits
            Best-fit orbit(s) with provenance columns populated.
        fitted_orbit_members : FittedOrbitMembers
            Per-observation residuals and outlier flags.

        Raises
        ------
        RuntimeError
            If the backend binary returns a non-zero exit code or produces
            unreadable output.
        ValueError
            If the backend output cannot be parsed.
        """
        ...

    def _check_available(self) -> None:
        """Raise :exc:`ImportError` with install instructions if not available."""
        if not self.AVAILABLE:
            raise ImportError(
                f"The '{self.BACKEND_NAME}' backend is not available.  "
                f"See the installation instructions for this backend in the "
                f"adam_core orbit determination documentation."
            )
