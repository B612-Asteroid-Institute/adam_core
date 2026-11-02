"""
Deprecated legacy least-squares entry point.

`LeastSquares` keeps its historical interface but no longer runs its own
finite-difference (Vallado) loop: `LeastSquares.least_squares` refines the
orbit with `refine` (``rejection="none"``), the native refinement stage. It
will be removed in a later release.
"""

import logging
import warnings
from abc import ABC
from typing import Any, Dict, Optional

import numpy as np

from ..coordinates import CoordinateCovariances, SphericalCoordinates
from ..orbits.orbits import Orbits
from ..propagator import Propagator
from .evaluate import FittedOrbits, OrbitDeterminationObservations

logger = logging.getLogger(__name__)


def _angular_only(
    observations: OrbitDeterminationObservations,
) -> OrbitDeterminationObservations:
    """The observations' RA and Dec with their 2x2 covariance block: the
    legacy solver fitted only those, whatever else the rows carried."""
    coordinates = observations.coordinates
    matrix = coordinates.covariance.to_matrix()
    angular = np.full_like(matrix, np.nan)
    angular[:, 1:3, 1:3] = matrix[:, 1:3, 1:3]
    return observations.set_column(
        "coordinates",
        SphericalCoordinates.from_kwargs(
            lon=coordinates.lon,
            lat=coordinates.lat,
            covariance=CoordinateCovariances.from_matrix(angular),
            time=coordinates.time,
            origin=coordinates.origin,
            frame=coordinates.frame,
        ),
    )


class LeastSquares(ABC):
    """
    Deprecated: orbit refinement through `refine` (``rejection="none"``).
    """

    def __init__(self, use_central_difference: bool):
        """
        Parameters:
        -----------
        use_central_difference: bool
            Accepted for compatibility and ignored: derivatives follow the
            propagator's dynamics (``jacobian="auto"``).
        """
        warnings.warn(
            "adam_core.orbit_determination.least_squares.LeastSquares is deprecated "
            "and will be removed. It now delegates to refine(orbit, observations, "
            "propagator, rejection='none'); call refine (or run_od for the whole "
            "step) directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_central_difference = use_central_difference

    def least_squares(
        self,
        initial_orbit: FittedOrbits | Orbits,
        observations: OrbitDeterminationObservations,
        prop: Propagator,
        perturbation_initial_fraction: float = 1e-6,
        perturbation_multiplier: float = 0.5,
        rms_epsilon: float = 1e-3,
        max_iterations: float = 20,
        debug_info: Dict[str, Any] | None = None,
    ) -> Optional[Orbits]:
        """
        Refine one orbit against the RA and Dec of every observation with
        `refine` (``rejection="none"``) on ``prop``, which must have a native
        backend. Other coordinate dimensions are ignored, as before.

        ``max_iterations`` bounds the solver's iterations. The legacy finite-
        difference settings (``perturbation_initial_fraction``,
        ``perturbation_multiplier``, ``rms_epsilon``) are ignored.

        Parameters:
        -----------
        initial_orbit: FittedOrbits | Orbits (length 1)
            the initial orbit to refine
        observations: OrbitDeterminationObservations (length N)
            observations to fit (RA and Dec)
        prop: Propagator
            propagator with a native backend
        max_iterations: float, default 20
            maximum number of solver iterations
        debug_info: Dict[str, Any] | None
            if provided, populated with ``num_observations``, ``iterations``,
            ``reduced_chi2``, ``success`` and ``exit_message``

        Returns:
        --------
        The refined orbit (length 1), also when the iteration limit was
        reached, or None when the solve failed.
        """
        from .od_orchestration import refine

        refined, _ = refine(
            initial_orbit,
            _angular_only(observations),
            prop,
            rejection="none",
            fit_settings={"max_iterations": int(max_iterations)},
        )
        success = bool(refined.success[0].as_py())
        status = refined.status_code[0].as_py()
        # As the legacy loop, reaching the iteration limit still returns the
        # best orbit found; only a failed solve (negative status, non-finite
        # state) returns None.
        failed = status < 0 or not np.all(np.isfinite(refined.coordinates.values))
        if debug_info is not None:
            debug_info["num_observations"] = len(observations)
            debug_info["iterations"] = refined.iterations[0].as_py()
            debug_info["reduced_chi2"] = refined.reduced_chi2[0].as_py()
            debug_info["success"] = success
            debug_info["exit_message"] = f"refine(rejection='none') status {status}"
        if failed:
            return None
        return refined.to_orbits()
