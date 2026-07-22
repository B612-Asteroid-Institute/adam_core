import logging
from typing import Literal, Tuple, Type

import pyarrow as pa

from ..propagator.propagator import Propagator
from .differential_correction import iterative_fit
from .evaluate import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers, FittedOrbits
from .iod import iod
from .orbit_fitter import OrbitFitter

logger = logging.getLogger(__name__)


class NativeOrbitFitter(OrbitFitter):
    """
    Orbit fitter using adam_core's native Gauss IOD and iterative least-squares DC.

    This fitter is a thin wrapper that chains:
    1. `iod()` — Gauss initial orbit determination (Milani 2008)
    2. `iterative_fit()` — scipy least-squares differential correction with
       outlier rejection

    Parameters
    ----------
    propagator_class : Type[Propagator]
        Propagator *class* (not instance) used during IOD ephemeris evaluation.
    propagator_kwargs : dict, optional
        Keyword arguments forwarded to the propagator constructor / IOD call.
    min_obs : int, optional
        Minimum number of observations required for a valid fit.  Default 6.
    min_arc_length : float, optional
        Minimum arc length in days required to retain a fit.  Default 1.0.
    contamination_percentage : float, optional
        Maximum percentage of observations that may be rejected as outliers
        across the full OD pipeline.  Default 20.0.
    rchi2_threshold : float, optional
        Reduced chi2 convergence threshold for differential correction.
        Default 10.0.
    iod_rchi2_threshold : float, optional
        Reduced chi2 threshold used during IOD to filter candidate orbits.
        Default 200.0.
    observation_selection_method : str, optional
        Strategy for selecting observation triplets in IOD.  One of
        ``"combinations"``, ``"first+middle+last"``, ``"thirds"``.
        Default ``"combinations"``.
    """

    def __init__(
        self,
        propagator_class: Type[Propagator],
        propagator_kwargs: dict = {},
        min_obs: int = 6,
        min_arc_length: float = 1.0,
        contamination_percentage: float = 20.0,
        rchi2_threshold: float = 10.0,
        iod_rchi2_threshold: float = 200.0,
        observation_selection_method: Literal[
            "combinations", "first+middle+last", "thirds"
        ] = "combinations",
    ) -> None:
        self.propagator_class = propagator_class
        self.propagator_kwargs = propagator_kwargs
        self.min_obs = min_obs
        self.min_arc_length = min_arc_length
        self.contamination_percentage = contamination_percentage
        self.rchi2_threshold = rchi2_threshold
        self.iod_rchi2_threshold = iod_rchi2_threshold
        self.observation_selection_method = observation_selection_method

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def initial_fit(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Run Gauss IOD on the observations.

        Parameters
        ----------
        object_id : str | pa.LargeStringScalar
            Object identifier for output tables.
        observations : OrbitDeterminationObservations
            Observations to fit, assumed to belong to a single object.

        Returns
        -------
        fitted_orbit : FittedOrbits
            Best IOD orbit(s) found (may be empty if IOD fails).
        fitted_orbit_members : FittedOrbitMembers
            Observations with solution/outlier flags from IOD.
        """
        fitted_orbits, fitted_orbit_members = iod(
            observations,
            self.propagator_class,
            min_obs=self.min_obs,
            min_arc_length=self.min_arc_length,
            contamination_percentage=self.contamination_percentage,
            rchi2_threshold=self.iod_rchi2_threshold,
            observation_selection_method=self.observation_selection_method,
            propagator_kwargs=self.propagator_kwargs,
        )
        return fitted_orbits, fitted_orbit_members

    def refine_fit(
        self,
        fitted_orbit: FittedOrbits,
        observations: OrbitDeterminationObservations,
        propagator: Propagator,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Refine an IOD orbit via iterative differential correction.

        Parameters
        ----------
        fitted_orbit : FittedOrbits (1)
            Orbit to refine, typically from `initial_fit`.
        observations : OrbitDeterminationObservations
            Observations to fit against.
        propagator : Propagator
            Propagator instance used during DC ephemeris evaluation.

        Returns
        -------
        fitted_orbit : FittedOrbits (1)
            DC-refined orbit with covariance and quality statistics.
        fitted_orbit_members : FittedOrbitMembers (N)
            Observations with residuals and outlier/solution flags.
        """
        assert len(fitted_orbit) == 1, "refine_fit expects exactly one orbit"
        orbit = fitted_orbit.to_orbits()
        return iterative_fit(
            orbit,
            observations,
            propagator,
            rchi2_threshold=self.rchi2_threshold,
            min_obs=self.min_obs,
            min_arc_length=self.min_arc_length,
            contamination_percentage=self.contamination_percentage,
        )
