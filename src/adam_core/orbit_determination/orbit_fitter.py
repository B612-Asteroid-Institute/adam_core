import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pyarrow as pa

from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from .evaluate import FittedOrbitMembers, FittedOrbits, OrbitDeterminationObservations

logger = logging.getLogger(__name__)


class OrbitFitter(ABC):
    """
    Abstract class for orbit fitting.
    """

    def __getstate__(self):
        """
        Get the state of the orbit fitter.

        Subclasses need to define what is picklable for multiprocessing.

        e.g.

        def __getstate__(self):
            state = self.__dict__.copy()
            state.pop("_stateful_attribute_that_is_not_pickleable")
            return state
        """
        raise NotImplementedError(
            "OrbitFitter must implement __getstate__ for multiprocessing serialization."
        )

    def __setstate__(self, state):
        """
        Set the state of the orbit fitter.

        Subclasses need to define what is unpicklable for multiprocessing.

        e.g.

        def __setstate__(self, state):
            self.__dict__.update(state)
            self._stateful_attribute_that_is_not_pickleable = None
        """
        raise NotImplementedError(
            "OrbitFitter must implement __setstate__ for multiprocessing serialization."
        )

    @abstractmethod
    def initial_fit(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
        reference_orbit: Optional[Orbits] = None,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Initial orbit fit for a single object.

        Parameters:
        -----------
        object_id: str | pa.LargeStringScalar
            id of the object we are fitting for to be used in the output
        observations: OrbitDeterminationObservations
            observations to fit, assuming all observations correspond to the same object
        reference_orbit: Orbits, optional
            Optional warm-start seed (length-1 Orbits) for the fit. When provided,
            implementations should use it as the starting point for differential
            correction instead of cold-bootstrapping via IOD. When ``None``,
            implementations must fall back to their default cold-start behavior
            (e.g. Gauss IOD) so the API remains backward-compatible for callers
            that have no seed available.

        Returns:
        --------
        fitted orbit:
            one (or zero) orbits fitted to the given input observations
        fitted members:
            input observations with flags solution and outlier set based on whether the
            observation was used by the fitter. Residuals are NOT set.
        """
        pass

    def refine_fit(
        self,
        fitted_orbit: FittedOrbits,
        observations: OrbitDeterminationObservations,
        propagator: Propagator,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Refine an existing orbit fit using differential correction.

        Takes a previously fitted orbit (e.g. from IOD) and improves it
        via iterative least-squares with outlier rejection.

        Fitters that support differential correction override this method.
        The default raises `NotImplementedError` (like `__getstate__` /
        `__setstate__` above) rather than being abstract, so that fitters
        written against the original ``initial_fit``-only interface, such as
        plugin releases predating this method, remain instantiable and usable
        for `initial_fit`. `full_od` and `~adam_core.orbit_determination.run_od`
        require an override.

        Parameters
        ----------
        fitted_orbit : FittedOrbits (1)
            Orbit to refine, typically output from `initial_fit`.
        observations : OrbitDeterminationObservations
            Observations to fit against.
        propagator : Propagator
            Propagator used to generate ephemeris during DC.

        Returns
        -------
        fitted_orbit : FittedOrbits (1)
            Refined orbit with covariance and quality statistics.
        fitted_orbit_members : FittedOrbitMembers (N)
            Observations with residuals and outlier/solution flags set.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement refine_fit; only "
            "initial_fit is available. Implement refine_fit to use full_od / "
            "run_od with this fitter."
        )

    def full_od(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
        propagator: Propagator,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Run full orbit determination: IOD followed by differential correction.

        This default implementation chains `initial_fit` and `refine_fit`.
        Subclasses may override for backend-specific behaviour (e.g. FindOrb's
        built-in DC pipeline).

        Parameters
        ----------
        object_id : str | pa.LargeStringScalar
            Object identifier used in output tables.
        observations : OrbitDeterminationObservations
            All observations for this object.
        propagator : Propagator
            Propagator used during differential correction.

        Returns
        -------
        fitted_orbit : FittedOrbits (1)
            Best orbit found across IOD + DC.
        fitted_orbit_members : FittedOrbitMembers (N)
            Observations with residuals and outlier/solution flags.

        Notes
        -----
        Observation uncertainty handling (e.g. an observatory bias model)
        deliberately happens ABOVE this interface: fitter implementations,
        including external plugins, are passed observations whose
        uncertainties have already been transformed. Apply a model upstream,
        e.g. ``fitter.full_od(object_id, model.apply(observations),
        propagator)``, use `~adam_core.orbit_determination.run_od`, or use the
        module-level OD functions which accept ``observatory_bias_model``
        directly.
        """
        iod_orbit, iod_members = self.initial_fit(object_id, observations)
        if len(iod_orbit) == 0:
            return iod_orbit, iod_members
        return self.refine_fit(iod_orbit, observations, propagator)
