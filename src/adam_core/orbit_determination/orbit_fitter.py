import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pyarrow as pa

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
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Initial orbit fit for a single object.

        Parameters:
        -----------
        object_id: str | pa.LargeStringScalar
            id of the object we are fitting for to be used in the output
        observations: OrbitDeterminationObservations
            observations to fit, assuming all observations correspond to the same object

        Returns:
        --------
        fitted orbit:
            one (or zero) orbits fitted to the given input observations
        fitted members:
            input observations with flags solution and outlier set based on whether the
            observation was used by the fitter. Residuals are NOT set.
        """
        pass
