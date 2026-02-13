from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from mpcq import MPCObservations

from . import mpc_to_od_observations
from .evaluate import FittedOrbitMembers, FittedOrbits, OrbitDeterminationObservations


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

    def initial_fit_all_objects(
        self, mpc_observations: MPCObservations
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """
        Split the MPC observation set by object id and fit orbits for all objects.

        Returns
        -------
        fitted orbits: one entry per object, where an orbit fit was successful
        fitted members: one entry for each input observation with flags for `solution` and `outlier` set based
            on the output of this specific orbit fitter algorithm. Residuals are NOT computed, so if residuals
            are desired, run `evaluate_orbits` separately.

        May skip over objects with malformed inputs or failed fits.
        TODO: do we want to allow it to fail on malformed instead?

        TODO: consider processing individual sets in parallel

        """
        orbits = []
        members = []
        object_ids = mpc_observations.requested_provid.unique()
        for id in object_ids:
            subset = mpc_observations.apply_mask(
                pc.equal(mpc_observations.requested_provid, id)
            )
            if not np.all(subset.stn):
                print(f"Skipping object {id.as_py()} that has null STNs")
                continue
            # Keep NaNs in the covariance matrix of the observations, since some of the fitter
            # algorithms work with them. Note that computing residuals would fail if NaNs are
            # present, which is why we are not computing residuals in the output fitted members.
            orbit, mems = self.initial_fit(
                id, mpc_to_od_observations(subset, prevent_nans=False)
            )
            if orbit is None or len(orbit) == 0:
                # Assume the error is reported in initial_fit, so just skip here
                continue
            orbits.append(orbit)
            members.append(mems)
        return qv.concatenate(orbits), qv.concatenate(members)
