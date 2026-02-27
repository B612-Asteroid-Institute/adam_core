import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from mpcq import MPCObservations

from ..coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    SphericalCoordinates,
)
from ..coordinates.residuals import Residuals, calculate_reduced_chi2
from ..orbits.orbits import Orbits
from ..propagator import Propagator
from . import mpc_to_od_observations
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
                logger.warning(f"Skipping object {id.as_py()} that has null STNs")
                continue
            logger.info(f"Object {id}, num observations {len(subset)}")
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

    ### Parts for orbit refinement
    # In general, assume we are only fitting ra and dec in the observations. Some methods
    # may work for other observation sets as well.

    def _update_orbit(
        self,
        base_coords: CartesianCoordinates,
        delta: np.ndarray,
        covariance_matrix: Optional[np.ndarray] = None,
    ) -> Orbits:
        """
        Create an orbit that differs from base orbit by delta in the state vector.

        Parameters:
        -----------
        base_coords: CartesianCoordinates
            state vector of the base orbit
        delta: np.ndarray (1, 6)
            deltas for the state vector
        covariance_matrix: Optional[np.ndarray] (6,6)
            if provided, used as covariance for the state vector of the new orbit;
            otherwise the covariance of the base_coords is copied over

        Returns:
        --------
        New Orbits (size 1)
        """
        cartesian_elements_p = base_coords.values + delta

        if covariance_matrix is not None:
            covariance = CoordinateCovariances.from_matrix(
                covariance_matrix.reshape(1, 6, 6)
            )
        else:
            covariance = base_coords.covariance
        return Orbits.from_kwargs(
            coordinates=CartesianCoordinates.from_kwargs(
                x=cartesian_elements_p[:, 0],
                y=cartesian_elements_p[:, 1],
                z=cartesian_elements_p[:, 2],
                vx=cartesian_elements_p[:, 3],
                vy=cartesian_elements_p[:, 4],
                vz=cartesian_elements_p[:, 5],
                covariance=covariance,
                time=base_coords.time,
                origin=base_coords.origin,
                frame=base_coords.frame,
            ),
        )

    def _residual_columns(
        self, coord1: SphericalCoordinates, coord2: SphericalCoordinates
    ) -> np.ndarray:
        """
        Compute residuals for the RA and DEC between the two sets of coordinates
        Parameters:
        -----------
        coords1, coords2: SphericalCoordinates
            the two sets of coordinates, length N, for which to compute residuals

        Returns:
        --------
        (N, 2) array of residuals for RA and DEC.
        """
        residuals = Residuals.calculate(coord1, coord2)
        residuals = np.stack(residuals.values.to_numpy(zero_copy_only=False))
        # Pull ra and dec columns for all observations, so Nx2
        return residuals[:, 1:3]

    def _compute_partials(
        self,
        delta: float,
        observations: OrbitDeterminationObservations,
        base_orbit_coordinates: CartesianCoordinates,
        nominal_ephemeris_coordinates: SphericalCoordinates,
        prop: Propagator,
    ) -> np.ndarray:
        """
        Compute the matrix of all partial derivatives for all observations.
        Assume we observe only RA and DEC, but tweak the state vector, so derivatives
        are d(ra|dec)/d(r|v), i.e. 2x6 matrix for each observation.

        Parameters:
        -----------
        delta: float
            fraction of the original state vector value to use for perturbation
        observations: OrbitDeterminationObservations (length N)
            observations to compute the derivatives for
        base_orbit_coordinates: CartesianCoordinates (length 1)
            the state vector of the orbit to perturb
        nominal_ephemeris_coordinates: SphericalCoordinates (length N)
            coordinates for the unperturbed orbit, so that we don't have to recompute them
        prop: Propagator
            the propagator to use to compute perturbed ephemeris

        Returns:
        --------
        (N, 2, 6) matrix of numerically computed partial derivatives

        TODO: maybe add central difference as well?
        """
        num_obs = len(observations)
        num_param = base_orbit_coordinates.values.shape[1]  # should be 6

        A = np.zeros((num_obs, 2, num_param))
        for i in range(num_param):
            # Perturb the orbit in one coordinate
            d = np.zeros((1, num_param))
            d[0, i] = base_orbit_coordinates.values[0, i] * delta
            orbit_iter_p = self._update_orbit(base_orbit_coordinates[0], d)
            # Calculate the modified ephemerides
            ephemeris_mod_p = prop.generate_ephemeris(
                orbit_iter_p, observations.observers, chunk_size=1, max_processes=1
            )
            col = self._residual_columns(
                ephemeris_mod_p.coordinates, nominal_ephemeris_coordinates
            )
            A[:, :, i] = col / d[0, i]
        return A

    def _rms(self, residuals: np.ndarray, weights: np.ndarray) -> float:
        """
        Compute RMS as sqrt( b.T * W * b / (n*N)), per Vallado.

        Parameters:
        -----------
        residuals: np.ndarray
            matrix b of residuals, size (N, m), where N is the number of observations and m
            is the number observed values. For RA and DEC observations m=2, but this method is
            agnostic to the value of m.
        weights: np.ndarray
            weights to use, (N, m). The weights are usually 1/sigma^2 from the observations. They
            are independents of the orbit.
        """
        N, m = residuals.shape
        return np.sqrt(np.sum(residuals**2 * weights) / (N * m))

    def least_squares(
        self,
        initial_orbit: FittedOrbits | Orbits,
        observations: OrbitDeterminationObservations,
        prop: Propagator,
        rms_epsilon: float = 1e-3,
        max_iterations: float = 20,
        debug_info: Dict[str, Any] | None = None,
    ) -> Optional[Orbits]:
        """
        Run least squares to refine the initial orbit.
        Assumes observations and ephemeris are in Spherical coordinates and only have RA and DEC,
        and we have only one orbit. Follows Vallado.

        Parameters:
        -----------
        initial_orbit: FittedOrbits | Orbits (length 1)
            the initial orbit to refine
        observations: OrbitDeterminationObservations (length N)
            observations to use for refinement, use RA and DEC only
        prop: Propagator
            propagator to use for ephemerides of perturbed orbits
        rms_epsilon: float, default 1e-3
            limit for relative change of RMS between iteration to declare convergence
        max_iterations: float, default 20
            maximum number of iterations before giving up
        debug_info: Dict[str, Any] | None
            if provided, this dictionary is populated with debug information
        Returns:
        --------
        Improved orbit (length 1) or None, if no improvement was found.

        """
        num_obs = len(observations)
        num_param = initial_orbit.coordinates.values.shape[1]
        if debug_info is not None:
            debug_info["num_observations"] = num_obs
        orbit_prev = initial_orbit

        # Weights depend only on observations, not the orbit, so pull them out once.
        # Note that we assume we only have RA and DEC here.
        W_all = observations.coordinates.covariance.sigmas[:, 1:3]
        W_all = 1 / W_all**2
        assert W_all.shape == (num_obs, 2), f"W_all shape {W_all.shape}"

        # These will change between iterations, but we'll reuse already computed ones, so get this before the loop
        ephemeris_nom = prop.generate_ephemeris(
            orbit_prev, observations.observers, chunk_size=1, max_processes=1
        )
        B = self._residual_columns(observations.coordinates, ephemeris_nom.coordinates)
        rms_initial = self._rms(B, W_all)
        if debug_info is not None:
            rchi2_initial = calculate_reduced_chi2(
                Residuals.calculate(
                    observations.coordinates, ephemeris_nom.coordinates
                ),
                num_param,
            )
            one_line = {"rchi2": rchi2_initial.item(), "rms": rms_initial.item()}
            debug_info["iterations"] = [one_line]
            debug_info["corrections"] = []

        converged = False
        iteration = 0
        while not converged and iteration < max_iterations:
            iteration += 1
            # partial derivatives for all examples, d(ra|dec)/d(r|v)
            A = self._compute_partials(
                1e-6,
                observations,
                orbit_prev.coordinates,
                ephemeris_nom.coordinates,
                prop,
            )

            # Accumulators for the matrices, see Vallado
            ATWA = np.zeros((num_param, num_param))  # params x params
            ATWb = np.zeros((num_param,))  # params x 1
            for i in range(len(observations)):
                W = np.diag(W_all[i, :])
                Ai = A[i, :, :]  # Ai is d(ra|dec) / d(r|v), so 2x6
                b = B[i, :]
                AtW = Ai.T @ W  # (6, 2) * (2, 2) -> (6, 2)
                ATWA += AtW @ Ai  # (6,2)*(2,6) -> (6, 6)
                ATWb += AtW @ b  # (6,2)*(2,) -> (6,)

            # AKA P, AKA covariance matrix, diagonal has squares of sigma_rI, sigma_rJ, ..., sigma_vK
            # Eigenvalues and eigenvectors would give error ellipse dimensions and orientation
            AtWA1 = np.linalg.inv(ATWA)
            corrections = AtWA1 @ ATWb
            logger.debug(
                f"Iteration {iteration}\ninitial {initial_orbit.coordinates.values}\nupdate {corrections}"
            )

            # Eval
            updated_orbit = self._update_orbit(
                orbit_prev.coordinates[0], corrections, AtWA1
            )
            ephemeris_updated = prop.generate_ephemeris(
                updated_orbit, observations.observers, chunk_size=1, max_processes=1
            )
            B_updated = self._residual_columns(
                observations.coordinates, ephemeris_updated.coordinates
            )
            rms_updated = self._rms(B_updated, W_all)
            delta_rms = (rms_initial - rms_updated) / rms_initial
            converged = np.abs(delta_rms) < rms_epsilon
            logger.debug(
                f"RMS old={rms_initial}, new {rms_updated}, change {delta_rms}, converged {converged}"
            )

            if debug_info is not None:
                rchi2_updated = calculate_reduced_chi2(
                    Residuals.calculate(
                        observations.coordinates, ephemeris_updated.coordinates
                    ),
                    num_param,
                )
                one_line = {
                    "rchi2": rchi2_updated.item(),
                    "rms": rms_updated.item(),
                    "delta_rms": delta_rms.item(),
                    "converged": converged.item(),
                }
                debug_info["iterations"].append(one_line)
                debug_info["corrections"].append(corrections.tolist())

            updated_orbit = updated_orbit.set_column(
                "object_id", initial_orbit.object_id
            ).set_column("orbit_id", initial_orbit.orbit_id)

            # Reuse computed values for the next iteration
            if rms_updated < rms_initial:
                orbit_prev = updated_orbit
                ephemeris_nom = ephemeris_updated
                B = B_updated
                rms_initial = rms_updated
            elif not converged:
                # if updated is worse than initial, exit with the previous version
                if debug_info is not None:
                    debug_info["exit_message"] = "RMS is worse. Stopping now"
                if iteration > 1:
                    return orbit_prev
                return None

        if converged:
            return orbit_prev
        return None
