import logging
from abc import ABC
from typing import Any, Dict, Optional

import numpy as np

from ..coordinates.origin import Origin
from ..time import Timestamp
from ..coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    SphericalCoordinates,
)
from ..coordinates.residuals import Residuals, calculate_reduced_chi2, compute_residuals_ndarray
from ..orbits.orbits import Orbits
from ..propagator import Propagator
from .evaluate import FittedOrbits, OrbitDeterminationObservations

logger = logging.getLogger(__name__)


class LeastSquares(ABC):
    """
    EXPERIMENTAL! Orbit refinement using least squares differential correction

    In general, assume we are only fitting ra and dec in the observations. Some methods
    may work for other observation sets as well.
    """

    def __init__(self, use_central_difference: bool):
        """
        Parameters:
        -----------
        use_central_difference: bool
            if true, numerical differentiation uses central difference instead of one-sided
        """
        self._use_central_difference = use_central_difference

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
        residuals = compute_residuals_ndarray(coord1, coord2)
        return residuals[:, 1:3]

    def _compute_partials(
        self,
        delta: float,
        observations: OrbitDeterminationObservations,
        base_orbit_coordinates: CartesianCoordinates,
        nominal_ephemeris_coordinates: SphericalCoordinates,
        prop: Propagator,
        use_central_difference: bool,
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
        use_central_difference: bool
            if true, numerical differentiation uses central difference instead of one-sided

        Returns:
        --------
        (N, 2, 6) matrix of numerically computed partial derivatives
        """
        num_obs = len(observations)
        num_param = base_orbit_coordinates.values.shape[1]  # should be 6

        # Build the full set of perturbed orbits in one shot (6 or 12 rows),
        # propagate all of them through the propagator in a single batched
        # call, then split back by orbit_id to assemble the Jacobian.
        base_values = base_orbit_coordinates.values  # (1, 6)
        deltas = np.zeros((num_param, num_param), dtype=np.float64)
        for i in range(num_param):
            di = base_values[0, i] * delta
            if abs(di) < 1e-20:
                di = (1 if i < 3 else 0.01) * delta
            deltas[i, i] = di

        d_per_param = np.diag(deltas).copy()  # (num_param,)

        num_pert = num_param * (2 if use_central_difference else 1)
        pert_values = np.empty((num_pert, num_param), dtype=np.float64)
        pert_values[:num_param] = base_values + deltas
        if use_central_difference:
            pert_values[num_param:] = base_values - deltas

        pert_ids = np.array(
            [f"_ls_p_{i}" for i in range(num_param)]
            + (
                [f"_ls_m_{i}" for i in range(num_param)]
                if use_central_difference
                else []
            ),
            dtype=object,
        )

        batched_orbits = Orbits.from_kwargs(
            orbit_id=pert_ids,
            coordinates=CartesianCoordinates.from_kwargs(
                x=pert_values[:, 0],
                y=pert_values[:, 1],
                z=pert_values[:, 2],
                vx=pert_values[:, 3],
                vy=pert_values[:, 4],
                vz=pert_values[:, 5],
                time=Timestamp.from_kwargs(
                    days=np.repeat(base_orbit_coordinates.time.days[0], num_pert),
                    nanos=np.repeat(base_orbit_coordinates.time.nanos[0], num_pert),
                    scale=base_orbit_coordinates.time.scale,
                ),
                origin=Origin.from_kwargs(
                    code=np.repeat(
                        base_orbit_coordinates.origin.code[0].as_py(), num_pert
                    )
                ),
                frame=base_orbit_coordinates.frame,
            ),
        )

        ephemeris_all = prop.generate_ephemeris(
            batched_orbits,
            observations.observers,
            chunk_size=num_pert,
            max_processes=1,
        )

        A = np.zeros((num_obs, 2, num_param))
        for i in range(num_param):
            eph_p = ephemeris_all.select("orbit_id", f"_ls_p_{i}")
            if use_central_difference:
                eph_m = ephemeris_all.select("orbit_id", f"_ls_m_{i}")
                col = self._residual_columns(
                    eph_p.coordinates, eph_m.coordinates
                ) / (2 * d_per_param[i])
            else:
                col = (
                    self._residual_columns(
                        eph_p.coordinates, nominal_ephemeris_coordinates
                    )
                    / d_per_param[i]
                )
            A[:, :, i] = col
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
        perturbation_initial_fraction: float = 1e-6,
        perturbation_multiplier: float = 0.5,
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
        perturbation_initial_fraction: float, default 1e-6,
            initial fraction of parameter value to use for perturbation when computing partial derivatives
        perturbation_multiplier: float, default 0.5,
            multiplier of the perturbation fraction when RMS change overshoots
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
        assert not np.any(
            np.isnan(W_all)
        ), "Weights have NaNs, check sigmas of observations"

        perturbation_fraction = perturbation_initial_fraction

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
            one_line = {
                "rchi2": rchi2_initial.item(),
                "rms": rms_initial.item(),
                "perturbation": perturbation_fraction,
            }
            debug_info["iterations"] = [one_line]
            debug_info["corrections"] = []

        converged = False
        iteration = 0
        while not converged and iteration < max_iterations:
            logger.debug(f"RMS at iteration {iteration} is {rms_initial}")
            if rms_initial < 1e-20:
                if debug_info is not None:
                    debug_info["exit_message"] = (
                        f"RMS is zero ({rms_initial}) after {iteration} iterations"
                    )
                converged = True
                break

            iteration += 1
            # partial derivatives for all examples, d(ra|dec)/d(r|v)
            A = self._compute_partials(
                perturbation_fraction,
                observations,
                orbit_prev.coordinates,
                ephemeris_nom.coordinates,
                prop,
                self._use_central_difference,
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
                    "perturbation": perturbation_fraction,
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
                # the updated stats are worse than initial
                # if the differentiation step is not already too small, reduce the step
                # and try again from the previous orbit
                if perturbation_fraction > 1e-12:
                    perturbation_fraction *= perturbation_multiplier
                    logger.debug(f"Reducing perturbation to {perturbation_fraction}")
                    continue
                # otherwise, exit with the previous version
                if debug_info is not None:
                    debug_info["exit_message"] = (
                        "RMS is worse and perturbation is already tiny. Stopping now"
                    )
                if iteration > 1:
                    return orbit_prev
                return None

        if converged:
            return orbit_prev
        if iteration >= max_iterations:
            if debug_info is not None:
                debug_info["exit_message"] = f"Reached max iteration of {iteration}"
            return orbit_prev
        return None
