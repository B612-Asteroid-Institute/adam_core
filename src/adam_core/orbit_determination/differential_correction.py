import warnings
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from scipy.optimize import least_squares

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin
from ..coordinates.residuals import Residuals
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from ..time.time import Timestamp
from .evaluate import OrbitDeterminationObservations, evaluate_orbits
from .fitted_orbits import FittedOrbitMembers, FittedOrbits
from .outliers import calculate_max_outliers, remove_lowest_probability_observation


def residual_function(
    state_vector: npt.NDArray[np.float64],
    mjd_tdb: float,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
) -> npt.NDArray[np.float64]:
    """
    Compute the residuals for a given Cartesian orbit with respect to a
    a set of observations.

    Parameters
    ----------
    state_vector : `~numpy.ndarray` (6,)
        State vector of the orbit in Cartesian coordinates.
    mjd_tdb : float
        Time of the state vector in MJD TDB.
    observations : `~thor.orbit_determination.DifferentialCorrectionObservations` (N)
        Observations to compute residuals for.
    propagator : `~thor.propagator.Propagator`
        Propagator to use to generate ephemeris.

    Returns
    -------
    residuals : `~numpy.ndarray` (N,)
        Weighted residual (chi) for each observation.
    """
    # Generate ephemeris and compute residuals
    orbit = Orbits.from_kwargs(
        coordinates=CartesianCoordinates.from_kwargs(
            x=state_vector[0:1],
            y=state_vector[1:2],
            z=state_vector[2:3],
            vx=state_vector[3:4],
            vy=state_vector[4:5],
            vz=state_vector[5:6],
            time=Timestamp.from_mjd([mjd_tdb], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        )
    )
    ephemeris = propagator.generate_ephemeris(
        orbit, observations.observers, max_processes=1
    )
    residuals = Residuals.calculate(observations.coordinates, ephemeris.coordinates)

    # We return the "chi2" value from adam_core.coordinates.residuals. This is equivalent to
    # the mahalanobis distance squared (which is just the squared residual normalized by the
    # uncertainty on the observations). This effectively allows us to the weight the
    # observations (and their corresponding residuals) by their uncertainty.
    return np.sqrt(residuals.chi2.to_numpy())


def fit_least_squares(
    orbit: Orbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    ignore: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Differentially correct a fitted orbit using least squares. correct a fitted orbit using least squares.

    Parameters
    ----------
    orbit : `~adam_core.orbits.Orbits` (1)
        Orbit to differentially correct.
    observations : `~adam_core.orbit_determination.DifferentialCorrectionObservations` (N)
        Observations.
    propagator : `~adam_core.propagator.Propagator`
        Propagator to use to generate ephemeris.
    ignore : list of str
        List of observation IDs to ignore when fitting the orbit with least squares.
        These observations will be marked as outliers in the fitted orbit members.
    **kwargs
        Additional keyword arguments to pass to `~scipy.optimize.least_squares`.
        Some of these parameters if not specified will be set to sensible defaults.
            xtol = 1e-12
            ftol = 1e-12
            gtol = 1e-12
            jac = "2-point"
            x_scale = "jac"
            bounds = (-np.inf, np.inf) (for each parameter)

    Returns
    -------
    fitted_orbit : `~adam_core.orbit_determination.FittedOrbits` (1)
        Fitted orbit.
    fitted_orbit_members : `~adam_core.orbit_determination.FittedOrbitMembers` (N)
        Fitted orbit members.
    """
    assert len(orbit) == 1, "Only one orbit can be differentially corrected"

    # TODO: Investigate whether we want to add fitting for the epoch as well
    # Set up least squares problem
    if ignore is not None:
        mask = pc.invert(pc.is_in(observations.id, pa.array(ignore)))
        observations_to_include = observations.apply_mask(mask)
    else:
        observations_to_include = observations

    parameters = 6
    # Extract epoch and state vector from orbit
    epoch = orbit.coordinates.time.mjd().to_numpy()
    state_vector = orbit.coordinates.values[0]
    args = (epoch[0], observations_to_include, propagator)

    # Define some sensible defaults for the least squares fitting procedure
    if "xtol" not in kwargs:
        kwargs["xtol"] = 1e-12
    if "ftol" not in kwargs:
        kwargs["ftol"] = 1e-12
    if "gtol" not in kwargs:
        kwargs["gtol"] = 1e-12
    if "jac" not in kwargs:
        kwargs["jac"] = "2-point"
    if "x_scale" not in kwargs:
        kwargs["x_scale"] = "jac"
    if "bounds" not in kwargs:
        kwargs["bounds"] = (
            np.full(parameters, -np.inf),
            np.full(parameters, np.inf),
        )
    if "args" in kwargs:
        kwargs.pop("args")
        warnings.warn(
            "The args parameter is not supported and will be ignored.",
            category=RuntimeWarning,
        )

    # Run least squares
    solution = least_squares(residual_function, state_vector, args=args, **kwargs)

    # Extract solution state vector and covariance matrix
    mjd_tdb = epoch[0]
    x, y, z, vx, vy, vz = solution.x
    try:
        covariance_matrix = np.linalg.inv(solution.jac.T @ solution.jac)
    except np.linalg.LinAlgError:
        warnings.warn(
            "The covariance matrix could not be computed. The solution may be "
            "unreliable.",
            category=RuntimeWarning,
        )
        covariance_matrix = np.full((6, 6), np.nan)

    # Create orbit with solution state vector and use it to generate ephemeris
    # and calculate the residuals with respect to the observations
    orbit = Orbits.from_kwargs(
        orbit_id=orbit.orbit_id,
        object_id=orbit.object_id,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[x],
            y=[y],
            z=[z],
            vx=[vx],
            vy=[vy],
            vz=[vz],
            time=Timestamp.from_mjd([mjd_tdb], scale="tdb"),
            covariance=CoordinateCovariances.from_matrix(
                covariance_matrix.reshape(1, 6, 6)
            ),
            origin=orbit.coordinates.origin,
            frame=orbit.coordinates.frame,
        ),
    )

    # Evaluate the solution orbit and return it as a fitted orbit and fitted orbit members
    # which contain the residuals with respect to the observations and the overall
    # quality of the fit
    fitted_orbit, fitted_orbit_members = evaluate_orbits(
        orbit,
        observations,
        propagator,
        parameters=parameters,
        ignore=ignore,
    )
    fitted_orbit = (
        fitted_orbit.set_column("iterations", [solution.nfev])
        .set_column("success", [solution.success])
        .set_column("status_code", [solution.status])
    )
    fitted_orbit_members = fitted_orbit_members.set_column(
        "solution", pc.invert(fitted_orbit_members.outlier)
    )

    return fitted_orbit, fitted_orbit_members


def iterative_fit(
    orbit: Orbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    rchi2_threshold: float = 10.0,
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 20.0,
    **kwargs,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Iteratively fit an orbit using least squares with outlier rejection.

    Wraps `fit_least_squares` with an outlier rejection loop: after each fit,
    if the reduced chi2 exceeds `rchi2_threshold`, the observation with the
    worst residual is removed and the fit is repeated. This continues until
    the fit converges, no more outliers are allowed, or arc length / minimum
    observation constraints would be violated.

    Parameters
    ----------
    orbit : `~adam_core.orbits.Orbits` (1)
        Initial orbit to differentially correct.
    observations : `~adam_core.orbit_determination.OrbitDeterminationObservations` (N)
        Observations to fit against.
    propagator : `~adam_core.propagator.Propagator`
        Propagator to use to generate ephemeris.
    rchi2_threshold : float, optional
        Reduced chi2 threshold below which the fit is considered converged.
        Default is 10.0.
    min_obs : int, optional
        Minimum number of observations required to retain the fit.
        Default is 6.
    min_arc_length : float, optional
        Minimum arc length in days required to retain the fit.
        Default is 1.0.
    contamination_percentage : float, optional
        Maximum percentage of observations that may be rejected as outliers.
        Range is [0, 100]. Default is 20.0.
    **kwargs
        Additional keyword arguments passed to `fit_least_squares` and
        ultimately to `~scipy.optimize.least_squares`.

    Returns
    -------
    fitted_orbit : `~adam_core.orbit_determination.FittedOrbits` (1)
        Best fitted orbit found.
    fitted_orbit_members : `~adam_core.orbit_determination.FittedOrbitMembers` (N)
        Fitted orbit members with residuals and outlier flags.
    """
    assert len(orbit) == 1, "Only one orbit can be iteratively fitted"

    num_obs = len(observations)
    max_outliers = calculate_max_outliers(num_obs, min_obs, contamination_percentage)

    ignore: List[str] = []
    best_fitted_orbit = None
    best_fitted_orbit_members = None

    for _ in range(max_outliers + 1):
        fitted_orbit, fitted_orbit_members = fit_least_squares(
            orbit,
            observations,
            propagator,
            ignore=ignore if ignore else None,
            **kwargs,
        )

        # Track the best fit seen so far (lowest reduced chi2 among successful fits)
        if best_fitted_orbit is None or (
            fitted_orbit.success[0].as_py()
            and fitted_orbit.reduced_chi2[0].as_py()
            < best_fitted_orbit.reduced_chi2[0].as_py()
        ):
            best_fitted_orbit = fitted_orbit
            best_fitted_orbit_members = fitted_orbit_members

        # Check convergence
        rchi2 = fitted_orbit.reduced_chi2[0].as_py()
        if rchi2 is not None and rchi2 <= rchi2_threshold:
            break

        # Stop if we've already used up all allowed outlier slots
        if len(ignore) >= max_outliers:
            break

        # Identify the worst non-outlier observation among the current solution members
        solution_members = fitted_orbit_members.apply_mask(
            pc.equal(fitted_orbit_members.outlier, False)
        )
        if len(solution_members) == 0:
            break

        obs_id, remaining_observations = remove_lowest_probability_observation(
            solution_members, observations
        )

        # Check that removing this observation still leaves enough arc length
        arc_length = remaining_observations.coordinates.time.mjd().to_numpy()
        if len(arc_length) < min_obs or (arc_length.max() - arc_length.min()) < min_arc_length:
            break

        ignore.append(obs_id)

    return best_fitted_orbit, best_fitted_orbit_members
