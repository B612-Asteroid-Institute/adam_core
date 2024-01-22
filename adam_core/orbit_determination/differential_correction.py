import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from scipy.optimize import least_squares

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin
from ..coordinates.residuals import Residuals, calculate_reduced_chi2
from ..coordinates.spherical import SphericalCoordinates
from ..observers.observers import Observers
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from ..time.time import Timestamp
from .fitted_orbits import FittedOrbitMembers, FittedOrbits


class OrbitDeterminationObservations(qv.Table):

    id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    observers = Observers.as_column()


def evaluate_orbit(
    orbit: Union[Orbits, FittedOrbits],
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    parameters: int = 6,
    ignore: Optional[List[str]] = None,
) -> Tuple["FittedOrbits", "FittedOrbitMembers"]:
    """
    Creates a fitted orbit and fitted orbit members from an input orbit and observations
    believed to belong to that orbit. This function takes the input orbit and calculates
    the residuals with respect to the observations. It then computes the chi2 and reduced
    chi2 values for the orbit. If outliers are provided, they are ignored when calculating
    the chi2, reduced chi2, and arc length values.

    Parameters
    ----------
    orbit : `~adam_core.orbits.Orbits` (1)
        Orbit to calculate residuals with respect to the observations for.
    observations : `~adam_core.orbit_determination.DifferentialCorrectionObservations` (N)
        Observations believed to belong to the input orbit.
    propagator : `~adam_core.propagator.Propagator`
        Propagator to use to generate ephemeris.
    parameters : int
        Number of parameters that were initially fit to the observations. This is typically
        6 for an orbit fit to observations (assuming the epoch was not fit).
    ignore : list of str
        List of observation IDs to ignore when calculating chi2 and reduced chi2 values. This
        is typically a list of outlier observation IDs.

    Returns
    -------
    fitted_orbit : `~adam_core.orbit_determination.FittedOrbits` (1)
        Fitted orbit.
    fitted_orbit_members : `~adam_core.orbit_determination.FittedOrbitMembers` (N)
        Fitted orbit members.
    """
    if isinstance(orbit, FittedOrbits):
        orbit = orbit.to_orbits()

    # Compute ephemeris and residuals
    ephemeris = propagator.generate_ephemeris(
        orbit, observations.observers, max_processes=1
    )
    residuals = Residuals.calculate(observations.coordinates, ephemeris.coordinates)

    # If outliers are provided, we need to mask them out of the residuals and observations
    # before we compute the chi2, reduced chi2, and arc length values.
    if ignore is not None:
        mask = pc.invert(pc.is_in(observations.id, pa.array(ignore)))
        observations_to_include = observations.apply_mask(mask)
        residuals_to_include = residuals.apply_mask(mask)
    else:
        mask = pa.repeat(True, len(observations))
        observations_to_include = observations
        residuals_to_include = residuals

    # Compute arc length
    arc_length = (
        observations_to_include.coordinates.time.max().mjd()[0].as_py()
        - observations_to_include.coordinates.time.min().mjd()[0].as_py()
    )

    # Now we create a fitted orbit and fitted orbit members from the solution orbit
    # and residuals. We also need to compute the chi2 and reduced chi2 values.
    fitted_orbit = FittedOrbits.from_kwargs(
        orbit_id=orbit.orbit_id,
        object_id=orbit.object_id,
        coordinates=orbit.coordinates,
        arc_length=[arc_length],
        num_obs=[len(observations_to_include)],
        chi2=[pc.sum(residuals_to_include.chi2)],
        reduced_chi2=[calculate_reduced_chi2(residuals_to_include, parameters)],
    )
    fitted_orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=np.full(
            len(observations), fitted_orbit.orbit_id[0].as_py(), dtype="object"
        ),
        obs_id=observations.id,
        residuals=residuals,
        outlier=pc.invert(mask),
    )

    return fitted_orbit, fitted_orbit_members


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
    **kwargs
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
    fitted_orbit, fitted_orbit_members = evaluate_orbit(
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
