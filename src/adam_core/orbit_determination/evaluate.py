import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import quivr as qv

from ..coordinates.residuals import Residuals
from ..coordinates.spherical import SphericalCoordinates
from ..observers.observers import Observers
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from .fitted_orbits import FittedOrbitMembers, FittedOrbits


class OrbitDeterminationPhotometry(qv.Table):
    mag = qv.Float64Column(nullable=True)
    rmsmag = qv.Float64Column(nullable=True)
    band = qv.LargeStringColumn(nullable=True)


class OrbitDeterminationObservations(qv.Table):

    id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    observers = Observers.as_column()
    photometry = OrbitDeterminationPhotometry.as_column()


def evaluate_orbits(
    orbits: Union[Orbits, FittedOrbits],
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    parameters: int = 6,
    ignore: Optional[List[str]] = None,
) -> Tuple["FittedOrbits", "FittedOrbitMembers"]:
    """
    Creates a fitted orbit and fitted orbit members from input orbits and observations.
    This function takes the input orbits and calculates the residuals with respect to the observations.
    It then computes the chi2 and reduced chi2 values for the orbits. If outliers are provided, they are ignored when calculating
    the chi2, reduced chi2, and arc length values.

    This function is intended to be used to evaluate the quality of orbits with respect to observations
    in scenarios for orbit determination.

    Parameters
    ----------
    orbit : `~adam_core.orbits.Orbits` (N)
        Orbits to calculate residuals with respect to the observations for.
    observations : `~adam_core.orbit_determination.DifferentialCorrectionObservations` (M)
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
    fitted_orbit : `~adam_core.orbit_determination.FittedOrbits` (N)
        Fitted orbit.
    fitted_orbit_members : `~adam_core.orbit_determination.FittedOrbitMembers` (N * M)
        Fitted orbit members.
    """
    num_orbits = len(orbits)
    if isinstance(orbits, FittedOrbits):
        orbits = orbits.to_orbits()

    assert len(orbits) == len(orbits.orbit_id.unique())

    # Propagation remains an explicit provider boundary. The generated product
    # then enters one Rust crossing for order validation, residuals, masks,
    # orbit statistics, arc length, and member indexing.
    ephemeris = propagator.generate_ephemeris(
        orbits,
        observations.observers,
        max_processes=1,
    )
    orbits = orbits.sort_by(["orbit_id"])

    observed = observations.coordinates
    predicted = ephemeris.coordinates
    from adam_core import _rust_native

    (
        residual_values,
        residual_chi2,
        residual_dof,
        residual_probability,
        chi2,
        reduced_chi2,
        arc_length,
        num_obs,
        observation_indices,
        member_outliers,
        had_off_diagonal_nan,
    ) = _rust_native.evaluate_orbits_numpy(
        orbits.orbit_id.to_pylist(),
        ephemeris.orbit_id.to_pylist(),
        observations.id.to_pylist(),
        observed.origin.code.to_pylist(),
        predicted.origin.code.to_pylist(),
        observed.frame,
        predicted.frame,
        np.ascontiguousarray(observed.values, dtype=np.float64),
        np.ascontiguousarray(predicted.values, dtype=np.float64),
        np.ascontiguousarray(observed.covariance.to_matrix(), dtype=np.float64),
        np.ascontiguousarray(predicted.covariance.to_matrix(), dtype=np.float64),
        np.ascontiguousarray(observed.time.days.to_numpy(), dtype=np.int64),
        np.ascontiguousarray(observed.time.nanos.to_numpy(), dtype=np.int64),
        [] if ignore is None else ignore,
        parameters,
    )
    if had_off_diagonal_nan:
        warnings.warn(
            "Covariance matrix has NaNs on the off-diagonal (these will be assumed to be 0.0).",
            UserWarning,
        )

    residuals = Residuals.from_kwargs(
        values=residual_values.tolist(),
        chi2=residual_chi2,
        dof=residual_dof,
        probability=residual_probability,
    )
    fitted_orbit = FittedOrbits.from_kwargs(
        orbit_id=orbits.orbit_id,
        object_id=orbits.object_id,
        coordinates=orbits.coordinates,
        arc_length=arc_length,
        num_obs=num_obs,
        chi2=chi2,
        reduced_chi2=reduced_chi2,
    )
    fitted_orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=ephemeris.orbit_id,
        obs_id=observations.id.take(observation_indices),
        residuals=residuals,
        outlier=member_outliers,
    )

    assert len(fitted_orbit) == num_orbits
    return fitted_orbit, fitted_orbit_members
