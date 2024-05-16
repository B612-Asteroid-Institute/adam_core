from typing import List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.residuals import Residuals, calculate_reduced_chi2
from ..coordinates.spherical import SphericalCoordinates
from ..observers.observers import Observers
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from .fitted_orbits import FittedOrbitMembers, FittedOrbits


class OrbitDeterminationObservations(qv.Table):

    id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    observers = Observers.as_column()


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

    # Compute ephemeris and residuals
    ephemeris = propagator.generate_ephemeris(
        orbits,
        observations.observers,
        max_processes=1,
    )

    # Sort the orbits by ID (the ephemeris is already sorted by
    # orbit ID, time, and origin)
    orbits = orbits.sort_by(["orbit_id"])

    # Stack the observations into a single table and compute the residuals with respect
    # to the predicted coordinates
    observations_stacked = qv.concatenate([observations for _ in range(num_orbits)])
    residuals = Residuals.calculate(
        observations_stacked.coordinates, ephemeris.coordinates
    )

    # If outliers are provided, we need to mask them out of the observations
    # before we compute the chi2, reduced chi2, and arc length values.
    if ignore is not None:
        mask = pc.invert(pc.is_in(observations.id, pa.array(ignore)))
        observations_to_include = observations.apply_mask(mask)
    else:
        mask = pa.repeat(True, len(observations))
        observations_to_include = observations

    # Compute number of observations
    num_obs = len(observations_to_include)

    # Compute chi2 and reduced chi2 for each orbit
    chi2 = np.empty(num_orbits, dtype=np.float64)
    reduced_chi2 = np.empty(num_orbits, dtype=np.float64)
    for i, orbit_id in enumerate(orbits.orbit_id):
        orbit_mask = pc.equal(ephemeris.orbit_id, orbit_id)
        residuals_to_include = residuals.apply_mask(orbit_mask).apply_mask(mask)
        chi2[i] = pc.sum(residuals_to_include.chi2).as_py()
        reduced_chi2[i] = calculate_reduced_chi2(residuals_to_include, parameters)

    # Compute arc length for the orbits (will be the same for all orbits)
    arc_length = (
        observations_to_include.coordinates.time.max().mjd()[0].as_py()
        - observations_to_include.coordinates.time.min().mjd()[0].as_py()
    )

    # Now we create a fitted orbit and fitted orbit members from the solution orbit
    # and residuals. We also need to compute the chi2 and reduced chi2 values.
    fitted_orbit = FittedOrbits.from_kwargs(
        orbit_id=orbits.orbit_id,
        object_id=orbits.object_id,
        coordinates=orbits.coordinates,
        arc_length=pa.repeat(arc_length, num_orbits),
        num_obs=pa.repeat(num_obs, num_orbits),
        chi2=chi2,
        reduced_chi2=reduced_chi2,
    )
    fitted_orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=ephemeris.orbit_id,
        obs_id=observations_stacked.id,
        residuals=residuals,
        outlier=pa.concat_arrays([pc.invert(mask) for i in range(num_orbits)]),
    )

    return fitted_orbit, fitted_orbit_members
