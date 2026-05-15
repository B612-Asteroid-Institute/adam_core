from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
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


def _mask_to_numpy(mask: pa.BooleanArray | pa.ChunkedArray) -> npt.NDArray[np.bool_]:
    return np.asarray(mask.to_numpy(zero_copy_only=False), dtype=np.bool_)


def _repeat_observation_indices(
    num_orbits: int, num_observations: int
) -> npt.NDArray[np.int64]:
    observation_indices = np.arange(num_observations, dtype=np.int64)
    return np.tile(observation_indices, num_orbits)


def _validate_ephemeris_orbit_order(
    ephemeris_orbit_ids: pa.Array | pa.ChunkedArray,
    orbit_ids: pa.Array | pa.ChunkedArray,
    num_observations: int,
) -> None:
    expected = np.repeat(
        orbit_ids.to_numpy(zero_copy_only=False),
        num_observations,
    )
    actual = ephemeris_orbit_ids.to_numpy(zero_copy_only=False)
    if np.array_equal(actual, expected):
        return

    raise ValueError(
        "Ephemeris rows must be grouped by sorted orbit_id with one block per "
        "observation; this is the documented Propagator.generate_ephemeris order."
    )


def _calculate_orbit_statistics(
    residuals: Residuals,
    num_orbits: int,
    num_observations: int,
    include_mask: npt.NDArray[np.bool_],
    parameters: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    expected_residuals = num_orbits * num_observations
    if len(residuals) != expected_residuals:
        raise ValueError(
            f"Expected {expected_residuals} residual rows, got {len(residuals)}."
        )

    chi2_rows = residuals.chi2.to_numpy(zero_copy_only=False).reshape(
        num_orbits, num_observations
    )
    dof_rows = residuals.dof.to_numpy(zero_copy_only=False).reshape(
        num_orbits, num_observations
    )

    chi2 = chi2_rows[:, include_mask].sum(axis=1)
    dof = dof_rows[:, include_mask].sum(axis=1) - parameters
    reduced_chi2 = chi2 / dof
    return chi2, reduced_chi2


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
    num_observations = len(observations)
    _validate_ephemeris_orbit_order(
        ephemeris.orbit_id,
        orbits.orbit_id,
        num_observations,
    )

    # Stack the observation coordinates logically by taking the observation rows
    # in orbit-major order. This avoids constructing full repeated observation
    # tables before residual calculation.
    observation_indices = _repeat_observation_indices(num_orbits, num_observations)
    observation_indices_arrow = pa.array(observation_indices)
    residuals = Residuals.calculate(
        observations.coordinates.take(observation_indices), ephemeris.coordinates
    )

    # If outliers are provided, we need to mask them out of the observations
    # before we compute the chi2, reduced chi2, and arc length values.
    if ignore is not None:
        mask = pc.invert(pc.is_in(observations.id, pa.array(ignore)))
        observations_to_include = observations.apply_mask(mask)
    else:
        mask = pa.repeat(True, len(observations))
        observations_to_include = observations
    include_mask = _mask_to_numpy(mask)

    # Compute number of observations
    num_obs = len(observations_to_include)

    # Compute chi2 and reduced chi2 for each orbit by reshaping the residual rows
    # into the documented Propagator output order: (orbit_id, observation).
    chi2, reduced_chi2 = _calculate_orbit_statistics(
        residuals,
        num_orbits,
        num_observations,
        include_mask,
        parameters,
    )

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
        obs_id=pc.take(observations.id, observation_indices_arrow),
        residuals=residuals,
        outlier=np.tile(~include_mask, num_orbits),
    )

    return fitted_orbit, fitted_orbit_members
