from typing import List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from mpcq import MPCObservations

from ..coordinates import CoordinateCovariances, SphericalCoordinates
from ..coordinates.origin import Origin
from ..coordinates.residuals import Residuals, calculate_reduced_chi2
from ..observers.observers import Observers
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from .fitted_orbits import FittedOrbitMembers, FittedOrbits


class OrbitDeterminationObservations(qv.Table):

    id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    observers = Observers.as_column()


def mpc_to_od_observations(
    obs_set: MPCObservations,
) -> OrbitDeterminationObservations | None:
    """Converts MPC observations into OD observations.
    Returns None if the input set is malformed, e.g. has NULLs in the set of STN codes.
    """
    obs_time = obs_set.obstime
    codes = obs_set.stn
    if not np.all(codes):
        print(
            f"STN codes for {obs_set.requested_provid.unique().to_pylist()} include nulls"
        )
        return None

    # `mpcq`'s `MPCObservations` includes uncertainty columns:
    # - rmsra, rmsdec, rmscorr (and rmsmag)
    #
    # These RMS values come from the MPC database and are in arcseconds. By ADES/MPC convention,
    # `rmsra` is RA uncertainty *cos(dec). We convert into degrees and back out RA sigma
    # (so that downstream `Residuals` can apply its own cos(latitude) scaling consistently).
    dec_deg = obs_set.dec.to_numpy(zero_copy_only=False)
    cos_dec = np.cos(np.deg2rad(dec_deg))

    sigma_ra_cosdec_deg = obs_set.rmsra.to_numpy(zero_copy_only=False) / 3600.0
    sigma_dec_deg = obs_set.rmsdec.to_numpy(zero_copy_only=False) / 3600.0
    sigma_ra_deg = np.where(
        np.isfinite(cos_dec) & (cos_dec != 0.0),
        sigma_ra_cosdec_deg / cos_dec,
        np.nan,
    )

    # Include RA/Dec correlation if present; treat missing correlation as 0 (uncorrelated).
    corr = obs_set.rmscorr.to_numpy(zero_copy_only=False)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    cov = np.full((len(obs_set), 6, 6), np.nan, dtype=np.float64)
    # Prevent 'Covariance matrix has NaNs on the diagonal' and 'Singular matrix
    cov[:, 1, 1] = np.nan_to_num(sigma_ra_deg**2, nan=1.0e-9)
    cov[:, 2, 2] = np.nan_to_num(sigma_dec_deg**2, nan=1.0e-9)
    cov[:, 1, 2] = corr * sigma_ra_deg * sigma_dec_deg
    cov[:, 2, 1] = cov[:, 1, 2]
    # Prevents 'UserWarning: Covariance matrix has NaNs on the off-diagonal (these will be assumed to be 0.0).'
    cov = np.nan_to_num(cov)

    coords = SphericalCoordinates.from_kwargs(
        lon=obs_set.ra.to_numpy(zero_copy_only=False),
        lat=obs_set.dec.to_numpy(zero_copy_only=False),
        time=obs_time,
        origin=Origin.from_kwargs(code=codes),
        frame="equatorial",
        covariance=CoordinateCovariances.from_matrix(cov),
    )

    observers = Observers.from_codes(codes=codes, times=obs_time)

    od_observations = OrbitDeterminationObservations.from_kwargs(
        id=obs_set.obsid.to_numpy(zero_copy_only=False),
        coordinates=coords,
        observers=observers,
    )
    return od_observations


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
