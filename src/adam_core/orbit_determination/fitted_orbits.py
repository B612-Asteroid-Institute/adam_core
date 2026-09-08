import uuid
from typing import List, Literal, Optional, Self, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.residuals import Residuals
from ..coordinates.spherical import SphericalCoordinates
from ..orbits.orbits import Orbits


def assign_duplicate_observations(
    orbits: "FittedOrbits", orbit_members: "FittedOrbitMembers"
) -> Tuple["FittedOrbits", "FittedOrbitMembers"]:
    """
    Assigns observations that have been assigned to multiple orbits to the orbit with the
    most observations, longest arc length, and lowest reduced chi2.

    Parameters
    ----------
    orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members.

    Returns
    -------
    filtered : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits with duplicate assignments removed.
    filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members with duplicate assignments removed.
    """
    # Sorting by priority criteria
    orbits = orbits.sort_by(
        [
            ("num_obs", "descending"),
            ("arc_length", "descending"),
            ("reduced_chi2", "ascending"),
        ]
    )

    # Extracting unique observation IDs
    unique_obs_ids = pc.unique(orbit_members.column("obs_id"))

    # Dictionary to store the best orbit for each observation
    best_orbit_for_obs = {}

    # Iterate over each unique observation ID
    for obs_id in unique_obs_ids:
        # Filter orbit_members for the current observation ID
        mask = pc.equal(orbit_members.column("obs_id"), obs_id)
        current_obs_members = orbit_members.where(mask)

        # Extract orbit IDs that this observation belongs to
        obs_orbit_ids = current_obs_members.column("orbit_id")

        # Find the best orbit for this observation based on the criteria
        for sorted_orbit_id in orbits.column("orbit_id"):
            if pc.any(pc.is_in(sorted_orbit_id, value_set=obs_orbit_ids)).as_py():
                best_orbit_for_obs[obs_id.as_py()] = sorted_orbit_id.as_py()
                break

    # Iteratively update orbit_members to drop rows where obs_id is the same,
    # but orbit_id is not the best orbit_id for that observation
    for obs_id, best_orbit_id in best_orbit_for_obs.items():
        mask_to_remove = pc.and_(
            pc.equal(orbit_members.column("obs_id"), pa.scalar(obs_id)),
            pc.not_equal(orbit_members.column("orbit_id"), pa.scalar(best_orbit_id)),
        )
        orbit_members = orbit_members.apply_mask(pc.invert(mask_to_remove))

    # Filtering self based on the filtered orbit_members
    orbits_mask = pc.is_in(
        orbits.column("orbit_id"), value_set=orbit_members.column("orbit_id")
    )
    filtered_orbits = orbits.apply_mask(orbits_mask)

    return filtered_orbits, orbit_members


def drop_duplicate_orbits(
    orbits: "FittedOrbits",
    orbit_members: "FittedOrbitMembers",
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last"] = "first",
) -> Tuple["FittedOrbits", "FittedOrbitMembers"]:
    """
    Drop duplicate orbits from the fitted orbits and remove
    the corresponding orbit members.

    Parameters
    ----------
    orbits : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits.
    orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members.
    subset : list of str, optional
        Subset of columns to consider when dropping duplicates. If not specified all the columns
        specifying unique state are used: time, x, y, z, vx, vy, vz.
    keep : {'first', 'last'}, default 'first'
        If there are duplicate rows then keep the first or last row.

    Returns
    -------
    filtered : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits without duplicates.
    filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members without duplicates.
    """
    if subset is None:
        subset = [
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.x",
            "coordinates.y",
            "coordinates.z",
            "coordinates.vx",
            "coordinates.vy",
            "coordinates.vz",
        ]

    filtered = orbits.drop_duplicates(subset=subset, keep=keep)
    filtered_orbit_members = orbit_members.apply_mask(
        pc.is_in(orbit_members.orbit_id, filtered.orbit_id)
    )
    return filtered, filtered_orbit_members


class FittedOrbits(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    chi2 = qv.Float64Column()
    reduced_chi2 = qv.Float64Column()
    iterations = qv.Int64Column(nullable=True)
    success = qv.BooleanColumn(nullable=True)
    status_code = qv.Int64Column(nullable=True)

    def to_orbits(self) -> Orbits:
        """
        Convert fitted orbits to orbits that can be used by
        a Propagator.

        Returns
        -------
        orbits : `~adam_core.orbits.Orbits`
            Orbits.
        """
        return Orbits.from_kwargs(
            orbit_id=self.orbit_id,
            object_id=self.object_id,
            coordinates=self.coordinates,
        )


class ObservationAstrometry(qv.Table):
    """
    Snapshot of one observation's astrometric position and uncertainty.

    `FittedOrbitMembers` embeds two of these per observation: the astrometry as
    ORIGINALLY supplied to orbit determination (``original_astrometry``) and
    the astrometry actually USED by the fitter (``used_astrometry``) after any
    observation models (star-catalog debiasing, uncertainty inflation,
    deweighting) were applied at fit time by
    `~adam_core.orbit_determination.run_od`. Values are embedded directly so
    that fitted orbit members are self-contained provenance: the observations
    the fit was based on can be inspected (and the model's effect audited)
    without re-joining to the input observations or re-running the models.

    Columns
    -------
    lon, lat : float, degrees
        Observed RA / Dec, in the same convention as
        `OrbitDeterminationObservations.coordinates` (lon = RA, lat = Dec).
    sigma_lon, sigma_lat : float, degrees
        1-sigma uncertainties, the square roots of the lon / lat covariance
        diagonal. ``sigma_lon`` is the uncertainty of RA itself (NOT
        cos(dec)-corrected), matching the `SphericalCoordinates` covariance
        convention. NaN where the covariance is not available.
    cov_lonlat : float, degrees^2
        RA/Dec covariance cross-term. NaN where not available.

    This table deliberately carries no attributes (e.g. a frame): members with
    and without provenance therefore concatenate freely, which a nested
    `SphericalCoordinates` column would not allow.
    """

    lon = qv.Float64Column(nullable=True)
    lat = qv.Float64Column(nullable=True)
    sigma_lon = qv.Float64Column(nullable=True)
    sigma_lat = qv.Float64Column(nullable=True)
    cov_lonlat = qv.Float64Column(nullable=True)

    @classmethod
    def from_spherical(cls, coordinates: SphericalCoordinates) -> Self:
        """
        Snapshot the lon / lat position and RA/Dec covariance block of
        spherical coordinates.

        Parameters
        ----------
        coordinates : `~adam_core.coordinates.SphericalCoordinates` (N)
            Coordinates to snapshot, typically
            ``OrbitDeterminationObservations.coordinates``.

        Returns
        -------
        astrometry : `ObservationAstrometry` (N)
            Position and 1-sigma uncertainties in degrees, in input order.
        """
        if len(coordinates) == 0:
            return cls.empty()

        covariances = coordinates.covariance.to_matrix()
        with np.errstate(invalid="ignore"):
            sigma_lon = np.sqrt(covariances[:, 1, 1])
            sigma_lat = np.sqrt(covariances[:, 2, 2])
        return cls.from_kwargs(
            lon=coordinates.lon,
            lat=coordinates.lat,
            sigma_lon=sigma_lon,
            sigma_lat=sigma_lat,
            cov_lonlat=covariances[:, 1, 2],
        )


class FittedOrbitMembers(qv.Table):
    """
    Per-observation record of an orbit fit.

    Columns
    -------
    orbit_id, obs_id : str
        Fitted orbit and observation identifiers (``obs_id`` matches
        `OrbitDeterminationObservations.id`).
    residuals : `~adam_core.coordinates.Residuals`, nullable
        Observed-minus-computed residuals of the fitted orbit.
    solution, outlier : bool, nullable
        Whether the observation constrained the solution / was rejected.
    weight : float, nullable
        Effective weight of the observation in the solution, as reported by
        the fitter: 0 for observations excluded from the fit, 1 for fully
        weighted observations and, for a robust loss (e.g.
        ``fit_least_squares(loss="huber")``), the smaller of the observation's
        per-component iteratively-reweighted-least-squares weights when it
        was downweighted. Null when the fitter does not report weights.
    original_astrometry : `ObservationAstrometry`, nullable
        Position and uncertainty of the observation as ORIGINALLY supplied.
    used_astrometry : `ObservationAstrometry`, nullable
        Position and uncertainty actually USED by the fitter, i.e. after the
        observation models applied at fit time by
        `~adam_core.orbit_determination.run_od`. Equal to
        ``original_astrometry`` when no model changed the observation.
    astcat : str, nullable
        Star catalog the observation was reduced against (see
        `OrbitDeterminationObservations.astcat`).

    The provenance columns (``original_astrometry``, ``used_astrometry``,
    ``astcat``) are populated by `run_od`. Fitters called directly leave them
    null; members with and without provenance share one schema and can be
    concatenated, and members serialized before these columns existed load
    with the columns null.
    """

    orbit_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()
    residuals = Residuals.as_column(nullable=True)
    solution = qv.BooleanColumn(nullable=True)
    outlier = qv.BooleanColumn(nullable=True)
    weight = qv.Float64Column(nullable=True)
    original_astrometry = ObservationAstrometry.as_column(nullable=True)
    used_astrometry = ObservationAstrometry.as_column(nullable=True)
    astcat = qv.LargeStringColumn(nullable=True)
