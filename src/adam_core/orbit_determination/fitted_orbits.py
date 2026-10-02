import uuid
from typing import List, Literal, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from typing_extensions import Self

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
    import numpy as np

    from adam_core import _rust_native

    # One Rust crossing owns the priority ordering, per-observation best-orbit
    # selection, member filtering, and surviving-orbit selection.
    orbit_take, member_keep = _rust_native.assign_duplicate_observations_numpy(
        orbits.orbit_id.to_pylist(),
        np.ascontiguousarray(
            orbits.num_obs.to_numpy(zero_copy_only=False), dtype=np.int64
        ),
        np.ascontiguousarray(
            orbits.arc_length.to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        np.ascontiguousarray(
            orbits.reduced_chi2.to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        orbit_members.orbit_id.to_pylist(),
        orbit_members.obs_id.to_pylist(),
    )

    filtered_orbits = orbits.take(pa.array(orbit_take, type=pa.int64()))
    orbit_members = orbit_members.apply_mask(pa.array(member_keep))

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

    The columns are:

    ``lon``, ``lat`` : float, degrees
        Observed RA / Dec, in the same convention as
        `OrbitDeterminationObservations.coordinates` (lon = RA, lat = Dec).
    ``sigma_lon``, ``sigma_lat`` : float, degrees
        1-sigma uncertainties, the square roots of the lon / lat covariance
        diagonal. ``sigma_lon`` is the uncertainty of RA itself (NOT
        cos(dec)-corrected), matching the `SphericalCoordinates` covariance
        convention. NaN where the covariance is not available.
    ``cov_lonlat`` : float, degrees^2
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

        from adam_core import _rust_native

        covariances = np.ascontiguousarray(
            coordinates.covariance.to_matrix(), dtype=np.float64
        )
        sigma_lon = _rust_native.sqrt_values_numpy(
            np.ascontiguousarray(covariances[:, 1, 1])
        )
        sigma_lat = _rust_native.sqrt_values_numpy(
            np.ascontiguousarray(covariances[:, 2, 2])
        )
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
