from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from typing_extensions import Self

from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin
from ..coordinates.residuals import Residuals, calculate_reduced_chi2
from ..coordinates.spherical import SphericalCoordinates
from ..observations.ades import ADESObservations
from ..observers.observers import Observers
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from .fitted_orbits import FittedOrbitMembers, FittedOrbits

if TYPE_CHECKING:
    # Imported lazily to avoid a circular import: observation_uncertainty
    # imports OrbitDeterminationObservations from this module.
    from .observation_uncertainty import ObservationUncertaintyModel


class OrbitDeterminationPhotometry(qv.Table):
    mag = qv.Float64Column(nullable=True)
    rmsmag = qv.Float64Column(nullable=True)
    band = qv.LargeStringColumn(nullable=True)


class OrbitDeterminationObservations(qv.Table):
    """
    Astrometric observations prepared for orbit determination.

    Columns
    -------
    id : str
        Unique observation identifier. Used to track outliers and to match
        `FittedOrbitMembers` back to their observations.
    coordinates : `~adam_core.coordinates.SphericalCoordinates`
        Topocentric equatorial astrometry (lon = RA, lat = Dec, degrees) with
        the per-observation uncertainty carried in the covariance. The lon
        variance is the variance of RA itself (not RA * cos(dec)).
    observers : `~adam_core.observers.Observers`
        Observer state (station code + heliocentric state) at each observation time.
    photometry : `OrbitDeterminationPhotometry`
        Optional magnitude, magnitude uncertainty and band.
    astcat : str, nullable
        Star catalog the astrometry was reduced against, carried verbatim as the
        MPC/ADES ``astCat`` code (e.g. "Gaia2", "UCAC4", "PPMXL"). It is a
        top-level column, parallel to ``observers.code`` (the station) and
        ``photometry.band`` (the filter), rather than a field of a sub-table:
        the reference catalog is an attribute of how each astrometric position
        was reduced, not of the observer or the photometry, and catalog-aware
        models (e.g. star-catalog debiasing, per-(station, catalog) uncertainty
        models) key on it per row alongside the position and time. It is
        nullable so that observations assembled without catalog metadata keep
        working unchanged; a null means "catalog unknown" and catalog-aware
        models must pass such rows through untouched. No normalization is
        applied to the code.
    """

    id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    observers = Observers.as_column()
    photometry = OrbitDeterminationPhotometry.as_column()
    astcat = qv.LargeStringColumn(nullable=True)

    @classmethod
    def from_ades(
        cls,
        ades: ADESObservations,
        ids: Optional[Union[List[str], npt.NDArray[np.str_], pa.Array]] = None,
    ) -> Self:
        """
        Build orbit determination observations from ADES observations.

        The mapping is:

        - ``ra``, ``dec`` -> ``coordinates.lon``, ``coordinates.lat`` (degrees),
          with ``coordinates.time`` = ``obsTime`` and ``coordinates.origin`` = ``stn``.
        - ``rmsRACosDec``, ``rmsDec`` (arcseconds) -> covariance of lon and lat in
          degrees^2. ADES reports the RA uncertainty scaled by cos(dec), so the lon
          sigma is ``rmsRACosDec / cos(dec)``. ``rmsCorr`` (dimensionless, nullable)
          is used as the RA/Dec correlation; a null correlation is treated as 0.
          Null ``rmsRACosDec``/``rmsDec`` produce NaN variances, i.e. no
          uncertainty is invented here; apply an
          `~adam_core.orbit_determination.ObservationUncertaintyModel` or fill the
          sigmas before fitting.
        - ``stn``, ``obsTime`` -> ``observers`` via `Observers.from_codes`.
        - ``mag``, ``rmsMag``, ``band`` -> ``photometry``.
        - ``astCat`` -> ``astcat`` (verbatim, no normalization).

        Parameters
        ----------
        ades : `~adam_core.observations.ADESObservations` (N)
            ADES observations to convert.
        ids : list, array or pyarrow array of str (N), optional
            Observation IDs to assign. If None, ``obsSubID`` is used when it is
            populated for every row; otherwise the zero-based row index is used
            (as a string). IDs must be unique.

        Returns
        -------
        observations : `OrbitDeterminationObservations` (N)
            Observations ready for orbit determination, in the same order as the input.
        """
        num_obs = len(ades)
        if num_obs == 0:
            return cls.empty()

        if ids is None:
            if ades.obsSubID.null_count == 0:
                ids_array = pa.chunked_array([ades.obsSubID]).combine_chunks()
            else:
                ids_array = pa.array(
                    [str(i) for i in range(num_obs)], type=pa.large_string()
                )
        elif isinstance(ids, (pa.Array, pa.ChunkedArray)):
            ids_array = pa.chunked_array([ids]).combine_chunks()
        else:
            ids_array = pa.array(list(ids), type=pa.large_string())
        ids_array = pc.cast(ids_array, pa.large_string())
        if len(ids_array) != num_obs:
            raise ValueError(
                f"Expected {num_obs} observation IDs, got {len(ids_array)}."
            )
        if ids_array.null_count > 0:
            raise ValueError("Observation IDs must not be null.")
        if pc.count_distinct(ids_array).as_py() != num_obs:
            raise ValueError("Observation IDs must be unique.")

        ra = ades.ra.to_numpy(zero_copy_only=False)
        dec = ades.dec.to_numpy(zero_copy_only=False)
        # Nulls become NaN on conversion to numpy for float columns
        rms_ra_cosdec = ades.rmsRACosDec.to_numpy(zero_copy_only=False)
        rms_dec = ades.rmsDec.to_numpy(zero_copy_only=False)
        corr = ades.rmsCorr.to_numpy(zero_copy_only=False)
        corr = np.where(np.isfinite(corr), corr, 0.0)

        # ADES uncertainties are in arcseconds and the RA uncertainty is
        # scaled by cos(dec); convert to degrees on the raw lon/lat angles.
        cos_dec = np.cos(np.radians(dec))
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_lon = np.where(
                cos_dec != 0.0, rms_ra_cosdec / 3600.0 / cos_dec, np.nan
            )
        sigma_lat = rms_dec / 3600.0

        covariance_matrix = np.full((num_obs, 6, 6), np.nan, dtype=np.float64)
        covariance_matrix[:, 1, 1] = sigma_lon**2
        covariance_matrix[:, 2, 2] = sigma_lat**2
        covariance_matrix[:, 1, 2] = corr * sigma_lon * sigma_lat
        covariance_matrix[:, 2, 1] = covariance_matrix[:, 1, 2]

        coordinates = SphericalCoordinates.from_kwargs(
            lon=ra,
            lat=dec,
            time=ades.obsTime,
            covariance=CoordinateCovariances.from_matrix(covariance_matrix),
            origin=Origin.from_kwargs(code=ades.stn),
            frame="equatorial",
        )
        observers = Observers.from_codes(ades.stn, ades.obsTime)
        photometry = OrbitDeterminationPhotometry.from_kwargs(
            mag=ades.mag,
            rmsmag=ades.rmsMag,
            band=ades.band,
        )
        return cls.from_kwargs(
            id=ids_array,
            coordinates=coordinates,
            observers=observers,
            photometry=photometry,
            astcat=ades.astCat,
        )


def evaluate_orbits(
    orbits: Union[Orbits, FittedOrbits],
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    parameters: int = 6,
    ignore: Optional[List[str]] = None,
    observatory_bias_model: Optional["ObservationUncertaintyModel"] = None,
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
    observatory_bias_model : `~adam_core.orbit_determination.ObservationUncertaintyModel`, optional
        Observation uncertainty model applied to the observations before evaluation
        (e.g. inflating per-station sigmas from an observatory bias table). Default
        None leaves the observations unchanged. The model is applied once at this
        entry point and is not forwarded to nested calls.

    Returns
    -------
    fitted_orbit : `~adam_core.orbit_determination.FittedOrbits` (N)
        Fitted orbit.
    fitted_orbit_members : `~adam_core.orbit_determination.FittedOrbitMembers` (N * M)
        Fitted orbit members.
    """
    if observatory_bias_model is not None:
        observations = observatory_bias_model.apply(observations)

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
