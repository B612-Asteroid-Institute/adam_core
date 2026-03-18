"""
FindOrb backend wrapper for
:func:`~adam_core.orbit_determination.fit_orbit.fit_orbit`.

This module wraps the ``adam_fo`` package, which provides a Python interface
to Bill Gray's Find_Orb orbit-determination program.

Installation
------------
Install the optional dependency with::

    pip install adam-fo

or, via the adam_core extras::

    pip install adam_core[find_orb]

Notes on the DELEGATE vs ADAM weighting policy
-----------------------------------------------
Find_Orb does not expose per-observation residuals through the ``adam_fo``
Python API.  When :attr:`WeightingPolicy.DELEGATE` (the default) is selected
the returned :class:`~adam_core.orbit_determination.fitted_orbits.FittedOrbitMembers`
will have ``residuals=None`` for every observation; ``chi2`` and
``reduced_chi2`` on
:class:`~adam_core.orbit_determination.fitted_orbits.FittedOrbits` will be
``NaN``.

When :attr:`WeightingPolicy.ADAM` is selected, a propagator class must be
supplied via ``BackendConfig.backend_kwargs["propagator"]``.  After Find_Orb
returns the orbit, :func:`~adam_core.orbit_determination.evaluate.evaluate_orbits`
is called to compute adam-managed residuals and chi2 values.

ADES conversion
---------------
:class:`OrbitDeterminationObservations` is converted to the ADES format that
Find_Orb expects.  The ``trkSub`` field (8-character limit in ADES) is
populated from the first observation's ``id``, truncated to 8 characters.
RA/Dec covariance sigma values are converted from degrees to arcseconds.
"""

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..config import BackendConfig, WeightingPolicy
from ..evaluate import OrbitDeterminationObservations, evaluate_orbits
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from .base import BackendWrapper

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Availability check — performed once at import time
# ---------------------------------------------------------------------------
try:
    from adam_fo import fo as _fo
    from adam_core.observations.ades import (
        ADES_to_string,
        ADESObservations,
        ObsContext,
        ObservatoryObsContext,
        SubmitterObsContext,
        TelescopeObsContext,
    )

    _FIND_ORB_AVAILABLE = True
except ImportError:
    _FIND_ORB_AVAILABLE = False


def _get_adam_fo_version() -> Optional[str]:
    """Return the installed ``adam_fo`` version string, or ``None``."""
    try:
        return version("adam-fo")
    except PackageNotFoundError:
        return None


def _observations_to_ades(
    observations: OrbitDeterminationObservations,
) -> Tuple[str, "ADESObservations"]:
    """Convert :class:`OrbitDeterminationObservations` to an ADES format string.

    Parameters
    ----------
    observations : OrbitDeterminationObservations
        Observations for a single object.  The ``id`` of the *first*
        observation is used as ``trkSub`` (truncated to 8 characters).

    Returns
    -------
    ades_string : str
        ADES-formatted text ready to be passed to ``adam_fo.fo``.
    ades_observations : ADESObservations
        Structured ADES table (returned for downstream use / logging).

    Raises
    ------
    ValueError
        If ``observations`` is empty.
    """
    if len(observations) == 0:
        raise ValueError("Cannot convert empty observations to ADES format.")

    # Use the first observation's id as trkSub (ADES 8-char limit)
    trk_sub = pc.utf8_slice_codeunits(observations.id, 0, 8)

    # Convert RA/Dec covariance sigmas from degrees to arcseconds.
    # The RA sigma must additionally be scaled by cos(Dec) to give the
    # on-sky (projected) uncertainty.
    dec_rad = np.radians(
        observations.coordinates.lat.to_numpy(zero_copy_only=False)
    )
    sigmas = observations.coordinates.covariance.sigmas  # shape (N, 6)
    # sigmas[:, 1] = RA sigma (degrees), sigmas[:, 2] = Dec sigma (degrees)
    sigma_ra_cos_dec = np.cos(dec_rad) * sigmas[:, 1]
    sigma_ra_arcsec = pa.array(sigma_ra_cos_dec * 3600.0, type=pa.float64())
    sigma_dec_arcsec = pa.array(
        sigmas[:, 2].to_numpy(zero_copy_only=False) * 3600.0, type=pa.float64()
    )

    # Replace NaN with null so ADES serialisation omits the field
    sigma_ra_arcsec = pc.if_else(pc.is_nan(sigma_ra_arcsec), None, sigma_ra_arcsec)
    sigma_dec_arcsec = pc.if_else(
        pc.is_nan(sigma_dec_arcsec), None, sigma_dec_arcsec
    )

    # Determine observatory code from coordinates.origin.code
    stn_codes = observations.coordinates.origin.code

    # Build photometry fields — may be null if photometry is absent
    phot = observations.photometry
    mag = phot.mag if phot is not None else pa.repeat(None, len(observations))
    rms_mag = (
        phot.rmsmag if phot is not None else pa.repeat(None, len(observations))
    )
    band = phot.band if phot is not None else pa.repeat(None, len(observations))

    ades_obs = ADESObservations.from_kwargs(
        trkSub=trk_sub,
        obsTime=observations.coordinates.time,
        ra=observations.coordinates.lon,
        dec=observations.coordinates.lat,
        rmsRACosDec=sigma_ra_arcsec,
        rmsDec=sigma_dec_arcsec,
        mag=mag,
        rmsMag=rms_mag,
        band=band,
        stn=stn_codes,
        mode=pa.repeat("NA", len(observations)),
        astCat=pa.repeat("NA", len(observations)),
    )

    # Build a minimal ObsContext for each unique observatory code
    unique_codes = pc.unique(stn_codes).to_pylist()
    obs_contexts = {
        code: ObsContext(
            observatory=ObservatoryObsContext(mpcCode=code),
            submitter=SubmitterObsContext(
                name="adam_core",
                institution="adam_core automated OD",
            ),
            observers=["adam_core"],
            measurers=["adam_core"],
            telescope=TelescopeObsContext(
                design="Unknown",
                aperture=1.0,
                detector="Unknown",
            ),
        )
        for code in unique_codes
    }

    ades_string = ADES_to_string(ades_obs, obs_contexts)
    return ades_string, ades_obs


class FindOrbBackend(BackendWrapper):
    """Orbit-determination backend that delegates to Find_Orb via ``adam_fo``.

    Parameters
    ----------
    out_dir : str, optional
        If provided, Find_Orb output files are copied to this directory after
        the run.  Useful for debugging.

    Examples
    --------
    >>> from adam_core.orbit_determination.backends import FindOrbBackend
    >>> backend = FindOrbBackend()
    >>> fitted, members = backend.fit(observations, config)
    """

    AVAILABLE: bool = _FIND_ORB_AVAILABLE
    BACKEND_NAME: str = "find_orb"

    def fit(
        self,
        observations: OrbitDeterminationObservations,
        config: BackendConfig,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Fit an orbit using Find_Orb.

        Parameters
        ----------
        observations : OrbitDeterminationObservations
            Observations for a single object.
        config : BackendConfig
            Runtime configuration.  Relevant ``backend_kwargs`` keys:

            * ``"out_dir"`` *(str)* – directory where Find_Orb writes its
              output files (forwarded to ``adam_fo.fo``).
            * ``"propagator"`` *(Type[Propagator])* – required when
              ``weighting_policy`` is :attr:`WeightingPolicy.ADAM`.

        Returns
        -------
        fitted_orbits : FittedOrbits
            Orbit returned by Find_Orb.  ``chi2`` and ``reduced_chi2`` are
            ``NaN`` in :attr:`WeightingPolicy.DELEGATE` mode.
        fitted_orbit_members : FittedOrbitMembers
            Per-observation membership table.  ``residuals`` is ``None`` in
            :attr:`WeightingPolicy.DELEGATE` mode.

        Raises
        ------
        ImportError
            If ``adam_fo`` is not installed.
        RuntimeError
            If Find_Orb returns an error or produces no orbit.
        ValueError
            If the output cannot be parsed.
        """
        self._check_available()

        out_dir: Optional[str] = config.backend_kwargs.get("out_dir", None)

        logger.info(
            "FindOrbBackend: fitting orbit for %d observations.", len(observations)
        )

        # --- Convert to ADES --------------------------------------------------
        ades_string, ades_obs = _observations_to_ades(observations)

        # --- Invoke Find_Orb --------------------------------------------------
        orbit_raw, rejected_raw, error = _fo(
            ades_string,
            out_dir=out_dir,
            clean_up=(out_dir is None),
        )

        if error is not None or len(orbit_raw) == 0:
            raise RuntimeError(
                f"Find_Orb failed to produce an orbit. "
                f"Error message: {error!r}. "
                f"Ensure the find_orb binary is installed correctly "
                f"('build-fo' command from adam-fo) and that the observations "
                f"have valid covariances and observer states."
            )

        logger.info("FindOrbBackend: Find_Orb returned %d orbit(s).", len(orbit_raw))

        # --- Determine which obs were rejected --------------------------------
        # rejected_raw is an ADESObservations table; match by time+code.
        rejected_times = set()
        if len(rejected_raw) > 0:
            for i in range(len(rejected_raw)):
                mjd = rejected_raw.obsTime.mjd()[i].as_py()
                code = rejected_raw.stn[i].as_py()
                rejected_times.add((round(mjd, 8), code))

        obs_mjds = observations.coordinates.time.mjd().to_pylist()
        obs_codes = observations.coordinates.origin.code.to_pylist()
        outlier_flags = pa.array(
            [
                (round(m, 8), c) in rejected_times
                for m, c in zip(obs_mjds, obs_codes)
            ],
            type=pa.bool_(),
        )

        # --- Build provenance -------------------------------------------------
        backend_version = _get_adam_fo_version()

        # --- Apply weighting policy -------------------------------------------
        if config.weighting_policy is WeightingPolicy.ADAM:
            propagator_cls = config.backend_kwargs.get("propagator")
            if propagator_cls is None:
                raise ValueError(
                    "WeightingPolicy.ADAM requires a 'propagator' key in "
                    "BackendConfig.backend_kwargs.  "
                    "Example: BackendConfig(weighting_policy=WeightingPolicy.ADAM, "
                    "backend_kwargs={'propagator': ASSISTPropagator})"
                )
            logger.info(
                "FindOrbBackend: computing adam-managed residuals with %s.",
                propagator_cls.__name__,
            )
            fitted_orbits, fitted_members = evaluate_orbits(
                orbit_raw,
                observations,
                propagator_cls(),
            )
            # Stamp provenance
            fitted_orbits = fitted_orbits.set_column(
                "backend",
                pa.repeat(self.BACKEND_NAME, len(fitted_orbits)),
            )
            fitted_orbits = fitted_orbits.set_column(
                "backend_version",
                pa.repeat(backend_version, len(fitted_orbits)),
            )
            # Preserve the outlier flags from Find_Orb
            fitted_members = FittedOrbitMembers.from_kwargs(
                orbit_id=fitted_members.orbit_id,
                obs_id=fitted_members.obs_id,
                residuals=fitted_members.residuals,
                solution=pc.invert(outlier_flags),
                outlier=outlier_flags,
            )
            return fitted_orbits, fitted_members

        # DELEGATE mode: build minimal FittedOrbits without chi2/residuals
        orbit_id = orbit_raw.orbit_id[0].as_py()
        num_inliers = int(pc.sum(pc.invert(outlier_flags)).as_py())
        arc_days = (
            observations.coordinates.time.max().mjd()[0].as_py()
            - observations.coordinates.time.min().mjd()[0].as_py()
        )

        fitted_orbits = FittedOrbits.from_kwargs(
            orbit_id=[orbit_id],
            object_id=orbit_raw.object_id,
            coordinates=orbit_raw.coordinates,
            arc_length=[arc_days],
            num_obs=[num_inliers],
            chi2=[float("nan")],
            reduced_chi2=[float("nan")],
            iterations=pa.repeat(None, 1),
            success=[True],
            status_code=[0],
            backend=pa.repeat(self.BACKEND_NAME, 1),
            backend_version=pa.repeat(backend_version, 1),
        )

        fitted_members = FittedOrbitMembers.from_kwargs(
            orbit_id=pa.repeat(orbit_id, len(observations)),
            obs_id=observations.id,
            residuals=None,
            solution=pc.invert(outlier_flags),
            outlier=outlier_flags,
        )

        return fitted_orbits, fitted_members
