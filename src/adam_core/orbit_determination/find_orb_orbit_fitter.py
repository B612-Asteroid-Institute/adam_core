import json
import logging
from typing import Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from adam_fo import fo
from mpc_obscodes import mpc_obscodes

from ..observations.ades import (
    ADES_to_string,
    ADESObservations,
    ObsContext,
    ObservatoryObsContext,
    SubmitterObsContext,
    TelescopeObsContext,
)
from .evaluate import FittedOrbitMembers, FittedOrbits, OrbitDeterminationObservations
from .orbit_fitter import OrbitFitter

logger = logging.getLogger(__name__)


class FindOrbOrbitFitter(OrbitFitter):
    """Implementation of OrbitFitter using Find_Orb."""

    def __init__(
        self,
        *args: object,  # Generic type for arbitrary positional arguments
        fo_result_dir: str,
        **kwargs: object,  # Generic type for arbitrary keyword arguments
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fo_result_dir = fo_result_dir
        with open(mpc_obscodes) as mpc_file:
            self.obscodes = json.load(mpc_file)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("obscodes")
        return state

    def __setstate__(self, state):
        saved_obscodes = self.obscodes
        self.__dict__.update(state)
        self.obscodes = saved_obscodes

    def _short_observation_id(self, obs_id: str) -> str:
        """Convert obs_id from MPC observation to string matching `packed` field of total.json."""
        return obs_id[0:4] + obs_id[-4:]

    def _observations_to_ades(
        self, observations: OrbitDeterminationObservations
    ) -> Tuple[str, ADESObservations]:
        """
        Convert OrbitDeterminationObservations to ADES format string.

        Parameters
        ----------
        observations : OrbitDeterminationObservations
            Observations to convert

        Returns
        -------
        Tuple[str, ADESObservations]
            Tuple containing:
            - ADES format string
            - ADESObservations object
        """
        # "packed" field in total.json ends with this, but that field is currently not exposed
        # so this is more for future extensions
        orbit_ids = [self._short_observation_id(id.as_py()) for id in observations.id]

        # Convert uncertainties in RA and Dec to arcseconds
        # and adjust the RA uncertainty to account for the cosine of the declination
        # The convesion matches MPC values if mpc_to_od_observations was called with prevent_nans=False
        sigma_ra_cos_dec = (
            np.cos(
                np.radians(observations.coordinates.lat.to_numpy(zero_copy_only=False))
            )
            * observations.coordinates.covariance.sigmas[:, 1]
        )
        sigma_ra_cos_dec_arcseconds = pa.array(
            sigma_ra_cos_dec * 3600, type=pa.float64()
        )
        sigma_dec_arcseconds = pa.array(
            observations.coordinates.covariance.sigmas[:, 2] * 3600, type=pa.float64()
        )

        # Replace nans with nulls using pyarrow
        sigma_ra_cos_dec_arcseconds = pc.if_else(
            pc.is_nan(sigma_ra_cos_dec_arcseconds), None, sigma_ra_cos_dec_arcseconds
        )
        sigma_dec_arcseconds = pc.if_else(
            pc.is_nan(sigma_dec_arcseconds), None, sigma_dec_arcseconds
        )

        # Serialize observations to an ADES table
        ades_observations = ADESObservations.from_kwargs(
            trkSub=orbit_ids,
            obsTime=observations.coordinates.time,
            ra=observations.coordinates.lon,
            dec=observations.coordinates.lat,
            rmsRACosDec=sigma_ra_cos_dec_arcseconds,
            rmsDec=sigma_dec_arcseconds,
            mag=observations.photometry.mag,
            rmsMag=observations.photometry.rmsmag,
            band=observations.photometry.band,
            stn=observations.observers.code,
            mode=pa.repeat("NA", len(observations)),
            astCat=pa.repeat("NA", len(observations)),
        )

        codes = [v.as_py() for v in pc.unique(observations.observers.code)]

        telescope = TelescopeObsContext(
            name="Thing",
            design="Reflector",
            detector="CCD",
            aperture=40.0,
        )

        obs_contexts = {
            code: ObsContext(
                observatory=ObservatoryObsContext(
                    mpcCode=code, name=self.obscodes[code]["Name"]
                ),
                submitter=SubmitterObsContext(
                    name="J. Doe",
                    institution="B612 Asteroid Institute",
                ),
                observers=["J. Doe"],
                measurers=["J. Doe"],
                telescope=telescope,
            )
            for code in codes
        }

        ades_string = ADES_to_string(ades_observations, obs_contexts)
        return ades_string, ades_observations

    def _rejected_observations_to_fitted_members(
        self,
        observations: OrbitDeterminationObservations,
        rejected: ADESObservations,
        orbit_id: str,
    ) -> FittedOrbitMembers:
        """
        Convert a set of observations rejected by Find_Orb into fitted members.

        Parameters
        ----------
        observations: OrbitDeterminationObservations (N)
           All observations used as input to the fitter
        rejected: ADESObservations (M)
           Observations rejected by the fitter. 0<=M<=N
        orbit_id: str
           orbit_id to be used in the output fitted members

        Returns
        -------
        Set of fitted members corresponding to all input observations. The solution and outlier
        flags are set based on whether the corresponding observation is found in the rejected list.
        Residuals are NOT computed, because doing so would require covariance matrix in observations
        to have no NaNs.
        """

        assert len(observations) >= len(rejected)
        obs_ids_all = observations.id
        rejected_ids = []
        for ades in rejected:
            # Observation id is not passed fully in the ADES format, so we need to match on fields.
            # Look for an observation from the same station within 1 second. There should be exactly 1
            same_station = pc.equal(observations.observers.code, ades.stn[0])
            same_time = pc.less(
                pc.abs(
                    pc.subtract(
                        observations.coordinates.time.mjd(), ades.obsTime.mjd()[0]
                    )
                ),
                1.0 / 86400,
            )
            original = observations.apply_mask(pc.and_(same_station, same_time))
            assert (
                len(original) == 1
            ), f"Expected 1 input observation for {ades.stn[0]} at {ades.obsTime.to_iso8601()[0]}, got {original.observers.code} {original.coordinates.time.mjd()}"
            rejected_ids.append(original.id[0].as_py())

        # To calculate residuals we need covariance matrix without NaNs, but
        # normal fitting prefers one with NaNs, so we can't compute residuals
        # here. We'll leave them null for now. Call evaluate_orbits later
        # to get residuals.

        outlier = np.isin(obs_ids_all, rejected_ids)
        assert np.sum(outlier) == len(
            rejected
        ), "Something failed in extracting rejected observations"

        od_orbit_members = FittedOrbitMembers.from_kwargs(
            orbit_id=np.full(len(obs_ids_all), orbit_id, dtype="object"),
            obs_id=obs_ids_all,
            # not setting residuals here
            solution=np.isin(obs_ids_all, rejected_ids, invert=True),
            outlier=outlier,
        )

        return od_orbit_members

    def initial_fit(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        if observations is None or len(observations) == 0:
            logger.error(f"No observation provided for object {object_id}")
            return FittedOrbits.empty(), FittedOrbitMembers.empty()
        logger.info(
            f"Initial propagation using Find_Orb for {object_id} using {len(observations)} observations"
        )
        ades_string, _ = self._observations_to_ades(observations)

        orbit, rejected, error = fo(
            ades_string,
            out_dir=self.fo_result_dir,
            clean_up=False,
        )
        if error is not None:
            logger.error(f"FindOrb failed for object {object_id} with error {error}")
            return FittedOrbits.empty(), FittedOrbitMembers.empty()

        N = len(orbit)
        if isinstance(object_id, str):
            object_id = pa.scalar(object_id, type=pa.large_string())
        orbit = orbit.set_column("object_id", [object_id] * N)
        if len(rejected) > 0:
            logger.info(
                f"Find_Orb rejected {len(rejected)} input observations out of {len(observations)}"
            )
        orbit_id = orbit.orbit_id[0].as_py()
        fitted_members = self._rejected_observations_to_fitted_members(
            observations, rejected, orbit_id
        )
        return orbit, fitted_members
