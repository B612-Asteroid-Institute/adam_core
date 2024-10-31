import os

import numpy as np
import pyarrow.compute as pc
import pytest

from adam_core.propagator.adam_assist import ASSISTPropagator

from ...coordinates import CoordinateCovariances, SphericalCoordinates
from ...coordinates.origin import Origin
from ...observers import Observers
from ...utils.helpers.observations import make_observations
from ...utils.helpers.orbits import make_real_orbits
from ..evaluate import OrbitDeterminationObservations
from ..iod import iod


@pytest.fixture
def real_data():
    # Generate real observations and orbits
    exposures, detections, associations = make_observations()
    orbits = make_real_orbits(num_orbits=1)

    # Select a specific object ID for testing
    object_id = orbits.object_id[0].as_py()
    orbit = orbits.select("object_id", object_id)

    # Filter observations for the selected object ID
    associations_i = associations.select("object_id", object_id)
    detections_i = detections.apply_mask(
        pc.is_in(detections.id, associations_i.detection_id)
    )

    exposures_i = exposures.apply_mask(pc.is_in(exposures.id, detections_i.exposure_id))

    sigmas = np.full((len(detections_i.ra_sigma), 6), np.nan)
    sigmas[:, 1] = detections_i.ra_sigma.to_numpy(zero_copy_only=False) / 3600
    sigmas[:, 2] = detections_i.dec_sigma.to_numpy(zero_copy_only=False) / 3600

    coordinates = SphericalCoordinates.from_kwargs(
        lon=detections_i.ra.to_numpy(),
        lat=detections_i.dec.to_numpy(),
        covariance=CoordinateCovariances.from_sigmas(sigmas),
        origin=Origin.from_kwargs(code=exposures_i.observatory_code),
        frame="equatorial",  # Assuming the frame is equatorial
    )

    # Generate Observers from exposures start_time and observatory codes
    observers = Observers.from_codes(
        times=exposures_i.start_time, codes=exposures_i.observatory_code
    )

    observations = OrbitDeterminationObservations.from_kwargs(
        id=detections_i.id.to_numpy(zero_copy_only=False),
        coordinates=coordinates,
        observers=observers,
    )

    return orbit, observations


@pytest.mark.skipif(
    os.environ.get("ASSIST_DATA_DIR") is None,
    reason="ASSIST_DATA_DIR environment variable not set",
)
@pytest.mark.skipif(ASSISTPropagator is None, reason="ASSISTPropagator not available")
def test_iod(real_data):
    orbit, observations = real_data

    # Call the iod function
    fitted_orbits, fitted_orbit_members = iod(
        observations,
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=200,
        observation_selection_method="combinations",
        iterate=False,
        light_time=True,
        propagator=ASSISTPropagator,
    )

    # Assertions
    assert len(fitted_orbits) > 0, "No orbits were fitted"
    assert len(fitted_orbit_members) > 0, "No orbit members were fitted"
    assert fitted_orbits.orbit_id[0].as_py() is not None, "Orbit ID is None"
    assert fitted_orbits.coordinates is not None, "Coordinates are None"
