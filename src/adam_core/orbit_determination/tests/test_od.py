import numpy as np
import pyarrow.compute as pc
import pytest
from adam_assist import ASSISTPropagator

from ...coordinates import CoordinateCovariances, SphericalCoordinates
from ...coordinates.origin import Origin
from ...observers import Observers
from ...utils.helpers.observations import make_observations
from ...utils.helpers.orbits import make_real_orbits
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..od import od


@pytest.fixture
def real_data():
    # Generate real observations and orbits
    exposures, detections, associations = make_observations()
    orbits = make_real_orbits(num_orbits=18)

    # Select Ivezic for testing
    object_id = "202930 Ivezic (1998 SG172)"
    orbit = orbits.select("object_id", object_id)

    # Filter observations for the selected object ID
    associations_i = associations.select("object_id", object_id)
    detections_i = detections.apply_mask(
        pc.is_in(detections.id, associations_i.detection_id)
    )

    exposures_i = exposures.apply_mask(pc.is_in(exposures.id, detections_i.exposure_id))

    sigmas = np.full((len(detections_i.ra_sigma), 6), np.nan)
    sigmas[:, 1] = detections_i.ra_sigma.to_numpy(zero_copy_only=False)
    sigmas[:, 2] = detections_i.dec_sigma.to_numpy(zero_copy_only=False)
    coordinates = SphericalCoordinates.from_kwargs(
        lon=detections_i.ra.to_numpy(),
        lat=detections_i.dec.to_numpy(),
        covariance=CoordinateCovariances.from_sigmas(sigmas),
        time=detections_i.time,
        origin=Origin.from_kwargs(code=exposures_i.observatory_code),
        frame="equatorial",  # Assuming the frame is equatorial
    )

    # Generate Observers from exposures start_time and observatory codes
    observers = Observers.from_codes(
        times=detections_i.time, codes=exposures_i.observatory_code
    )

    observations = OrbitDeterminationObservations.from_kwargs(
        id=detections_i.id.to_numpy(zero_copy_only=False),
        coordinates=coordinates,
        observers=observers,
    )

    # Use the first orbit as the starting orbit
    starting_orbit = orbit

    return starting_orbit, observations


def test_od(real_data):
    starting_orbit, observations = real_data

    # Run the orbit determination
    od_orbit, od_orbit_members = od(
        orbit=starting_orbit,
        observations=observations[:10],
        propagator=ASSISTPropagator,
        rchi2_threshold=10,
        min_obs=3,
        min_arc_length=1.0,
        contamination_percentage=0.0,
        delta=1e-6,
        max_iter=20,
        method="central",
        propagator_kwargs={},
    )
    # Assertions

    assert isinstance(od_orbit, FittedOrbits)
    assert isinstance(od_orbit_members, FittedOrbitMembers)
    assert len(od_orbit) == 1
    assert len(od_orbit_members) == 10
    assert od_orbit.success[0]
