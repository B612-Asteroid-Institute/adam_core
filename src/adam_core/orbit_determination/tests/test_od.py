import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pyarrow.compute as pc
import pytest

from ...coordinates import CoordinateCovariances, SphericalCoordinates
from ...coordinates.origin import Origin
from ...observers import Observers
from ...utils.helpers.observations import make_observations
from ...utils.helpers.orbits import make_real_orbits
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..od import od

# Specify the path to `adam_assist` in `site-packages`
# site_packages_path = next(p for p in sys.path if "__pypackages__/3.11/lib" in p)
site_packages_path = "/Users/natetellis/code/adam_core/__pypackages__/3.11/lib"
assist_path = os.path.join(
    site_packages_path, "adam_core", "propagator", "adam_assist.py"
)

# Import `adam_assist` from `site-packages`
spec = importlib.util.spec_from_file_location(
    "adam_core.propagator.adam_assist", assist_path
)
adam_assist = importlib.util.module_from_spec(spec)
sys.modules["adam_core.propagator.adam_assist"] = adam_assist
spec.loader.exec_module(adam_assist)
from adam_core.propagator.adam_assist import ASSISTPropagator


@pytest.fixture
def real_data():
    # Generate real observations and orbits
    exposures, detections, associations = make_observations()
    orbits = make_real_orbits(num_orbits=18)

    # Select Ivezic for testing
    object_id = orbits.object_id[17].as_py()
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
        time=exposures_i.midpoint(),
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

    # Use the first orbit as the starting orbit
    starting_orbit = orbit

    return starting_orbit, observations


@pytest.mark.skipif(
    os.environ.get("ASSIST_DATA_DIR") is None,
    reason="ASSIST_DATA_DIR environment variable not set",
)
@pytest.mark.skipif(ASSISTPropagator is None, reason="ASSISTPropagator not available")
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
