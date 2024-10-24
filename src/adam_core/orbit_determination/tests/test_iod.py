import numpy as np
import pytest

try:
    from adam_core.propagator.adam_assist import ASSISTPropagator
except ImportError:
    ASSISTPropagator = None

import os

from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...observers.observers import Observers
from ...time.time import Timestamp
from .. import FittedOrbitMembers, OrbitDeterminationObservations
from ..iod import iod


@pytest.mark.skipif(
    os.environ.get("ASSIST_DATA_DIR") is None,
    reason="ASSIST_DATA_DIR environment variable not set",
)
@pytest.mark.skipif(ASSISTPropagator is None, reason="ASSISTPropagator not available")
def test_iod_function():
    # Create mock observations
    time = Timestamp.from_mjd(np.arange(59000, 59006), scale="utc")
    observations = OrbitDeterminationObservations.from_kwargs(
        id=["obs01", "obs02", "obs03", "obs04", "obs05", "obs06"],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=np.random.rand(6),
            lat=np.random.rand(6),
            origin=Origin.from_kwargs(code=np.full(6, "500", dtype="object")),
            time=time,
        ),
        observers=Observers.from_code("500", time),
    )

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
