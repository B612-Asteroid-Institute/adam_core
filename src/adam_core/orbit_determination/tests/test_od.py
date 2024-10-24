from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from adam_core.propagator.adam_assist import ASSISTPropagator
except ImportError:
    ASSISTPropagator = None

import os

from ...coordinates import CartesianCoordinates
from ...propagator import Propagator
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..od import od


@pytest.fixture
def mock_data():
    # Create mock observations
    observations = OrbitDeterminationObservations(
        id=np.array([1, 2, 3]),
        coordinates=CartesianCoordinates(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            z=np.array([1.0, 2.0, 3.0]),
            vx=np.array([0.1, 0.2, 0.3]),
            vy=np.array([0.1, 0.2, 0.3]),
            vz=np.array([0.1, 0.2, 0.3]),
            time=np.array([2451545.0, 2451546.0, 2451547.0]),
            origin="SSB",
            frame="ICRF",
        ),
        observers=np.array(["Earth", "Earth", "Earth"]),
    )

    # Create a mock starting orbit
    starting_orbit = FittedOrbits.from_kwargs(
        orbit_id=np.array(["test_orbit"]),
        object_id=np.array(["test_object"]),
        coordinates=CartesianCoordinates(
            x=np.array([1.0]),
            y=np.array([1.0]),
            z=np.array([1.0]),
            vx=np.array([0.1]),
            vy=np.array([0.1]),
            vz=np.array([0.1]),
            time=np.array([2451545.0]),
            origin="SSB",
            frame="ICRF",
        ),
        arc_length=np.array([1.0]),
        num_obs=np.array([3]),
        chi2=np.array([0.0]),
        reduced_chi2=np.array([0.0]),
        iterations=np.array([0]),
        success=np.array([False]),
        status_code=np.array([0]),
    )

    # Mock the propagator
    propagator = MagicMock(spec=Propagator)
    propagator.generate_ephemeris.return_value = observations

    return starting_orbit, observations, propagator


@pytest.mark.skipif(
    os.environ.get("ASSIST_DATA_DIR") is None,
    reason="ASSIST_DATA_DIR environment variable not set",
)
@pytest.mark.skipif(ASSISTPropagator is None, reason="ASSISTPropagator not available")
def test_od(mock_data):
    starting_orbit, observations, propagator = mock_data

    # Run the orbit determination
    od_orbit, od_orbit_members = od(
        orbit=starting_orbit,
        observations=observations,
        propagator=propagator,
        rchi2_threshold=100,
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
    assert len(od_orbit) >= 1
    assert len(od_orbit_members) >= 1
    assert od_orbit.success[0]
