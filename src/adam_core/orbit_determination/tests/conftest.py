from importlib.resources import files

import pytest

from ..differential_correction import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits


@pytest.fixture
def pure_iod_orbit():
    orbit = FittedOrbits.from_parquet(
        files("adam_core.orbit_determination.tests.data").joinpath(
            "pure_iod_orbit.parquet"
        )
    )
    orbit_members = FittedOrbitMembers.from_parquet(
        files("adam_core.orbit_determination.tests.data").joinpath(
            "pure_iod_orbit_members.parquet"
        )
    )
    observations = OrbitDeterminationObservations.from_parquet(
        files("adam_core.orbit_determination.tests.data").joinpath(
            "pure_iod_orbit_observations.parquet"
        )
    )
    return orbit, orbit_members, observations
