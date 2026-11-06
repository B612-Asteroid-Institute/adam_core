from importlib.resources import files
from typing import TypeAlias

import pytest

from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits

PureIodOrbit: TypeAlias = tuple[
    FittedOrbits,
    FittedOrbitMembers,
    OrbitDeterminationObservations,
]


@pytest.fixture
def pure_iod_orbit() -> PureIodOrbit:
    data = files("adam_core.orbit_determination.tests.data")
    orbit = FittedOrbits.from_parquet(str(data.joinpath("pure_iod_orbit.parquet")))
    orbit_members = FittedOrbitMembers.from_parquet(
        str(data.joinpath("pure_iod_orbit_members.parquet"))
    )
    observations = OrbitDeterminationObservations.from_parquet(
        str(data.joinpath("pure_iod_orbit_observations.parquet"))
    )
    return orbit, orbit_members, observations
