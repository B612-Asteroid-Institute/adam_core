from importlib.resources import files

import pytest

from .._native_dispatch import _PYTHON_PROPAGATORS, _allow_python_propagators
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


@pytest.fixture(autouse=True, scope="session")
def python_test_propagators():
    """The suite exercises the Rust OD loops with Python test dynamics
    (linear, counting, failing propagators) through the private callback
    adapter; production OD refuses them (``native_only`` tests check that)."""
    with _allow_python_propagators():
        yield


@pytest.fixture(autouse=True)
def _native_only(request):
    """Tests marked ``native_only`` run without the Python-propagator
    adapter, as production OD does."""
    if not request.node.get_closest_marker("native_only"):
        yield
        return
    token = _PYTHON_PROPAGATORS.set(False)
    try:
        yield
    finally:
        _PYTHON_PROPAGATORS.reset(token)
