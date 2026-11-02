from importlib.resources import files

import pytest

from ..algorithms.fitting import OrbitDeterminationObservations
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


@pytest.fixture
def allow_synthetic_model_distribution(monkeypatch):
    """Bypass installed-distribution checks for narrow synthetic model tests.

    Production-route attestation tests restore and exercise the returned real
    validator explicitly. Other tests use local classes only to isolate
    canonical configuration or routing behavior.
    """
    from .. import provenance

    validator = provenance._custom_contract
    monkeypatch.setattr(
        provenance, "_custom_contract", provenance._declared_custom_contract
    )
    return validator
