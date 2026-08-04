import numpy as np
import pytest

from ...coordinates.origin import Origin
from ...coordinates.residuals import Residuals
from ...coordinates.spherical import SphericalCoordinates
from ...observers.observers import Observers
from ...time.time import Timestamp
from ..differential_correction import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers
from ..outliers import remove_lowest_probability_observation


def test_remove_lowest_probability_observation():
    # Test that remove lowest probability observation correctly identifies and
    # removes the worst outlier

    # Create orbit members with 4 observations, one of which is a bad outlier
    orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=["orbit01", "orbit01", "orbit01", "orbit01"],
        obs_id=["obs01", "obs02", "obs03", "obs04"],
        residuals=Residuals.from_kwargs(
            values=[
                [np.nan, 0.1, 0.1, np.nan, np.nan, np.nan],
                [np.nan, 0.1, 0.1, np.nan, np.nan, np.nan],
                [np.nan, 0.1, 0.1, np.nan, np.nan, np.nan],
                [np.nan, 0.1, 0.1, np.nan, np.nan, np.nan],
            ],
            dof=np.full(4, 2),
            probability=[0.2, 0.1, 0.0001, 0.3],
        ),
    )

    # Create observations with 4 observations
    time = Timestamp.from_mjd(np.arange(59000, 59004), scale="utc")
    observations = OrbitDeterminationObservations.from_kwargs(
        id=["obs01", "obs02", "obs03", "obs04"],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=np.random.rand(4),
            lat=np.random.rand(4),
            origin=Origin.from_kwargs(code=np.full(4, "500", dtype="object")),
            time=time,
        ),
        observers=Observers.from_code("500", time),
    )

    obs_id, observations = remove_lowest_probability_observation(
        orbit_members, observations
    )
    assert len(observations) == 3
    assert observations.id.tolist() == ["obs01", "obs02", "obs04"]
    assert obs_id == "obs03"


def test_remove_lowest_probability_observation_multiple():
    # Test that remove lowest probability observation correctly identifies and
    # removes the worst outlier when there are multiple observations with the same
    # probability

    # Create orbit members with 4 observations, one of which is a bad outlier
    orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=["orbit01", "orbit01", "orbit01", "orbit01"],
        obs_id=["obs01", "obs02", "obs03", "obs04"],
        residuals=Residuals.from_kwargs(
            values=[
                [np.nan, 0.1, -0.1, np.nan, np.nan, np.nan],
                [np.nan, 0.6, -0.6, np.nan, np.nan, np.nan],
                [np.nan, 0.3, -0.3, np.nan, np.nan, np.nan],
                [np.nan, 0.5, -0.5, np.nan, np.nan, np.nan],
            ],
            dof=np.full(4, 2),
            probability=[0.2, 0.2, 0.2, 0.2],
        ),
    )

    # Create observations with 4 observations
    time = Timestamp.from_mjd(np.arange(59000, 59004), scale="utc")
    observations = OrbitDeterminationObservations.from_kwargs(
        id=["obs01", "obs02", "obs03", "obs04"],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=np.random.rand(4),
            lat=np.random.rand(4),
            origin=Origin.from_kwargs(code=np.full(4, "500", dtype="object")),
            time=time,
        ),
        observers=Observers.from_code("500", time),
    )

    obs_id, observations = remove_lowest_probability_observation(
        orbit_members, observations
    )
    assert len(observations) == 3
    assert observations.id.tolist() == ["obs01", "obs03", "obs04"]
    assert obs_id == "obs02"


def test_remove_lowest_probability_observation_assertions():

    # Create observations with 4 observations
    time = Timestamp.from_mjd(np.arange(59000, 59004), scale="utc")
    observations = OrbitDeterminationObservations.from_kwargs(
        id=["obs01", "obs02", "obs03", "obs04"],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=np.random.rand(4),
            lat=np.random.rand(4),
            origin=Origin.from_kwargs(code=np.full(4, "500", dtype="object")),
            time=time,
        ),
        observers=Observers.from_code("500", time),
    )

    # Create orbit members with two orbits
    orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=["orbit01", "orbit01", "orbit01", "orbit02"],
        obs_id=["obs01", "obs02", "obs03", "obs04"],
        residuals=Residuals.from_kwargs(
            values=[
                [np.nan, 0.1, -0.1, np.nan, np.nan, np.nan],
                [np.nan, 0.6, -0.6, np.nan, np.nan, np.nan],
                [np.nan, 0.3, -0.3, np.nan, np.nan, np.nan],
                [np.nan, 0.5, -0.5, np.nan, np.nan, np.nan],
            ],
            dof=np.full(4, 2),
            probability=[0.2, 0.2, 0.2, 0.2],
        ),
    )

    with pytest.raises(
        AssertionError, match=r"Orbit members must only contain one orbit"
    ):
        obs_id, observations = remove_lowest_probability_observation(
            orbit_members, observations
        )

    # Create orbit members with 4 observations, two of which do not
    # exit in the observations
    orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=["orbit01", "orbit01", "orbit01", "orbit01"],
        obs_id=["obs03", "obs04", "obs05", "obs06"],
        residuals=Residuals.from_kwargs(
            values=[
                [np.nan, 0.1, -0.1, np.nan, np.nan, np.nan],
                [np.nan, 0.6, -0.6, np.nan, np.nan, np.nan],
                [np.nan, 0.3, -0.3, np.nan, np.nan, np.nan],
                [np.nan, 0.5, -0.5, np.nan, np.nan, np.nan],
            ],
            dof=np.full(4, 2),
            probability=[0.2, 0.2, 0.2, 0.2],
        ),
    )

    with pytest.raises(
        AssertionError, match=r"Observations must contain all orbit member observations"
    ):
        obs_id, observations = remove_lowest_probability_observation(
            orbit_members, observations
        )
