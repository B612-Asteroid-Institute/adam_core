import numpy as np

from ...coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    KeplerianCoordinates,
)
from ...coordinates.origin import Origin
from ...orbits.non_gravitational_parameters import NonGravitationalParameters
from ...time import Timestamp
from ...utils.helpers import orbits as orbits_helpers
from ..orbits import Orbits


def test_orbits__init__():
    coordinates = CartesianCoordinates.from_kwargs(
        x=[0.5],
        y=[0.5],
        z=[0.004],
        vx=[0.005],
        vy=[-0.005],
        vz=[0.0002],
        time=Timestamp.from_mjd([59000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )

    orbits = Orbits.from_kwargs(
        coordinates=coordinates, orbit_id=["1"], object_id=["Test Orbit"]
    )

    assert orbits.orbit_id[0].as_py() == "1"
    assert orbits.object_id[0].as_py() == "Test Orbit"
    assert orbits.coordinates.time.days.to_numpy()[0] == 59000
    assert orbits.coordinates.time.nanos.to_numpy()[0] == 0
    assert orbits.coordinates.time.scale == "tdb"
    assert orbits.coordinates.x.to_numpy()[0] == 0.5
    assert orbits.coordinates.y.to_numpy()[0] == 0.5
    assert orbits.coordinates.z.to_numpy()[0] == 0.004
    assert orbits.coordinates.vx.to_numpy()[0] == 0.005
    assert orbits.coordinates.vy.to_numpy()[0] == -0.005
    assert orbits.coordinates.vz.to_numpy()[0] == 0.0002


def test_orbit_iteration():
    orbits = orbits_helpers.make_simple_orbits(num_orbits=10)
    for o in orbits:
        assert len(o) == 1


def test_orbits_without_non_gravitational_parameters():
    covariance = np.full((1, 9, 9), np.nan)
    covariance[0] = np.zeros((9, 9))
    covariance[0, :7, :7] = np.eye(7)
    orbits = Orbits.from_kwargs(
        orbit_id=["1"],
        object_id=["Test Orbit"],
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            source=["SBDB"],
            A1=[None],
            A2=[-2.9e-14],
            A3=[None],
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[0.5],
            y=[0.5],
            z=[0.004],
            vx=[0.005],
            vy=[-0.005],
            vz=[0.0002],
            time=Timestamp.from_mjd([59000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
            covariance=CoordinateCovariances.from_matrix(covariance),
        ),
    )

    stripped = orbits.without_non_gravitational_parameters()

    assert orbits.has_non_gravitational_parameters()
    assert orbits.has_non_gravitational_solution()
    assert orbits.coordinates.covariance.has_nongrav_block()
    assert not stripped.has_non_gravitational_parameters()
    assert not stripped.has_non_gravitational_solution()
    assert stripped.non_gravitational_parameters.A2[0].as_py() is None
    assert not stripped.coordinates.covariance.has_nongrav_block()
    np.testing.assert_allclose(
        stripped.coordinates.covariance.to_matrix()[0], np.eye(6)
    )


def test_orbits_extended_covariance_to_keplerian():
    covariance = np.zeros((1, 9, 9))
    covariance[0, :7, :7] = np.eye(7)
    orbits = Orbits.from_kwargs(
        orbit_id=["1"],
        object_id=["Test Orbit"],
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            source=["SBDB"],
            A1=[-2.9e-14],
            A2=[None],
            A3=[None],
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[0.5],
            y=[0.5],
            z=[0.004],
            vx=[0.005],
            vy=[-0.005],
            vz=[0.0002],
            time=Timestamp.from_mjd([59000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
            covariance=CoordinateCovariances.from_matrix(covariance),
        ),
    )

    coords = orbits.to_keplerian()

    assert isinstance(coords, KeplerianCoordinates)
    assert coords.frame == "ecliptic"
    assert coords.covariance.nongrav_block_mask().tolist() == [True]
    full = coords.covariance.to_full_matrix()[0]
    # The A1 dimension carries through the transform unchanged (identity
    # block), and the orbital block matches the transformed 6x6 covariance
    # since the input orbital block is the identity.
    np.testing.assert_allclose(full[6, 6], 1.0, rtol=1e-12)
    np.testing.assert_allclose(full[7:, 7:], np.zeros((2, 2)), atol=0)
    np.testing.assert_allclose(
        full[:6, :6],
        coords.covariance.to_matrix()[0],
        rtol=0,
        atol=0,
    )
