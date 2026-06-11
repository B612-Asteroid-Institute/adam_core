import numpy as np

from ...coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    KeplerianCoordinates,
)
from ...coordinates.origin import Origin
from ...orbits.non_gravitational_parameters import NonGravitationalParameters
from ...orbits.solved_state_covariances import SolvedStateCovariances
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
    orbits = Orbits.from_kwargs(
        orbit_id=["1"],
        object_id=["Test Orbit"],
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            source=["SBDB"],
            model=["nongrav"],
            solution_dimension=[7],
            parameter_count=[1],
            estimated_parameter_names=["A2"],
            A1=[None],
            A1_sigma=[None],
            A2=[-2.9e-14],
            A2_sigma=[1e-15],
            A3=[None],
            A3_sigma=[None],
            DT=[None],
            DT_sigma=[None],
            R0=[None],
            R0_sigma=[None],
            ALN=[None],
            ALN_sigma=[None],
            NK=[None],
            NK_sigma=[None],
            NM=[None],
            NM_sigma=[None],
            NN=[None],
            NN_sigma=[None],
            AMRAT=[None],
            AMRAT_sigma=[None],
            RHO=[None],
            RHO_sigma=[None],
        ),
        solved_state_covariance=SolvedStateCovariances.from_matrix(
            [np.eye(7)], [["x", "y", "z", "vx", "vy", "vz", "A2"]]
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
        ),
    )

    stripped = orbits.without_non_gravitational_parameters()

    assert orbits.has_non_gravitational_parameters()
    assert not stripped.has_non_gravitational_parameters()
    assert stripped.non_gravitational_parameters.A2[0].as_py() is None
    assert stripped.solved_state_covariance.dimension[0].as_py() is None


def test_orbits_solved_state_covariance_to_keplerian():
    orbits = Orbits.from_kwargs(
        orbit_id=["1"],
        object_id=["Test Orbit"],
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            source=["SBDB"],
            model=["nongrav"],
            solution_dimension=[7],
            parameter_count=[1],
            estimated_parameter_names=["A2"],
            A1=[None],
            A1_sigma=[None],
            A2=[-2.9e-14],
            A2_sigma=[1e-15],
            A3=[None],
            A3_sigma=[None],
            DT=[None],
            DT_sigma=[None],
            R0=[None],
            R0_sigma=[None],
            ALN=[None],
            ALN_sigma=[None],
            NK=[None],
            NK_sigma=[None],
            NM=[None],
            NM_sigma=[None],
            NN=[None],
            NN_sigma=[None],
            AMRAT=[None],
            AMRAT_sigma=[None],
            RHO=[None],
            RHO_sigma=[None],
        ),
        solved_state_covariance=SolvedStateCovariances.from_matrix(
            [np.eye(7)], [["x", "y", "z", "vx", "vy", "vz", "A2"]]
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
            covariance=CoordinateCovariances.from_matrix(
                np.eye(6).reshape(1, 6, 6)
            ),
        ),
    )

    coords = orbits.to_keplerian()
    solved = orbits.solved_state_covariance_to(KeplerianCoordinates)

    assert isinstance(coords, KeplerianCoordinates)
    assert coords.frame == "ecliptic"
    assert solved.dimension[0].as_py() == 7
    assert solved.parameter_names[0].as_py() == "a,e,i,raan,ap,M,A2"
    # The solved-state covariance fixture's orbital block is the identity, the
    # same as the coordinate covariance, so the transformed orbital block must
    # match the independently transformed 6x6 coordinate covariance.
    np.testing.assert_allclose(
        solved.to_orbital_covariances().to_matrix(),
        coords.covariance.to_matrix(),
        rtol=1e-12,
        atol=0,
    )
