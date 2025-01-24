import quivr as qv
import spiceypy as sp

from ...constants import KM_P_AU, S_P_DAY
from ...time import Timestamp
from ...utils.spice import get_perturber_state, setup_SPICE
from ..cartesian import CartesianCoordinates
from ..origin import Origin, OriginCodes
from ..transform import cartesian_to_frame, transform_coordinates
from .test_transforms_translation import assert_coords_equal


def test_cartesian_to_frame(orbital_elements, orbital_elements_equatorial):
    # Test cartesian_to_frame correctly converts between ecliptic and
    # equatorial cartesian coordinates

    cartesian_coordinates_ecliptic = CartesianCoordinates.from_kwargs(
        x=orbital_elements["x"].values,
        y=orbital_elements["y"].values,
        z=orbital_elements["z"].values,
        vx=orbital_elements["vx"].values,
        vy=orbital_elements["vy"].values,
        vz=orbital_elements["vz"].values,
        time=Timestamp.from_mjd(orbital_elements["mjd_tdb"].values, scale="tdb"),
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN" for i in range(len(orbital_elements))]),
    )

    cartesian_coordinates_equatorial = CartesianCoordinates.from_kwargs(
        x=orbital_elements_equatorial["x"].values,
        y=orbital_elements_equatorial["y"].values,
        z=orbital_elements_equatorial["z"].values,
        vx=orbital_elements_equatorial["vx"].values,
        vy=orbital_elements_equatorial["vy"].values,
        vz=orbital_elements_equatorial["vz"].values,
        time=Timestamp.from_mjd(
            orbital_elements_equatorial["mjd_tdb"].values, scale="tdb"
        ),
        frame="equatorial",
        origin=Origin.from_kwargs(
            code=["SUN" for i in range(len(orbital_elements_equatorial))]
        ),
    )

    # Convert equatorial coordinates to ecliptic
    cartesian_coordinates_ecliptic_actual = cartesian_to_frame(
        cartesian_coordinates_equatorial,
        "ecliptic",
    )
    assert cartesian_coordinates_ecliptic_actual.frame == "ecliptic"

    # Test that the two coordinates are equal to 10 mm in position and 10 nm/s in velocity
    assert_coords_equal(
        cartesian_coordinates_ecliptic_actual, cartesian_coordinates_ecliptic
    )

    # Convert ecliptic coordinates to equatorial
    cartesian_coordinates_equatorial_actual = cartesian_to_frame(
        cartesian_coordinates_ecliptic,
        "equatorial",
    )
    assert cartesian_coordinates_equatorial_actual.frame == "equatorial"

    # Test that the two coordinates are equal to 10 mm in position and 10 nm/s in velocity
    assert_coords_equal(
        cartesian_coordinates_equatorial_actual, cartesian_coordinates_equatorial
    )


def test_transform_coordinates_frame(orbital_elements, orbital_elements_equatorial):
    # Test transform_coordinates correctly converts between ecliptic and
    # equatorial cartesian coordinates

    cartesian_coordinates_ecliptic = CartesianCoordinates.from_kwargs(
        x=orbital_elements["x"].values,
        y=orbital_elements["y"].values,
        z=orbital_elements["z"].values,
        vx=orbital_elements["vx"].values,
        vy=orbital_elements["vy"].values,
        vz=orbital_elements["vz"].values,
        time=Timestamp.from_mjd(
            orbital_elements["mjd_tdb"].values,
            scale="tdb",
        ),
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN" for i in range(len(orbital_elements))]),
    )

    cartesian_coordinates_equatorial = CartesianCoordinates.from_kwargs(
        x=orbital_elements_equatorial["x"].values,
        y=orbital_elements_equatorial["y"].values,
        z=orbital_elements_equatorial["z"].values,
        vx=orbital_elements_equatorial["vx"].values,
        vy=orbital_elements_equatorial["vy"].values,
        vz=orbital_elements_equatorial["vz"].values,
        time=Timestamp.from_mjd(
            orbital_elements_equatorial["mjd_tdb"].values,
            scale="tdb",
        ),
        frame="equatorial",
        origin=Origin.from_kwargs(
            code=["SUN" for i in range(len(orbital_elements_equatorial))]
        ),
    )

    # Convert equatorial coordinates to ecliptic
    cartesian_coordinates_ecliptic_actual = transform_coordinates(
        cartesian_coordinates_equatorial,
        frame_out="ecliptic",
    )
    assert cartesian_coordinates_ecliptic_actual.frame == "ecliptic"

    # Test that the two coordinates are equal to 10 mm in position and 10 nm/s in velocity
    assert_coords_equal(
        cartesian_coordinates_ecliptic_actual, cartesian_coordinates_ecliptic
    )

    # Convert ecliptic coordinates to equatorial
    cartesian_coordinates_equatorial_actual = transform_coordinates(
        cartesian_coordinates_ecliptic,
        frame_out="equatorial",
    )
    assert cartesian_coordinates_equatorial_actual.frame == "equatorial"

    # Test that the two coordinates are equal to 10 mm in position and 10 nm/s in velocity
    assert_coords_equal(
        cartesian_coordinates_equatorial_actual, cartesian_coordinates_equatorial
    )


def test_transform_coordinates_to_itrf93():
    """
    Test that transform_coordinates correctly converts between ecliptic and
    ITRF93 cartesian coordinates
    """
    setup_SPICE()

    # Get perturber states for a few of the planets
    times = Timestamp.from_mjd([59000, 59500, 60000], scale="tdb")
    states = CartesianCoordinates.empty()
    target_ids = []
    for perturber in [
        OriginCodes.MOON,
        OriginCodes.VENUS,
        OriginCodes.MARS_BARYCENTER,
        OriginCodes.JUPITER_BARYCENTER,
        OriginCodes.SATURN_BARYCENTER,
        OriginCodes.URANUS_BARYCENTER,
        OriginCodes.NEPTUNE_BARYCENTER,
    ]:
        states = qv.concatenate(
            [
                states,
                get_perturber_state(
                    perturber, times, origin=OriginCodes.SUN, frame="ecliptic"
                ),
            ]
        )
        target_ids.extend(perturber.name for _ in range(len(times)))

    states_itrf93 = transform_coordinates(states, frame_out="itrf93")

    # Repeat with SPICE
    states_spice_itrf93 = CartesianCoordinates.empty()
    for coord, target_id in zip(states_itrf93, target_ids):
        # Rotate the coordinates to the ITRF93 frame using SPICE
        state, lt = sp.spkezr(
            target_id, coord.time.et()[0].as_py(), "ITRF93", "NONE", "SUN"
        )
        states_spice_itrf93 = qv.concatenate(
            [
                states_spice_itrf93,
                CartesianCoordinates.from_kwargs(
                    x=[state[0] / KM_P_AU],
                    y=[state[1] / KM_P_AU],
                    z=[state[2] / KM_P_AU],
                    vx=[state[3] / KM_P_AU * S_P_DAY],
                    vy=[state[4] / KM_P_AU * S_P_DAY],
                    vz=[state[5] / KM_P_AU * S_P_DAY],
                    time=coord.time,  # Add time from original coordinate
                    frame="itrf93",  # Specify the frame
                    origin=Origin.from_kwargs(code=["SUN"]),  # Specify the origin
                ),
            ]
        )

    # Test that the two coordinate sets are equal within tolerance
    assert_coords_equal(
        states_itrf93, states_spice_itrf93, position_tol_mm=10, velocity_tol_nm_s=500
    )


def test_transform_coordinates_from_itrf93():
    """
    Test that transform_coordinates correctly converts between ITRF93 and
    ecliptic cartesian coordinates
    """
    setup_SPICE()

    # Get perturber states for a few of the planets
    times = Timestamp.from_mjd([59000, 59500, 60000], scale="tdb")
    states = CartesianCoordinates.empty()
    target_ids = []
    for perturber in [
        OriginCodes.MOON,
        OriginCodes.VENUS,
        OriginCodes.MARS_BARYCENTER,
        OriginCodes.JUPITER_BARYCENTER,
        OriginCodes.SATURN_BARYCENTER,
        OriginCodes.URANUS_BARYCENTER,
        OriginCodes.NEPTUNE_BARYCENTER,
    ]:
        states = qv.concatenate(
            [
                states,
                get_perturber_state(
                    perturber, times, origin=OriginCodes.SUN, frame="itrf93"
                ),
            ]
        )
        target_ids.extend(perturber.name for _ in range(len(times)))

    states_ecliptic = transform_coordinates(states, frame_out="ecliptic")

    # Repeat with SPICE
    states_spice_ecliptic = CartesianCoordinates.empty()
    for coord, target_id in zip(states_ecliptic, target_ids):
        # Rotate the coordinates to the ecliptic frame using SPICE
        state, lt = sp.spkezr(
            target_id, coord.time.et()[0].as_py(), "ECLIPJ2000", "NONE", "SUN"
        )
        states_spice_ecliptic = qv.concatenate(
            [
                states_spice_ecliptic,
                CartesianCoordinates.from_kwargs(
                    x=[state[0] / KM_P_AU],
                    y=[state[1] / KM_P_AU],
                    z=[state[2] / KM_P_AU],
                    vx=[state[3] / KM_P_AU * S_P_DAY],
                    vy=[state[4] / KM_P_AU * S_P_DAY],
                    vz=[state[5] / KM_P_AU * S_P_DAY],
                    time=coord.time,  # Add time from original coordinate
                    frame="ecliptic",  # Specify the frame
                    origin=Origin.from_kwargs(code=["SUN"]),  # Specify the origin
                ),
            ]
        )

    # Test that the two coordinate sets are equal within tolerance
    assert_coords_equal(
        states_ecliptic,
        states_spice_ecliptic,
        position_tol_mm=10,
        velocity_tol_nm_s=500,
    )
