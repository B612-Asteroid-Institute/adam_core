from ...time import Timestamp
from ..cartesian import CartesianCoordinates
from ..origin import Origin
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
