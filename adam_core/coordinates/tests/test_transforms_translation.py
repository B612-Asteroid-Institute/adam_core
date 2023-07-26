import numpy as np
from astropy import units as u
from astropy.time import Time
from quivr.concat import concatenate

from ..cartesian import CartesianCoordinates
from ..origin import Origin, OriginCodes
from ..times import Times
from ..transform import cartesian_to_origin


def test_cartesian_to_origin(orbital_elements, orbital_elements_barycentric):
    # Test cartesian_to_origin correctly converts between heliocentric and
    # barycentric cartesian coordinates

    cartesian_coordinates_heliocentric = CartesianCoordinates.from_kwargs(
        x=orbital_elements["x"].values,
        y=orbital_elements["y"].values,
        z=orbital_elements["z"].values,
        vx=orbital_elements["vx"].values,
        vy=orbital_elements["vy"].values,
        vz=orbital_elements["vz"].values,
        time=Times.from_astropy(
            Time(orbital_elements["mjd_tdb"].values, format="mjd", scale="tdb")
        ),
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN" for i in range(len(orbital_elements))]),
    )

    cartesian_coordinates_barycentric = CartesianCoordinates.from_kwargs(
        x=orbital_elements_barycentric["x"].values,
        y=orbital_elements_barycentric["y"].values,
        z=orbital_elements_barycentric["z"].values,
        vx=orbital_elements_barycentric["vx"].values,
        vy=orbital_elements_barycentric["vy"].values,
        vz=orbital_elements_barycentric["vz"].values,
        time=Times.from_astropy(
            Time(
                orbital_elements_barycentric["mjd_tdb"].values,
                format="mjd",
                scale="tdb",
            )
        ),
        frame="ecliptic",
        origin=Origin.from_kwargs(
            code=[
                "SOLAR_SYSTEM_BARYCENTER"
                for i in range(len(orbital_elements_barycentric))
            ]
        ),
    )

    # Convert heliocentric cartesian coordinates to barycentric cartesian coordinates
    cartesian_coordinates_barycentric_actual = cartesian_to_origin(
        cartesian_coordinates_heliocentric,
        OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    )

    diff = (
        cartesian_coordinates_barycentric_actual.values
        - cartesian_coordinates_barycentric.values
    )

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in nm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.nm / u.s)

    # Assert positions are to within 10 mm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 10 nm/s
    np.testing.assert_array_less(v_diff, 10)

    # Convert barycentric cartesian coordinates to heliocentric cartesian coordinates
    cartesian_coordinates_heliocentric_actual = cartesian_to_origin(
        cartesian_coordinates_barycentric,
        OriginCodes.SUN,
    )

    diff = (
        cartesian_coordinates_heliocentric_actual.values
        - cartesian_coordinates_heliocentric.values
    )

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in nm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.nm / u.s)

    # Assert positions are to within 10 mm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 10 nm/s
    np.testing.assert_array_less(v_diff, 10)

    # Now lets complicate matters by concatenating heliocentric and barycentric
    # cartesian coordinates and converting them to barycentric cartesian coordinates
    cartesian_coordinates_mixed = concatenate(
        [cartesian_coordinates_heliocentric, cartesian_coordinates_barycentric]
    )

    cartesian_coordinates_barycentric_actual = cartesian_to_origin(
        cartesian_coordinates_mixed,
        OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    )

    diff = cartesian_coordinates_barycentric_actual.values - np.tile(
        cartesian_coordinates_barycentric.values, (2, 1)
    )

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in nm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.nm / u.s)

    # Assert positions are to within 10 mm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 10 nm/s
    np.testing.assert_array_less(v_diff, 10)

    # Now lets complicate matters again by concatenating heliocentric and barycentric
    # cartesian coordinates and converting them to heliocentric cartesian coordinates
    cartesian_coordinates_mixed = concatenate(
        [cartesian_coordinates_heliocentric, cartesian_coordinates_barycentric]
    )

    cartesian_coordinates_heliocentric_actual = cartesian_to_origin(
        cartesian_coordinates_mixed,
        OriginCodes.SUN,
    )

    diff = cartesian_coordinates_heliocentric_actual.values - np.tile(
        cartesian_coordinates_heliocentric.values, (2, 1)
    )

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in nm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.nm / u.s)

    # Assert positions are to within 10 mm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 10 nm/s
    np.testing.assert_array_less(v_diff, 10)
