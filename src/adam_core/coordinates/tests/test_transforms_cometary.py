import numpy as np
from astropy import units as u

from ..origin import Origin
from ..transform import cartesian_to_cometary, cometary_to_cartesian


def test_cometary_to_cartesian_elliptical(orbital_elements):
    # Test cometary_to_cartesian (vmapped) against the expected cartesian elements
    # of a series of sample orbital elements.

    # Limit to elliptical orbits
    orbital_elements = orbital_elements[orbital_elements["e"] < 1]

    cometary_elements = orbital_elements[
        ["q", "e", "incl", "Omega", "w", "tp_mjd"]
    ].values
    cartesian_elements_expected = orbital_elements[
        ["x", "y", "z", "vx", "vy", "vz"]
    ].values
    t0_mjd = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    cartesian_elements_actual = cometary_to_cartesian(
        cometary_elements, t0_mjd, origin.mu()
    )
    diff = cartesian_elements_actual - cartesian_elements_expected

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 20 m
    np.testing.assert_array_less(
        r_diff, 10 * 1000 * 2
    )  # TODO: why does this need to be so much larger than the keplerian_to_cartesian test?
    # Assert velocities are to within 10 mm/s
    np.testing.assert_array_less(v_diff, 10)


def test_cometary_to_cartesian_hyperbolic(orbital_elements):
    # Test cometary_to_cartesian (vmapped) against the expected cartesian elements
    # of a series of sample orbital elements.

    # Limit to hyperbolic orbits
    orbital_elements = orbital_elements[orbital_elements["e"] > 1]

    cometary_elements = orbital_elements[
        ["q", "e", "incl", "Omega", "w", "tp_mjd"]
    ].values
    cartesian_elements_expected = orbital_elements[
        ["x", "y", "z", "vx", "vy", "vz"]
    ].values
    t0_mjd = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    cartesian_elements_actual = cometary_to_cartesian(
        cometary_elements, t0_mjd, origin.mu()
    )
    diff = cartesian_elements_actual - cartesian_elements_expected

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 20 m
    np.testing.assert_array_less(
        r_diff, 10 * 1000 * 2
    )  # TODO: why does this need to be so much larger than the keplerian_to_cartesian test?
    # Assert velocities are to within 10 mm/s
    np.testing.assert_array_less(v_diff, 10)


def test_cartesian_to_cometary_elliptical(orbital_elements):
    # Test cartesian_to_cometary (vmapped) against the expected cometary elements
    # of a series of sample orbital elements.

    # Limit to elliptical orbits
    orbital_elements = orbital_elements[orbital_elements["e"] < 1]

    cometary_elements_expected = orbital_elements[
        ["q", "e", "incl", "Omega", "w", "tp_mjd"]
    ].values
    epochs = orbital_elements["mjd_tdb"].values
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    # Cartesian to cometary returns an N, 13 array containing the cometary elements and cometary
    # elements, with some additional columns.
    cometary_elements_actual = cartesian_to_cometary(
        cartesian_elements, epochs, origin.mu()
    )

    # Calculate the difference
    diff = cometary_elements_actual - cometary_elements_expected

    # Calculate offset in periapsis distance in m
    q_diff = np.abs(diff[:, 0]) * u.au.to(u.m)
    # Assert periapsis distance is to within 100 m
    np.testing.assert_array_less(q_diff, 200)  # TODO

    # Calculate offset in eccentricity
    e_diff = np.abs(diff[:, 1])
    # Assert eccentricity is to within 1e-10
    np.testing.assert_array_less(e_diff, 1e-10)

    # Calculate offset in inclination in microarcseconds
    i_diff = np.abs(diff[:, 2]) * u.degree.to(u.microarcsecond)
    # Assert inclination is to within 10 microarcseconds
    np.testing.assert_array_less(i_diff, 10)

    # Calculate offset in longitude of ascending node in microarcseconds
    raan_diff = np.abs(diff[:, 3]) * u.degree.to(u.microarcsecond)
    # Assert longitude of ascending node is to within 10 microarcseconds
    np.testing.assert_array_less(raan_diff, 10)

    # Calculate offset in argument of periapsis in microarcseconds
    ap_diff = np.abs(diff[:, 4]) * u.degree.to(u.microarcsecond)
    # Assert argument of periapsis is to within 100 microarcseconds
    np.testing.assert_array_less(ap_diff, 100)  # TODO

    # Calculate offset in time of perihelion passage in seconds
    tp_diff = np.abs(diff[:, 5]) * u.d.to(u.s)
    # Assert time of perihelion passage is to within 1 second
    np.testing.assert_array_less(tp_diff, 1)

    # TODO: All TODOs should be checked to see if they are reasonable tolerances.
    #  I'd expect the semi-major axis to be much better.


def test_cartesian_to_cometary_hyperbolic(orbital_elements):
    # Test cartesian_to_cometary (vmapped) against the expected cometary elements
    # of a series of sample orbital elements.

    # Limit to hyperbolic orbits
    orbital_elements = orbital_elements[orbital_elements["e"] > 1]

    cometary_elements_expected = orbital_elements[
        ["q", "e", "incl", "Omega", "w", "tp_mjd"]
    ].values
    epochs = orbital_elements["mjd_tdb"].values
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    # Cartesian to cometary returns an N, 13 array containing the cometary elements and cometary
    # elements, with some additional columns.
    cometary_elements_actual = cartesian_to_cometary(
        cartesian_elements, epochs, origin.mu()
    )

    # Calculate the difference
    diff = cometary_elements_actual - cometary_elements_expected

    # Calculate offset in periapsis distance in m
    q_diff = np.abs(diff[:, 0]) * u.au.to(u.m)
    # Assert periapsis distance is to within 100 m
    np.testing.assert_array_less(q_diff, 200)  # TODO

    # Calculate offset in eccentricity
    e_diff = np.abs(diff[:, 1])
    # Assert eccentricity is to within 1e-10
    np.testing.assert_array_less(e_diff, 1e-10)

    # Calculate offset in inclination in microarcseconds
    i_diff = np.abs(diff[:, 2]) * u.degree.to(u.microarcsecond)
    # Assert inclination is to within 10 microarcseconds
    np.testing.assert_array_less(i_diff, 10)

    # Calculate offset in longitude of ascending node in microarcseconds
    raan_diff = np.abs(diff[:, 3]) * u.degree.to(u.microarcsecond)
    # Assert longitude of ascending node is to within 10 microarcseconds
    np.testing.assert_array_less(raan_diff, 10)

    # Calculate offset in argument of periapsis in microarcseconds
    ap_diff = np.abs(diff[:, 4]) * u.degree.to(u.microarcsecond)
    # Assert argument of periapsis is to within 100 microarcseconds
    np.testing.assert_array_less(ap_diff, 100)  # TODO

    # Calculate offset in time of perihelion passage in seconds
    tp_diff = np.abs(diff[:, 5]) * u.d.to(u.s)
    # Assert time of perihelion passage is to within 1 second
    np.testing.assert_array_less(tp_diff, 1)

    # TODO: All TODOs should be checked to see if they are reasonable tolerances.
    #  I'd expect the semi-major axis to be much better.
