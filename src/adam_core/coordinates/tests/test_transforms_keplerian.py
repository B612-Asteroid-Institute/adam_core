import numpy as np
from astropy import units as u

from ..origin import Origin
from ..transform import cartesian_to_keplerian, keplerian_to_cartesian


def test_keplerian_to_cartesian_elliptical(orbital_elements):
    # Test keplerian_to_cartesian (vmapped) against the expected cartesian elements
    # of a series of sample orbital elements.

    # Limit to elliptical orbits
    orbital_elements = orbital_elements[orbital_elements["e"] < 1]

    keplerian_elements = orbital_elements[["a", "e", "incl", "Omega", "w", "M"]].values
    cartesian_elements_expected = orbital_elements[
        ["x", "y", "z", "vx", "vy", "vz"]
    ].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    cartesian_elements_actual = keplerian_to_cartesian(keplerian_elements, origin.mu())
    diff = cartesian_elements_actual - cartesian_elements_expected

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 10 mm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 10 mm/s
    np.testing.assert_array_less(v_diff, 10)


def test_keplerian_to_cartesian_hyperbolic(orbital_elements):
    # Test keplerian_to_cartesian (vmapped) against the expected cartesian elements
    # of a series of sample orbital elements.

    # Limit to hyperbolic orbits
    orbital_elements = orbital_elements[orbital_elements["e"] > 1]

    keplerian_elements = orbital_elements[["a", "e", "incl", "Omega", "w", "M"]].values
    cartesian_elements_expected = orbital_elements[
        ["x", "y", "z", "vx", "vy", "vz"]
    ].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    cartesian_elements_actual = keplerian_to_cartesian(keplerian_elements, origin.mu())
    diff = cartesian_elements_actual - cartesian_elements_expected

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 10 mm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 10 mm/s
    np.testing.assert_array_less(v_diff, 10)


def test_cartesian_to_keplerian_elliptical(orbital_elements):
    # Test cartesian_to_keplerian (vmapped) against the expected keplerian elements
    # of a series of sample orbital elements.

    # Limit to elliptical orbits
    orbital_elements = orbital_elements[orbital_elements["e"] < 1]

    keplerian_elements_expected = orbital_elements[
        ["a", "q", "Q", "e", "incl", "Omega", "w", "M", "nu", "n", "P", "tp_mjd"]
    ].values
    epochs = orbital_elements["mjd_tdb"].values
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    # Cartesian to keplerian returns an N, 13 array containing the keplerian elements and cometary
    # elements, with some additional columns.
    keplerian_elements_actual = cartesian_to_keplerian(
        cartesian_elements, epochs, origin.mu()
    )

    # Remove semi-latus rectum
    keplerian_elements_actual = keplerian_elements_actual[
        :, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ]

    # Calculate the difference
    diff = keplerian_elements_actual - keplerian_elements_expected

    # Calculate offset in semi-major axis in m
    a_diff = np.abs(diff[:, 0]) * u.au.to(u.m)
    # Assert semi-major axis is to within 100 m
    np.testing.assert_array_less(a_diff, 200)  # TODO

    # Calculate offset in periapsis distance in m
    q_diff = np.abs(diff[:, 1]) * u.au.to(u.m)
    # Assert periapsis distance is to within 100 m
    np.testing.assert_array_less(q_diff, 200)  # TODO

    # Calculate offset in apoapsis distance in m
    Q_diff = np.abs(diff[:, 2]) * u.au.to(u.m)
    # Assert apoapsis distance is to within 100 m
    np.testing.assert_array_less(Q_diff, 200)  # TODO

    # Calculate offset in eccentricity
    e_diff = np.abs(diff[:, 3])
    # Assert eccentricity is to within 1e-10
    np.testing.assert_array_less(e_diff, 1e-10)

    # Calculate offset in inclination in microarcseconds
    i_diff = np.abs(diff[:, 4]) * u.degree.to(u.microarcsecond)
    # Assert inclination is to within 10 microarcseconds
    np.testing.assert_array_less(i_diff, 10)

    # Calculate offset in longitude of ascending node in microarcseconds
    raan_diff = np.abs(diff[:, 5]) * u.degree.to(u.microarcsecond)
    # Assert longitude of ascending node is to within 10 microarcseconds
    np.testing.assert_array_less(raan_diff, 10)

    # Calculate offset in argument of periapsis in microarcseconds
    ap_diff = np.abs(diff[:, 6]) * u.degree.to(u.microarcsecond)
    # Assert argument of periapsis is to within 100 microarcseconds
    np.testing.assert_array_less(ap_diff, 100)  # TODO

    # Calculate offset in mean anomaly in microarcseconds
    M_diff = np.abs(diff[:, 7]) * u.degree.to(u.microarcsecond)
    # Assert mean anomaly is to within 100 microarcseconds
    np.testing.assert_array_less(M_diff, 100)  # TODO

    # Calculate offset in true anomaly in microarcseconds
    nu_diff = np.abs(diff[:, 8]) * u.degree.to(u.microarcsecond)
    # Assert true anomaly is to within 100 microarcseconds
    np.testing.assert_array_less(nu_diff, 100)  # TODO

    # Calculate offset in mean motion in microarcseconds per day
    n_diff = np.abs(diff[:, 9]) * (u.degree / u.d).to(u.microarcsecond / u.d)
    # Assert mean motion is to within 1 microarcseconds per day
    np.testing.assert_array_less(n_diff, 1)

    # Calculate offset in period in seconds
    P_diff = np.abs(diff[:, 10]) * u.d.to(u.s)
    # Assert period is to within 1 second
    np.testing.assert_array_less(P_diff, 1)

    # Calculate offset in time of perihelion passage in seconds
    tp_diff = np.abs(diff[:, 11]) * u.d.to(u.s)
    # Assert time of perihelion passage is to within 1 second
    np.testing.assert_array_less(tp_diff, 1)

    # TODO: All TODOs should be checked to see if they are reasonable tolerances.
    #  I'd expect the semi-major axis to be much better.


def test_cartesian_to_keplerian_hyperbolic(orbital_elements):
    # Test cartesian_to_keplerian (vmapped) against the expected keplerian elements
    # of a series of sample orbital elements.

    # Limit to hyperbolic orbits
    orbital_elements = orbital_elements[orbital_elements["e"] > 1]

    keplerian_elements_expected = orbital_elements[
        ["a", "q", "Q", "e", "incl", "Omega", "w", "M", "nu", "n", "P", "tp_mjd"]
    ].values
    epochs = orbital_elements["mjd_tdb"].values
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    # Cartesian to keplerian returns an N, 13 array containing the keplerian elements and cometary
    # elements, with some additional columns.
    keplerian_elements_actual = cartesian_to_keplerian(
        cartesian_elements, epochs, origin.mu()
    )

    # Remove semi-latus rectum
    keplerian_elements_actual = keplerian_elements_actual[
        :, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ]

    # Calculate the difference
    diff = keplerian_elements_actual - keplerian_elements_expected

    # Calculate offset in semi-major axis in m
    a_diff = np.abs(diff[:, 0]) * u.au.to(u.m)
    # Assert semi-major axis is to within 100 m
    np.testing.assert_array_less(a_diff, 200)  # TODO

    # Calculate offset in periapsis distance in m
    q_diff = np.abs(diff[:, 1]) * u.au.to(u.m)
    # Assert periapsis distance is to within 100 m
    np.testing.assert_array_less(q_diff, 200)  # TODO

    # Apoapsis distance should be infinite
    assert np.all(np.isinf(keplerian_elements_actual[:, 2]))

    # Calculate offset in eccentricity
    e_diff = np.abs(diff[:, 3])
    # Assert eccentricity is to within 1e-10
    np.testing.assert_array_less(e_diff, 1e-10)

    # Calculate offset in inclination in microarcseconds
    i_diff = np.abs(diff[:, 4]) * u.degree.to(u.microarcsecond)
    # Assert inclination is to within 10 microarcseconds
    np.testing.assert_array_less(i_diff, 10)

    # Calculate offset in longitude of ascending node in microarcseconds
    raan_diff = np.abs(diff[:, 5]) * u.degree.to(u.microarcsecond)
    # Assert longitude of ascending node is to within 10 microarcseconds
    np.testing.assert_array_less(raan_diff, 10)

    # Calculate offset in argument of periapsis in microarcseconds
    ap_diff = np.abs(diff[:, 6]) * u.degree.to(u.microarcsecond)
    # Assert argument of periapsis is to within 100 microarcseconds
    np.testing.assert_array_less(ap_diff, 100)  # TODO

    # Calculate offset in mean anomaly in microarcseconds
    M_diff = np.abs(diff[:, 7]) * u.degree.to(u.microarcsecond)
    # Assert mean anomaly is to within 100 microarcseconds
    np.testing.assert_array_less(M_diff, 100)  # TODO

    # Calculate offset in true anomaly in microarcseconds
    nu_diff = np.abs(diff[:, 8]) * u.degree.to(u.microarcsecond)
    # Assert true anomaly is to within 100 microarcseconds
    np.testing.assert_array_less(nu_diff, 100)  # TODO

    # Calculate offset in mean motion in microarcseconds per day
    n_diff = np.abs(diff[:, 9]) * (u.degree / u.d).to(u.microarcsecond / u.d)
    # Assert mean motion is to within 1 microarcseconds per day
    np.testing.assert_array_less(n_diff, 1)

    # Period should be infinite
    assert np.all(np.isinf(keplerian_elements_actual[:, 10]))

    # Calculate offset in time of perihelion passage in seconds
    tp_diff = np.abs(diff[:, 11]) * u.d.to(u.s)
    # Assert time of perihelion passage is to within 1 second
    np.testing.assert_array_less(tp_diff, 1)

    # TODO: All TODOs should be checked to see if they are reasonable tolerances.
    #  I'd expect the semi-major axis to be much better.
