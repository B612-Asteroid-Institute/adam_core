import numpy as np
import spiceypy as sp
from astropy import units as u

from ..origin import Origin
from ..transform import cartesian_to_keplerian, keplerian_to_cartesian


def test_keplerian_to_cartesian_elliptical_against_spice(orbital_elements):
    # Test keplerian_to_cartesian (vmapped) against the cartesian elements
    # calculated by SPICE for a series of sample orbital elements.

    # Limit to elliptical orbits
    orbital_elements = orbital_elements[orbital_elements["e"] < 1]

    keplerian_elements = orbital_elements[["a", "e", "incl", "Omega", "w", "M"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    cartesian_elements_actual = keplerian_to_cartesian(keplerian_elements, origin.mu())

    # Keplerian elements for SPICE are expected to be defined with periapse distance
    # not the semi-major axis
    t0_jd = t0 + 2400000.5
    keplerian_elements_spice = np.empty((len(keplerian_elements), 8))
    keplerian_elements_spice[:, 0] = orbital_elements["q"].values
    keplerian_elements_spice[:, 1] = orbital_elements["e"].values
    keplerian_elements_spice[:, 2] = np.radians(orbital_elements["incl"].values)
    keplerian_elements_spice[:, 3] = np.radians(orbital_elements["Omega"].values)
    keplerian_elements_spice[:, 4] = np.radians(orbital_elements["w"].values)
    keplerian_elements_spice[:, 5] = np.radians(orbital_elements["M"].values)
    keplerian_elements_spice[:, 6] = t0_jd
    keplerian_elements_spice[:, 7] = origin.mu()

    cartesian_elements_spice = np.empty((len(keplerian_elements), 6))
    for i in range(len(keplerian_elements)):
        cartesian_elements_spice[i] = sp.conics(keplerian_elements_spice[i], t0_jd[i])

    diff = cartesian_elements_actual - cartesian_elements_spice

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in nm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.nm / u.s)

    # Assert positions are to within 20 mm
    np.testing.assert_array_less(r_diff, 20)
    # Assert velocities are to within 10 nm/s
    np.testing.assert_array_less(v_diff, 10)


def test_keplerian_to_cartesian_hyperbolic_against_spice(orbital_elements):
    # Test keplerian_to_cartesian (vmapped) against the cartesian elements
    # calculated by SPICE for a series of sample orbital elements.

    # Limit to hyperbolic orbits
    orbital_elements = orbital_elements[orbital_elements["e"] > 1]

    keplerian_elements = orbital_elements[["a", "e", "incl", "Omega", "w", "M"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))

    cartesian_elements_actual = keplerian_to_cartesian(keplerian_elements, origin.mu())

    # Keplerian elements for SPICE are expected to be defined with periapse distance
    # not the semi-major axis
    t0_jd = t0 + 2400000.5
    keplerian_elements_spice = np.empty((len(keplerian_elements), 8))
    keplerian_elements_spice[:, 0] = orbital_elements["q"].values
    keplerian_elements_spice[:, 1] = orbital_elements["e"].values
    keplerian_elements_spice[:, 2] = np.radians(orbital_elements["incl"].values)
    keplerian_elements_spice[:, 3] = np.radians(orbital_elements["Omega"].values)
    keplerian_elements_spice[:, 4] = np.radians(orbital_elements["w"].values)
    keplerian_elements_spice[:, 5] = np.radians(orbital_elements["M"].values)
    keplerian_elements_spice[:, 6] = t0_jd
    keplerian_elements_spice[:, 7] = origin.mu()

    cartesian_elements_spice = np.empty((len(keplerian_elements), 6))
    for i in range(len(keplerian_elements)):
        cartesian_elements_spice[i] = sp.conics(keplerian_elements_spice[i], t0_jd[i])

    diff = cartesian_elements_actual - cartesian_elements_spice

    # Calculate offset in position in mm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.mm)
    # Calculate offset in velocity in nm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.nm / u.s)

    # Assert positions are to within 10 mm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 10 nm/s
    np.testing.assert_array_less(v_diff, 10)


def test_cartesian_to_keplerian_elliptical_against_spice(orbital_elements):
    # Test cartesian_to_keplerian (vmapped) against the keplerian elements
    # calculated by SPICE for a series of sample orbital elements.

    # Limit to elliptical orbits
    orbital_elements = orbital_elements[orbital_elements["e"] < 1]

    t0 = orbital_elements["mjd_tdb"].values
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0_jd = t0 + 2400000.5
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))
    mu = origin.mu()

    # Cartesian to keplerian returns an N, 13 array containing the keplerian elements and cometary
    # elements, with some additional columns.
    keplerian_elements_actual = cartesian_to_keplerian(cartesian_elements, t0, mu)

    # SPICE returns periapse distance instead of semi-major axis so lets test
    # that as well
    q_actual = keplerian_elements_actual[:, 2]
    keplerian_elements_actual = keplerian_elements_actual[:, [0, 4, 5, 6, 7, 8]]

    keplerian_elements_spice = np.empty((len(cartesian_elements), 8))
    for i in range(len(keplerian_elements_spice)):
        keplerian_elements_spice[i] = sp.oscelt(
            np.copy(cartesian_elements[i]), t0_jd[i], mu[i]
        )

    # Keplerian elements from SPICE report periapse distance instead of semi-major axis
    # and angles are in radians instead of degrees, also returned are the gravitational
    # parameter and epoch (so lets ignore those)
    keplerian_elements_spice = keplerian_elements_spice[:, :6]
    q = keplerian_elements_spice[:, 0].copy()
    e = keplerian_elements_spice[:, 1]
    a = q / (1 - e)
    keplerian_elements_spice[:, 0] = a
    keplerian_elements_spice[:, 2:6] = np.degrees(keplerian_elements_spice[:, 2:6])

    # Calculate the difference
    diff = keplerian_elements_actual - keplerian_elements_spice

    # Calculate offset in semi-major axis in mm
    a_diff = np.abs(diff[:, 0]) * u.au.to(u.mm)
    # Assert semi-major axis is to within 10 mm
    np.testing.assert_array_less(a_diff, 10)

    # Calculate the offset in periapse distance in mm
    q_diff = np.abs(q_actual - q) * u.au.to(u.mm)
    # Assert periapse distance is to within 10 mm
    np.testing.assert_array_less(q_diff, 10)

    # Calculate offset in eccentricity
    e_diff = np.abs(diff[:, 1])
    # Assert eccentricity is to within 1e-15
    np.testing.assert_array_less(e_diff, 1e-15)

    # Calculate offset in inclination in nanoarcseconds
    i_diff = np.abs(diff[:, 2]) * u.degree.to(u.nanoarcsecond)
    # Assert inclination is to within 10 nanoarcseconds
    np.testing.assert_array_less(i_diff, 10)

    # Calculate offset in longitude of ascending node in nanoarcseconds
    raan_diff = np.abs(diff[:, 3]) * u.degree.to(u.nanoarcsecond)
    # Assert longitude of ascending node is to within 10 nanoarcseconds
    np.testing.assert_array_less(raan_diff, 10)

    # Calculate offset in argument of periapsis in nanoarcseconds
    ap_diff = np.abs(diff[:, 4]) * u.degree.to(u.nanoarcsecond)
    # Assert argument of periapsis is to within 10 nanoarcseconds
    np.testing.assert_array_less(ap_diff, 10)

    # Calculate offset in mean anomaly in nanoarcseconds
    M_diff = np.abs(diff[:, 5]) * u.degree.to(u.nanoarcsecond)
    # Assert mean anomaly is to within 10 nanoarcseconds
    np.testing.assert_array_less(M_diff, 10)


def test_cartesian_to_keplerian_hyperbolic_against_spice(orbital_elements):
    # Test cartesian_to_keplerian (vmapped) against the keplerian elements
    # calculated by SPICE for a series of sample orbital elements.

    # Limit to hyperbolic orbits
    orbital_elements = orbital_elements[orbital_elements["e"] > 1]

    t0 = orbital_elements["mjd_tdb"].values
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0_jd = t0 + 2400000.5
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))
    mu = origin.mu()

    # Cartesian to keplerian returns an N, 13 array containing the keplerian elements and cometary
    # elements, with some additional columns.
    keplerian_elements_actual = cartesian_to_keplerian(cartesian_elements, t0, mu)

    # SPICE returns periapse distance instead of semi-major axis so lets test
    # that as well
    q_actual = keplerian_elements_actual[:, 2]
    keplerian_elements_actual = keplerian_elements_actual[:, [0, 4, 5, 6, 7, 8]]

    keplerian_elements_spice = np.empty((len(cartesian_elements), 8))
    for i in range(len(keplerian_elements_spice)):
        keplerian_elements_spice[i] = sp.oscelt(
            np.copy(cartesian_elements[i]), t0_jd[i], mu[i]
        )

    # Keplerian elements from SPICE report periapse distance instead of semi-major axis
    # and angles are in radians instead of degrees, also returned are the gravitational
    # parameter and epoch (so lets ignore those)
    keplerian_elements_spice = keplerian_elements_spice[:, :6]
    q = keplerian_elements_spice[:, 0].copy()
    e = keplerian_elements_spice[:, 1]
    a = q / (1 - e)
    keplerian_elements_spice[:, 0] = a
    keplerian_elements_spice[:, 2:6] = np.degrees(keplerian_elements_spice[:, 2:6])

    # Calculate the difference
    diff = keplerian_elements_actual - keplerian_elements_spice

    # Calculate offset in semi-major axis in mm
    a_diff = np.abs(diff[:, 0]) * u.au.to(u.mm)
    # Assert semi-major axis is to within 10 mm
    np.testing.assert_array_less(a_diff, 10)

    # Calculate the offset in periapse distance in mm
    q_diff = np.abs(q_actual - q) * u.au.to(u.mm)
    # Assert periapse distance is to within 10 mm
    np.testing.assert_array_less(q_diff, 10)

    # Calculate offset in eccentricity
    e_diff = np.abs(diff[:, 1])
    # Assert eccentricity is to within 1e-15
    np.testing.assert_array_less(e_diff, 1e-15)

    # Calculate offset in inclination in nanoarcseconds
    i_diff = np.abs(diff[:, 2]) * u.degree.to(u.nanoarcsecond)
    # Assert inclination is to within 10 nanoarcseconds
    np.testing.assert_array_less(i_diff, 10)

    # Calculate offset in longitude of ascending node in nanoarcseconds
    raan_diff = np.abs(diff[:, 3]) * u.degree.to(u.nanoarcsecond)
    # Assert longitude of ascending node is to within 10 nanoarcseconds
    np.testing.assert_array_less(raan_diff, 10)

    # Calculate offset in argument of periapsis in nanoarcseconds
    ap_diff = np.abs(diff[:, 4]) * u.degree.to(u.nanoarcsecond)
    # Assert argument of periapsis is to within 10 nanoarcseconds
    np.testing.assert_array_less(ap_diff, 10)

    # Calculate offset in mean anomaly in nanoarcseconds
    M_diff = np.abs(diff[:, 5]) * u.degree.to(u.nanoarcsecond)
    # Assert mean anomaly is to within 10 nanoarcseconds
    np.testing.assert_array_less(M_diff, 10)
